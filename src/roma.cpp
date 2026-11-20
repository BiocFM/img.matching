#define LANTERN_HEADERS_ONLY
#include <lantern/lantern.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "roma.h"

namespace {

// Input sizes fixed by the trace (coarse pass, refinement pass)
const int kCoarseSize = 560;
const int kUpsampleSize = 864;
const float kImagenetMean[3] = {0.485f, 0.456f, 0.406f};
const float kImagenetStd[3]  = {0.229f, 0.224f, 0.225f};

// ---------------------------------------------------------------------------
// RAII wrapper so every Lantern object is freed even if a call throws
// ---------------------------------------------------------------------------

class LPtr {
 public:
  LPtr(void* p, void (*del)(void*)) : p_(p), del_(del) {}
  ~LPtr() { if (p_ != nullptr) del_(p_); }
  LPtr(const LPtr&) = delete;
  LPtr& operator=(const LPtr&) = delete;
  void* get() const { return p_; }
 private:
  void* p_;
  void (*del_)(void*);
};

void del_tensor(void* x)       { lantern_Tensor_delete(x); }
void del_ivalue(void* x)       { lantern_IValue_delete(x); }
void del_stack(void* x)        { lantern_Stack_delete(x); }
void del_string(void* x)       { lantern_string_delete(x); }
void del_options(void* x)      { lantern_TensorOptions_delete(x); }
void del_dtype(void* x)        { lantern_Dtype_delete(x); }
void del_device(void* x)       { lantern_Device_delete(x); }
void del_method(void* x)       { lantern_jit_ScriptMethod_delete(x); }
void del_ivalue_vec(void* x)   { lantern_jit_vector_IValue_delete(x); }
void del_tuple_helper(void* x) { lantern_NamedTupleHelper_delete(x); }

// ---------------------------------------------------------------------------
// Cached model (loaded once per path, reused across calls)
// ---------------------------------------------------------------------------

void* g_module = nullptr;
std::string g_model_path;

void* get_module(const std::string& model_path) {
  if (g_module != nullptr && g_model_path == model_path) return g_module;
  if (g_module != nullptr) {
    lantern_JITModule_delete(g_module);
    g_module = nullptr;
  }
  g_module = lantern_jit_load(model_path.c_str());
  lantern_ScriptModule_train(g_module, false);
  g_model_path = model_path;
  return g_module;
}

// ---------------------------------------------------------------------------
// Image preparation
// ---------------------------------------------------------------------------

// Grayscale, min-max scaled to [0, 1], resized to size x size,
// copied into 3 channels and ImageNet-normalized (CHW buffer).
std::vector<float> to_roma_input(const cv::Mat& image, int size) {
  if (image.empty()) throw std::runtime_error("Input image is empty.");

  cv::Mat gray;
  if (image.channels() == 3)      cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  else if (image.channels() == 4) cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
  else if (image.channels() == 1) gray = image;
  else throw std::runtime_error("Input image must have 1, 3 or 4 channels.");

  cv::Mat f;
  gray.convertTo(f, CV_32F);

  //set min to 0, max to 1
  double mn = 0.0, mx = 0.0;
  cv::minMaxLoc(f, &mn, &mx);
  if (mx > mn) f = (f - mn) / (mx - mn);
  else f.setTo(0.0f);

  //INTER_AREA for shrinking (antialiased like PIL), INTER_CUBIC for enlarging
  const bool shrink = f.rows > size || f.cols > size;
  cv::Mat resized;
  cv::resize(f, resized, cv::Size(size, size), 0, 0, shrink ? cv::INTER_AREA : cv::INTER_CUBIC);

  std::vector<float> chw(static_cast<size_t>(3) * size * size);
  for (int c = 0; c < 3; ++c) {
    float* out = chw.data() + static_cast<size_t>(c) * size * size;
    for (int y = 0; y < size; ++y) {
      const float* row = resized.ptr<float>(y);
      for (int x = 0; x < size; ++x) {
        out[static_cast<size_t>(y) * size + x] = (row[x] - kImagenetMean[c]) / kImagenetStd[c];
      }
    }
  }
  return chw;
}

// ---------------------------------------------------------------------------
// Output parsing
// ---------------------------------------------------------------------------

// Host-side copy of one output tensor (float64, CPU, contiguous).
std::vector<double> to_host(void* tensor, std::vector<int64_t>& shape) {
  LPtr opts0(lantern_TensorOptions(), del_options);
  LPtr dtype(lantern_Dtype_float64(), del_dtype);
  LPtr opts1(lantern_TensorOptions_dtype(opts0.get(), dtype.get()), del_options);
  LPtr cpu(lantern_Device("cpu", 0, false), del_device);
  LPtr opts(lantern_TensorOptions_device(opts1.get(), cpu.get()), del_options);
  LPtr converted(lantern_Tensor_to(tensor, opts.get()), del_tensor);
  LPtr contig(lantern_Tensor_contiguous(converted.get()), del_tensor);

  shape.clear();
  const int64_t ndim = lantern_Tensor_ndimension(contig.get());
  for (int64_t i = 0; i < ndim; ++i) shape.push_back(lantern_Tensor_size(contig.get(), i));

  const int64_t n = lantern_Tensor_numel(contig.get());
  const double* p = lantern_Tensor_data_ptr_double(contig.get());
  return std::vector<double>(p, p + n);
}

// ---------------------------------------------------------------------------
// Sampling (same steps as RoMa's sample(), "threshold_balanced")
// ---------------------------------------------------------------------------

// Weighted draw without replacement, like torch.multinomial. Zero weights are never drawn.
std::vector<int64_t> weighted_sample(const std::vector<double>& weights,
                                     const std::vector<int64_t>& candidates,
                                     size_t k, std::mt19937& rng) {
  std::uniform_real_distribution<double> uni(std::nextafter(0.0, 1.0), 1.0);
  std::vector<std::pair<double, int64_t>> keyed;   // (log(u) / w, index)
  keyed.reserve(candidates.size());
  for (size_t i = 0; i < candidates.size(); ++i) {
    const double w = weights[i];
    if (w > 0.0) keyed.emplace_back(std::log(uni(rng)) / w, candidates[i]);
  }
  k = std::min(k, keyed.size());
  std::partial_sort(keyed.begin(), keyed.begin() + static_cast<std::ptrdiff_t>(k), keyed.end(),
                    [](const std::pair<double, int64_t>& a, const std::pair<double, int64_t>& b) {
                      return a.first > b.first;
                    });
  std::vector<int64_t> picked(k);
  for (size_t i = 0; i < k; ++i) picked[i] = keyed[i].second;
  return picked;
}

// RoMa's kde() in 4-D (xA, yA, xB, yB); pairs further than 6 std apart are skipped.
std::vector<double> kde_density(const std::vector<double>& warp, const std::vector<int64_t>& idx,
                                double std_dev) {
  const size_t n = idx.size();
  const double inv_two_var = 1.0 / (2.0 * std_dev * std_dev);
  const double cutoff_sq = 36.0 * std_dev * std_dev;
  std::vector<double> density(n, 1.0);
  for (size_t i = 0; i < n; ++i) {
    const double* a = &warp[static_cast<size_t>(idx[i]) * 4];
    for (size_t j = i + 1; j < n; ++j) {
      const double* b = &warp[static_cast<size_t>(idx[j]) * 4];
      const double d0 = a[0] - b[0], d1 = a[1] - b[1];
      double d_sq = d0 * d0 + d1 * d1;
      if (d_sq > cutoff_sq) continue;
      const double d2 = a[2] - b[2], d3 = a[3] - b[3];
      d_sq += d2 * d2 + d3 * d3;
      if (d_sq > cutoff_sq) continue;
      const double k = std::exp(-d_sq * inv_two_var);
      density[i] += k;
      density[j] += k;
    }
  }
  return density;
}

// Returns indices into the H * 2W warp positions.
std::vector<int64_t> sample_matches(const std::vector<double>& warp, const std::vector<double>& certainty,
                                    const RomaParameters& params) {
  const size_t m = certainty.size();
  std::mt19937 rng(params.seed);

  //certainty above the threshold counts as 1, draw 4 x num candidates
  std::vector<double> w(m);
  std::vector<int64_t> all(m);
  for (size_t i = 0; i < m; ++i) {
    w[i] = certainty[i] > params.certainty_threshold ? 1.0 : certainty[i];
    all[i] = static_cast<int64_t>(i);
  }
  const size_t num = static_cast<size_t>(std::max(params.num_matches, 0));
  std::vector<int64_t> good = weighted_sample(w, all, std::min(4 * num, m), rng);

  //prefer sparse regions, drop isolated candidates (density < 10)
  const std::vector<double> density = kde_density(warp, good, 0.1);
  std::vector<double> p(good.size());
  for (size_t i = 0; i < good.size(); ++i) {
    p[i] = density[i] < 10.0 ? 1e-7 : 1.0 / (density[i] + 1.0);
  }
  return weighted_sample(p, good, std::min(num, good.size()), rng);
}

// Runs forward(im_A_lr, im_B_lr, im_A_hr, im_B_hr) -> (warp, certainty).
void run_roma(void* module, const std::vector<void*>& inputs,
              std::vector<double>& warp, std::vector<int64_t>& warp_shape,
              std::vector<double>& certainty, std::vector<int64_t>& cert_shape) {
  LPtr name(lantern_string_new("forward", 7), del_string);
  void* method_raw = lantern_ScriptModule_find_method(module, name.get());
  if (method_raw == nullptr) throw std::runtime_error("The RoMa model has no forward() method.");
  LPtr method(method_raw, del_method);

  LPtr stack(lantern_Stack_new(), del_stack);
  for (void* t : inputs) {
    LPtr arg(lantern_IValue_from_Tensor(t), del_ivalue);
    lantern_Stack_push_back_IValue(stack.get(), arg.get());
  }
  LPtr out_stack(lantern_ScriptMethod_call(method.get(), stack.get()), del_stack);
  if (lantern_Stack_size(out_stack.get()) < 1) throw std::runtime_error("RoMa forward() returned nothing.");
  LPtr out(lantern_Stack_at(out_stack.get(), 0), del_ivalue);

  if (lantern_IValue_type(out.get()) != IValueTupleType) {
    throw std::runtime_error("RoMa forward() must return a (warp, certainty) tuple.");
  }
  LPtr helper(lantern_IValue_Tuple(out.get()), del_tuple_helper);
  LPtr elems(lantern_jit_NamedTupleHelper_elements(helper.get()), del_ivalue_vec);
  if (lantern_jit_vector_IValue_size(elems.get()) < 2) {
    throw std::runtime_error("RoMa forward() must return a (warp, certainty) tuple.");
  }
  LPtr warp_iv(lantern_jit_vector_IValue_at(elems.get(), 0), del_ivalue);
  LPtr cert_iv(lantern_jit_vector_IValue_at(elems.get(), 1), del_ivalue);
  LPtr warp_t(lantern_IValue_Tensor(warp_iv.get()), del_tensor);
  LPtr cert_t(lantern_IValue_Tensor(cert_iv.get()), del_tensor);
  warp = to_host(warp_t.get(), warp_shape);
  certainty = to_host(cert_t.get(), cert_shape);
}

// Restores grad mode on scope exit, even if inference throws.
class NoGradGuard {
 public:
  NoGradGuard() : prev_(lantern_autograd_is_enabled()) { lantern_autograd_set_grad_mode(false); }
  ~NoGradGuard() {
    try { lantern_autograd_set_grad_mode(prev_); } catch (...) {}
  }
 private:
  bool prev_;
};

}  // namespace

// ---------------------------------------------------------------------------
// C++ API
// ---------------------------------------------------------------------------

void detectRoma(const cv::Mat& image1, const cv::Mat& image2, const std::string& model_path,
                const RomaParameters& params,
                std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2,
                std::vector<cv::DMatch>& matches) {
  check_lantern_loaded();
  keypoints1.clear();
  keypoints2.clear();
  matches.clear();

  // must outlive the tensors below
  std::vector<float> a_lr = to_roma_input(image1, kCoarseSize);
  std::vector<float> b_lr = to_roma_input(image2, kCoarseSize);
  std::vector<float> a_hr = to_roma_input(image1, kUpsampleSize);
  std::vector<float> b_hr = to_roma_input(image2, kUpsampleSize);

  void* module = get_module(model_path);
  NoGradGuard no_grad;

  LPtr opts0(lantern_TensorOptions(), del_options);
  LPtr f32(lantern_Dtype_float32(), del_dtype);
  LPtr opts(lantern_TensorOptions_dtype(opts0.get(), f32.get()), del_options);
  int64_t lr_sizes[4] = {1, 3, kCoarseSize, kCoarseSize};
  int64_t hr_sizes[4] = {1, 3, kUpsampleSize, kUpsampleSize};
  LPtr t_a_lr(lantern_from_blob(a_lr.data(), lr_sizes, 4, nullptr, 0, opts.get()), del_tensor);
  LPtr t_b_lr(lantern_from_blob(b_lr.data(), lr_sizes, 4, nullptr, 0, opts.get()), del_tensor);
  LPtr t_a_hr(lantern_from_blob(a_hr.data(), hr_sizes, 4, nullptr, 0, opts.get()), del_tensor);
  LPtr t_b_hr(lantern_from_blob(b_hr.data(), hr_sizes, 4, nullptr, 0, opts.get()), del_tensor);

  std::vector<double> warp, certainty;
  std::vector<int64_t> warp_shape, cert_shape;
  run_roma(module, {t_a_lr.get(), t_b_lr.get(), t_a_hr.get(), t_b_hr.get()},
           warp, warp_shape, certainty, cert_shape);

  // warp: [1, H, 2W, 4] as (xA, yA, xB, yB) in [-1, 1]; certainty: [1, H, 2W]
  if (warp_shape.size() != 4 || warp_shape[3] != 4 ||
      cert_shape.size() != 3 || warp_shape[1] * warp_shape[2] != cert_shape[1] * cert_shape[2]) {
    throw std::runtime_error("Unexpected RoMa output shapes.");
  }

  const std::vector<int64_t> picked = sample_matches(warp, certainty, params);

  // [-1, 1] -> pixels; -0.5 so pixel centres are at i (OpenCV), like SIFT/SuperPoint
  const double wA = image1.cols, hA = image1.rows, wB = image2.cols, hB = image2.rows;
  keypoints1.reserve(picked.size());
  keypoints2.reserve(picked.size());
  matches.reserve(picked.size());
  for (size_t i = 0; i < picked.size(); ++i) {
    const double* m = &warp[static_cast<size_t>(picked[i]) * 4];
    const float cert = static_cast<float>(certainty[static_cast<size_t>(picked[i])]);
    cv::KeyPoint k1, k2;
    k1.pt = cv::Point2f(static_cast<float>(wA / 2.0 * (m[0] + 1.0) - 0.5),
                        static_cast<float>(hA / 2.0 * (m[1] + 1.0) - 0.5));
    k2.pt = cv::Point2f(static_cast<float>(wB / 2.0 * (m[2] + 1.0) - 0.5),
                        static_cast<float>(hB / 2.0 * (m[3] + 1.0) - 0.5));
    k1.response = k2.response = cert;
    k1.size = k2.size = 1.0f;   // nominal, RoMa has no keypoint scale
    keypoints1.push_back(k1);
    keypoints2.push_back(k2);
    const int idx = static_cast<int>(i);
    matches.emplace_back(idx, idx, 1.0f - cert);
  }
}
