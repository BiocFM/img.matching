#define LANTERN_HEADERS_ONLY
#include <lantern/lantern.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "superpoint.h"

// ---------------------------------------------------------------------------
// RAII wrapper so every Lantern object is freed even if a call throws
// ---------------------------------------------------------------------------

namespace {

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
void del_tensor_list(void* x)  { lantern_TensorList_delete(x); }
void del_ivalue(void* x)       { lantern_IValue_delete(x); }
void del_stack(void* x)        { lantern_Stack_delete(x); }
void del_string(void* x)       { lantern_string_delete(x); }
void del_options(void* x)      { lantern_TensorOptions_delete(x); }
void del_dtype(void* x)        { lantern_Dtype_delete(x); }
void del_device(void* x)       { lantern_Device_delete(x); }
void del_method(void* x)       { lantern_jit_ScriptMethod_delete(x); }
void del_tensor_dict(void* x)  { lantern_jit_TensorDict_delete(x); }
void del_generic_dict(void* x) { lantern_jit_GenericDict_delete(x); }
void del_generic_list(void* x) { lantern_jit_GenericList_delete(x); }
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
  lantern_ScriptModule_train(g_module, false);  // eval mode
  g_model_path = model_path;
  return g_module;
}

// ---------------------------------------------------------------------------
// Output parsing
// ---------------------------------------------------------------------------

// Host-side copy of one output tensor (float64, CPU, contiguous).
struct HostTensor {
  std::vector<int64_t> shape;  
  std::vector<double> data;
};

HostTensor to_host(void* tensor) {
  LPtr opts0(lantern_TensorOptions(), del_options);
  LPtr dtype(lantern_Dtype_float64(), del_dtype);
  LPtr opts1(lantern_TensorOptions_dtype(opts0.get(), dtype.get()), del_options);
  LPtr cpu(lantern_Device("cpu", 0, false), del_device);
  LPtr opts(lantern_TensorOptions_device(opts1.get(), cpu.get()), del_options);

  LPtr converted(lantern_Tensor_to(tensor, opts.get()), del_tensor);
  LPtr contig(lantern_Tensor_contiguous(converted.get()), del_tensor);

  HostTensor out;
  const int64_t ndim = lantern_Tensor_ndimension(contig.get());
  for (int64_t i = 0; i < ndim; ++i) {
    out.shape.push_back(lantern_Tensor_size(contig.get(), i));
  }
  // Drop leading batch dims of size 1 ([1, N, 2] -> [N, 2]), keep at least 1 dim.
  while (out.shape.size() > 1 && out.shape.front() == 1) {
    out.shape.erase(out.shape.begin());
  }

  const int64_t n = lantern_Tensor_numel(contig.get());
  const double* p = lantern_Tensor_data_ptr_double(contig.get());
  out.data.assign(p, p + n);
  return out;
}

// First tensor inside an IValue: a Tensor itself, or element 0 of 
// the per-sample lists the scripted model returns.
void* first_tensor(void* ivalue) {
  switch (lantern_IValue_type(ivalue)) {
    case IValueTensorType:
      return lantern_IValue_Tensor(ivalue);

    case IValueTensorListType: {
      LPtr list(lantern_IValue_TensorList(ivalue), del_tensor_list);
      if (lantern_TensorList_size(list.get()) < 1) {
        throw std::runtime_error("SuperPoint returned an empty tensor list.");
      }
      return lantern_TensorList_at(list.get(), 0);
    }

    case IValueListType: {
      LPtr list(lantern_IValue_List(ivalue), del_generic_list);
      if (lantern_jit_GenericList_size(list.get()) < 1) {
        throw std::runtime_error("SuperPoint returned an empty list.");
      }
      LPtr el(lantern_jit_GenericList_at(list.get(), 0), del_ivalue);
      return first_tensor(el.get());
    }

    case IValueTupleType: {
      LPtr helper(lantern_IValue_Tuple(ivalue), del_tuple_helper);
      LPtr elems(lantern_jit_NamedTupleHelper_elements(helper.get()), del_ivalue_vec);
      if (lantern_jit_vector_IValue_size(elems.get()) < 1) {
        throw std::runtime_error("SuperPoint returned an empty tuple.");
      }
      LPtr el(lantern_jit_vector_IValue_at(elems.get(), 0), del_ivalue);
      return first_tensor(el.get());
    }

    default:
      throw std::runtime_error(
          "Unexpected SuperPoint output: expected a tensor or a list/tuple of tensors.");
  }
}

struct RawOutputs {
  HostTensor keypoints, scores, descriptors;
};

// Reads the three outputs from either a dict, a list/tuple or 
// a length-1 list / per sample wrapper
RawOutputs parse_outputs(void* ivalue) {
  const int type = lantern_IValue_type(ivalue);

  if (type == IValueGenericDictType) {
    LPtr dict(lantern_IValue_GenericDict(ivalue), del_generic_dict);
    LPtr keys(lantern_jit_GenericDict_keys(dict.get()), del_ivalue_vec);

    void* kp = nullptr; void* sc = nullptr; void* de = nullptr;
    const int64_t n_keys = lantern_jit_vector_IValue_size(keys.get());
    for (int64_t i = 0; i < n_keys; ++i) {
      LPtr key(lantern_jit_vector_IValue_at(keys.get(), i), del_ivalue);
      if (lantern_IValue_type(key.get()) != IValueStringType) continue;
      LPtr key_str(lantern_IValue_String(key.get()), del_string);
      char* raw = lantern_string_get(key_str.get());
      const std::string name(raw, lantern_string_size(key_str.get()));  
      lantern_const_char_delete(raw);                                    // free Lantern's buffer

      void** slot = nullptr;
      if (name == "keypoints") slot = &kp;
      else if (name == "keypoint_scores" || name == "scores") slot = &sc;
      else if (name == "descriptors") slot = &de;
      if (slot == nullptr) continue;

      LPtr val(lantern_jit_GenericDict_at(dict.get(), key.get()), del_ivalue);
      *slot = first_tensor(val.get());
    }
    LPtr kp_(kp, del_tensor), sc_(sc, del_tensor), de_(de, del_tensor);
    if (kp == nullptr || sc == nullptr || de == nullptr) {
      throw std::runtime_error(
          "SuperPoint dict output is missing 'keypoints', 'keypoint_scores' or 'descriptors'.");
    }
    return RawOutputs{to_host(kp), to_host(sc), to_host(de)};
  }

  
  std::vector<void*> elems;
  std::vector<LPtr*> owners;  
  auto cleanup = [&]() { for (LPtr* o : owners) delete o; };

  try {
    if (type == IValueTupleType) {
      LPtr helper(lantern_IValue_Tuple(ivalue), del_tuple_helper);
      LPtr vec(lantern_jit_NamedTupleHelper_elements(helper.get()), del_ivalue_vec);
      const int64_t n = lantern_jit_vector_IValue_size(vec.get());
      for (int64_t i = 0; i < n; ++i) {
        owners.push_back(new LPtr(lantern_jit_vector_IValue_at(vec.get(), i), del_ivalue));
        elems.push_back(owners.back()->get());
      }
    } else if (type == IValueListType) {
      LPtr list(lantern_IValue_List(ivalue), del_generic_list);
      const int64_t n = lantern_jit_GenericList_size(list.get());
      for (int64_t i = 0; i < n; ++i) {
        owners.push_back(new LPtr(lantern_jit_GenericList_at(list.get(), i), del_ivalue));
        elems.push_back(owners.back()->get());
      }
    } else {
      throw std::runtime_error(
          "Unexpected SuperPoint output type: expected a dict, tuple or list.");
    }

    RawOutputs out;
    if (elems.size() == 1) {
      out = parse_outputs(elems[0]);  // per-sample wrapper, batch of 1
    } else if (elems.size() >= 3) {
      LPtr kp(first_tensor(elems[0]), del_tensor);
      LPtr sc(first_tensor(elems[1]), del_tensor);
      LPtr de(first_tensor(elems[2]), del_tensor);
      out = RawOutputs{to_host(kp.get()), to_host(sc.get()), to_host(de.get())};
    } else {
      throw std::runtime_error(
          "SuperPoint returned " + std::to_string(elems.size()) +
          " outputs; expected (keypoints, scores, descriptors).");
    }
    cleanup();
    return out;
  } catch (...) {
    cleanup();
    throw;
  }
}

// ---------------------------------------------------------------------------
// Image preparation
// ---------------------------------------------------------------------------

// Grayscale, float32, min-max scaled to [0, 1].
// Works for 8-bit, 16-bit and float input. No resizing here: preprocessing
cv::Mat to_superpoint_input(const cv::Mat& image) {
  if (image.empty()) throw std::runtime_error("Input image is empty.");

  cv::Mat gray;
  if (image.channels() == 3)      cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  else if (image.channels() == 4) cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
  else if (image.channels() == 1) gray = image;
  else throw std::runtime_error("Input image must have 1, 3 or 4 channels.");

  cv::Mat f;
  gray.convertTo(f, CV_32F);

  //make divisible by stride, cut pixels from bottom/right to not alter the coordinates
  const int stride = 8;
  const int h = (f.rows / stride) * stride;
  const int w = (f.cols / stride) * stride;
  if (h == 0 || w == 0) throw std::runtime_error("Image is smaller than 8x8 pixels.");
  f = f(cv::Rect(0, 0, w, h));
  
  //set min to 0, max to 1
  double mn = 0.0, mx = 0.0;
  cv::minMaxLoc(f, &mn, &mx);
  if (mx > mn) f = (f - mn) / (mx - mn);
  else f.setTo(0.0f);

  return f.isContinuous() ? f : f.clone();
}

// Runs forward() with the image as a [1,1,H,W] tensor. If the scripted
// forward takes a dict instead (LightGlue-style forward(data)), retries with
// {"image": tensor}.
void* call_forward(void* module, void* image_tensor) {
  LPtr name(lantern_string_new("forward", 7), del_string);
  void* method_raw = lantern_ScriptModule_find_method(module, name.get());
  if (method_raw == nullptr) {
    throw std::runtime_error("The SuperPoint model has no forward() method.");
  }
  LPtr method(method_raw, del_method);

  try {
    LPtr inputs(lantern_Stack_new(), del_stack);
    LPtr arg(lantern_IValue_from_Tensor(image_tensor), del_ivalue);
    lantern_Stack_push_back_IValue(inputs.get(), arg.get());
    return lantern_ScriptMethod_call(method.get(), inputs.get());
  } catch (const std::exception& tensor_err) {
    try {
      LPtr dict(lantern_jit_TensorDict_new(), del_tensor_dict);
      LPtr key(lantern_string_new("image", 5), del_string);
      lantern_jit_TensorDict_push_back(dict.get(), key.get(), image_tensor);

      LPtr inputs(lantern_Stack_new(), del_stack);
      LPtr arg(lantern_IValue_from_TensorDict(dict.get()), del_ivalue);
      lantern_Stack_push_back_IValue(inputs.get(), arg.get());
      return lantern_ScriptMethod_call(method.get(), inputs.get());
    } catch (const std::exception&) {
      throw std::runtime_error(std::string("SuperPoint forward() failed: ") +
                               tensor_err.what());
    }
  }
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

// nms_radius and remove_borders are defined in superpoint.pt when it is
// scripted; they cannot be changed from here. max_pixels, detection_threshold
// and max_num_keypoints are applied below
void detectSuperPoint(const cv::Mat& image, const std::string& model_path,
                      const SuperPointParameters& params,
                      std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
  check_lantern_loaded();  
  keypoints.clear();
  descriptors.release();

  const double n_pixels = static_cast<double>(image.rows) * image.cols;
  if (n_pixels > params.max_pixels) {
    throw std::runtime_error(
        "Image has " + std::to_string(static_cast<long long>(n_pixels)) +
        " pixels; SuperPoint max_pixels is " + std::to_string(params.max_pixels) +
        ". Downscale it first.");
  }

  cv::Mat input = to_superpoint_input(image);  // must outlive the tensor below
  void* module = get_module(model_path);

  NoGradGuard no_grad;

  // Wrap the cv::Mat buffer as a [1, 1, H, W] float32 tensor (no copy).
  int64_t sizes[4] = {1, 1, input.rows, input.cols};
  LPtr opts0(lantern_TensorOptions(), del_options);
  LPtr f32(lantern_Dtype_float32(), del_dtype);
  LPtr opts(lantern_TensorOptions_dtype(opts0.get(), f32.get()), del_options);
  LPtr tensor(lantern_from_blob(input.ptr<float>(), sizes, 4, nullptr, 0, opts.get()),
              del_tensor);

  LPtr out_stack(call_forward(module, tensor.get()), del_stack);
  if (lantern_Stack_size(out_stack.get()) < 1) {
    throw std::runtime_error("SuperPoint forward() returned nothing.");
  }
  LPtr out(lantern_Stack_at(out_stack.get(), 0), del_ivalue);
  RawOutputs raw = parse_outputs(out.get());

  // --- keypoints: [N, 2] as (x, y) ---------------------------------------
  const HostTensor& kp = raw.keypoints;
  if (kp.shape.back() != 2 || kp.data.size() % 2 != 0) {
    throw std::runtime_error("SuperPoint keypoints must have shape [N, 2].");
  }
  const int n = static_cast<int>(kp.data.size() / 2);

  // --- scores: [N] ---------------------------------------------------------
  if (static_cast<int>(raw.scores.data.size()) != n) {
    throw std::runtime_error("SuperPoint scores do not match the number of keypoints.");
  }

  // --- descriptors: [N, D] (transpose if the model returns [D, N]) ----------
  const HostTensor& de = raw.descriptors;
  if (de.shape.size() < 2 && n > 1) {
    throw std::runtime_error("SuperPoint descriptors must be 2-D.");
  }
  const int64_t r = de.shape.size() >= 2 ? de.shape[de.shape.size() - 2] : 1;
  const int64_t c = de.shape.back();
  const bool transposed = (r != n && c == n);
  const int d = static_cast<int>(transposed ? r : c);
  if ((transposed ? c : r) != n) {
    throw std::runtime_error("SuperPoint descriptors do not match the number of keypoints.");
  }

  // Keep keypoints above the threshold, strongest first, capped at max_num_keypoints.
  std::vector<int> order;
  order.reserve(n);
  for (int i = 0; i < n; ++i) {
    if (raw.scores.data[i] >= params.detection_threshold) order.push_back(i);
  }
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return raw.scores.data[a] > raw.scores.data[b];
  });
  if (params.max_num_keypoints > 0 &&
      static_cast<int>(order.size()) > params.max_num_keypoints) {
    order.resize(params.max_num_keypoints);
  }

  const int m = static_cast<int>(order.size());
  keypoints.reserve(m);
  descriptors.create(m, d, CV_32F);  // CV_32F rows: ready for FLANN

  for (int r_out = 0; r_out < m; ++r_out) {
    const int i = order[r_out];
    cv::KeyPoint k;
    k.pt = cv::Point2f(static_cast<float>(kp.data[2 * i]),
                       static_cast<float>(kp.data[2 * i + 1]));
    k.response = static_cast<float>(raw.scores.data[i]);
    k.size = 8.0f;  // nominal (SuperPoint's 8x8 cell); not used for matching
    keypoints.push_back(k);

    float* row = descriptors.ptr<float>(r_out);
    for (int j = 0; j < d; ++j) {
      row[j] = static_cast<float>(transposed ? de.data[static_cast<size_t>(j) * n + i]
                                             : de.data[static_cast<size_t>(i) * d + j]);
    }
  }
}