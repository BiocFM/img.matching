// Registration pipeline: detection (SIFT or SuperPoint) -> FLANN + Lowe's
// ratio test, or RoMa (dense matching, matches directly) -> RANSAC homography
// (outlier filter) -> affine on the inliers.

#include "superpoint.h"
#include "roma.h"
#include <Rcpp.h>

#include <chrono>
#include <cmath>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// R <-> OpenCV conversion
// ---------------------------------------------------------------------------

// R matrix -> single-channel CV_32F cv::Mat, values unchanged (scaling happens per detector)
static cv::Mat rmatToMat(const Rcpp::NumericMatrix& m, const char* name)
{
    const int rows = m.nrow(), cols = m.ncol();
    if (rows < 8 || cols < 8) {
        Rcpp::stop(std::string(name) + " must be at least 8 x 8 pixels.");
    }
    cv::Mat out(rows, cols, CV_32F);
    for (int r = 0; r < rows; ++r) {
        float* row = out.ptr<float>(r);
        for (int c = 0; c < cols; ++c) {
            const double v = m(r, c);
            if (!std::isfinite(v)) {
                Rcpp::stop(std::string(name) + " contains NA, NaN or Inf values.");
            }
            row[c] = static_cast<float>(v);
        }
    }
    return out;
}

// Min-max scale to 0..255 and convert to 8 bit).
static cv::Mat toUint8(const cv::Mat& f)
{
    cv::Mat u8;
    cv::normalize(f, u8, 0, 255, cv::NORM_MINMAX, CV_8U);
    return u8;
}

static Rcpp::NumericMatrix matToRmat(const cv::Mat& m)
{
    cv::Mat d;
    m.convertTo(d, CV_64F);
    Rcpp::NumericMatrix out(d.rows, d.cols);
    for (int r = 0; r < d.rows; ++r)
        for (int c = 0; c < d.cols; ++c)
            out(r, c) = d.at<double>(r, c);
    return out;
}

static Rcpp::DataFrame keypointsToDf(const std::vector<cv::KeyPoint>& kp)
{
    const R_xlen_t n = static_cast<R_xlen_t>(kp.size());
    Rcpp::NumericVector x(n), y(n), response(n), size(n);
    for (R_xlen_t i = 0; i < n; ++i) {
        x[i]        = kp[i].pt.x;
        y[i]        = kp[i].pt.y;
        response[i] = kp[i].response;
        size[i]     = kp[i].size;
    }
    return Rcpp::DataFrame::create(Rcpp::Named("x") = x, Rcpp::Named("y") = y,
                                   Rcpp::Named("response") = response,
                                   Rcpp::Named("size") = size);
}

// ---------------------------------------------------------------------------
// Parameters from the R list (with defaults)
// ---------------------------------------------------------------------------

static std::string getString(const Rcpp::List& p, const char* name, const std::string& def)
{
    return p.containsElementNamed(name) ? Rcpp::as<std::string>(p[name]) : def;
}

static double getNumber(const Rcpp::List& p, const char* name, double def)
{
    return p.containsElementNamed(name) ? Rcpp::as<double>(p[name]) : def;
}

// ---------------------------------------------------------------------------
// SIFT
// ---------------------------------------------------------------------------

static void computeSIFT(const cv::Mat& image8u, const SIFTParameters& params,
                        std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(params.sift_nfeatures);
    sift->detectAndCompute(image8u, cv::noArray(), keypoints, descriptors);
}

// ---------------------------------------------------------------------------
// Shared matching + RANSAC
// ---------------------------------------------------------------------------

static void getFLANNMatches(const cv::Mat& descriptors1, const cv::Mat& descriptors2,
                            std::vector<std::vector<cv::DMatch>>& matches)
{
    cv::FlannBasedMatcher matcher(cv::makePtr<cv::flann::KDTreeIndexParams>(5),
                                  cv::makePtr<cv::flann::SearchParams>(50, 0, true));
    matcher.knnMatch(descriptors1, descriptors2, matches, 2);
}

// Lowe's ratio test
static std::vector<cv::DMatch> filterMatchesRatio(
    const std::vector<std::vector<cv::DMatch>>& matches, float lowe_ratio)
{
    std::vector<cv::DMatch> good;
    for (const auto& m : matches) {
        if (m.size() < 2) continue;
        if (m[0].distance < lowe_ratio * m[1].distance) good.push_back(m[0]);
    }
    return good;
}

// RANSAC homography (image1 -> image2) as outlier filter, empty Mat if it fails
static cv::Mat estimateHomographyRANSAC(const std::vector<cv::KeyPoint>& kp1,
                                        const std::vector<cv::KeyPoint>& kp2,
                                        const std::vector<cv::DMatch>& matches,
                                        const SIFTParameters& params,
                                        std::vector<bool>& inlier_flags)
{
    inlier_flags.assign(matches.size(), false);
    if (matches.size() < 4) return cv::Mat();

    std::vector<cv::Point2f> pts1, pts2;
    pts1.reserve(matches.size());
    pts2.reserve(matches.size());
    for (const auto& m : matches) {
        pts1.push_back(kp1[m.queryIdx].pt);
        pts2.push_back(kp2[m.trainIdx].pt);
    }

    std::vector<uchar> mask;
    cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, params.ransac_pixel_threshold,
                                   mask, params.ransac_maxIters, params.ransac_confidence);

    if (H.empty() || mask.size() != matches.size()) return cv::Mat();
    for (size_t i = 0; i < matches.size(); ++i) inlier_flags[i] = mask[i] != 0;
    return H;
}

// Affine transform (2x3, image1 -> image2) fitted on the RANSAC inliers.
static cv::Mat estimateAffineOnInliers(const std::vector<cv::KeyPoint>& kp1,
                                       const std::vector<cv::KeyPoint>& kp2,
                                       const std::vector<cv::DMatch>& matches,
                                       const std::vector<bool>& inlier_flags)
{
    std::vector<cv::Point2f> pts1, pts2;
    for (size_t i = 0; i < matches.size(); ++i) {
        if (!inlier_flags[i]) continue;
        pts1.push_back(kp1[matches[i].queryIdx].pt);
        pts2.push_back(kp2[matches[i].trainIdx].pt);
    }
    if (pts1.size() < 3) return cv::Mat();
    return cv::estimateAffine2D(pts1, pts2);   
}

// ---------------------------------------------------------------------------
// Function called from R
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List registerimages_cpp(Rcpp::NumericMatrix image1,
                              Rcpp::NumericMatrix image2,
                              Rcpp::List parameters)
{
    const auto t0 = std::chrono::steady_clock::now();

    // --- parameters -------------------------------------------------------
    const std::string detection = getString(parameters, "detection", "SIFT");
    const std::string matching  = getString(parameters, "matching", "FLANN");
    const float lowe_ratio      = static_cast<float>(getNumber(parameters, "ratio", 0.8));

    if (matching != "FLANN") {
        Rcpp::stop("matching = '" + matching + "' is not supported (only 'FLANN').");
    }

    SIFTParameters sift_params;
    sift_params.sift_nfeatures = static_cast<int>(getNumber(parameters, "max_keypoints", sift_params.sift_nfeatures));

    SuperPointParameters sp_params;
    sp_params.max_num_keypoints   = static_cast<int>(getNumber(parameters, "max_keypoints", sp_params.max_num_keypoints));
    sp_params.detection_threshold = static_cast<float>(getNumber(parameters, "detection_threshold", sp_params.detection_threshold));
    sp_params.max_pixels          = static_cast<int>(getNumber(parameters, "max_pixels", sp_params.max_pixels));

    RomaParameters roma_params;
    roma_params.num_matches         = static_cast<int>(getNumber(parameters, "num_matches", roma_params.num_matches));
    roma_params.certainty_threshold = static_cast<float>(getNumber(parameters, "certainty_threshold", roma_params.certainty_threshold));
    roma_params.seed                = static_cast<uint32_t>(getNumber(parameters, "seed", roma_params.seed));

    SIFTParameters ransac_params;
    ransac_params.ransac_pixel_threshold = static_cast<int>(getNumber(parameters, "ransac_threshold", ransac_params.ransac_pixel_threshold));
    ransac_params.ransac_confidence      = static_cast<float>(getNumber(parameters, "ransac_confidence", ransac_params.ransac_confidence));
    ransac_params.ransac_maxIters        = static_cast<int>(getNumber(parameters, "ransac_max_iters", ransac_params.ransac_maxIters));

    // --- images -----------------------------------------------------------
    const cv::Mat img1 = rmatToMat(image1, "image1");
    const cv::Mat img2 = rmatToMat(image2, "image2");

    // --- detection (declared here so they exist after the if/else) ----------
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    std::vector<cv::DMatch> good;   // matches for RANSAC

    if (detection == "SIFT") {
        computeSIFT(toUint8(img1), sift_params, kp1, desc1);
        computeSIFT(toUint8(img2), sift_params, kp2, desc2);
    } else if (detection == "SuperPoint") {
        if (!parameters.containsElementNamed("model_path")) {
            Rcpp::stop("parameters$model_path (path to superpoint.pt) is required for SuperPoint.");
        }
        const std::string model_path = Rcpp::as<std::string>(parameters["model_path"]);
        detectSuperPoint(img1, model_path, sp_params, kp1, desc1);
        detectSuperPoint(img2, model_path, sp_params, kp2, desc2);
    } else if (detection == "RoMa") {
        if (!parameters.containsElementNamed("model_path")) {
            Rcpp::stop("parameters$model_path (path to the traced RoMa .pt) is required for RoMa.");
        }
        const std::string model_path = Rcpp::as<std::string>(parameters["model_path"]);
        detectRoma(img1, img2, model_path, roma_params, kp1, kp2, good);
    } else {
        Rcpp::stop("detection = '" + detection + "' is not supported ('SIFT', 'SuperPoint' or 'RoMa').");
    }

    // --- matching (RoMa already returns matches) ----------------------------
    if (detection != "RoMa") {
        if (desc1.rows < 2 || desc2.rows < 2) {
            Rcpp::stop("Too few keypoints found (image1: " + std::to_string(desc1.rows) +
                       ", image2: " + std::to_string(desc2.rows) + "); need at least 2 each.");
        }
        if (desc1.type() != CV_32F) desc1.convertTo(desc1, CV_32F);   // FLANN needs float
        if (desc2.type() != CV_32F) desc2.convertTo(desc2, CV_32F);

        std::vector<std::vector<cv::DMatch>> knn;
        getFLANNMatches(desc1, desc2, knn);
        good = filterMatchesRatio(knn, lowe_ratio);
    }

    // --- RANSAC + affine ----------------------------------------------------
    std::vector<bool> inlier_flags;
    const cv::Mat H = estimateHomographyRANSAC(kp1, kp2, good, ransac_params, inlier_flags);
    const cv::Mat A = H.empty() ? cv::Mat()
                                : estimateAffineOnInliers(kp1, kp2, good, inlier_flags);

    // --- results for R -----------------------------------------------------
    const R_xlen_t n_good = static_cast<R_xlen_t>(good.size());
    Rcpp::IntegerVector idx1(n_good), idx2(n_good);
    Rcpp::NumericVector distance(n_good);
    Rcpp::LogicalVector inlier(n_good);
    int n_inliers = 0;
    for (R_xlen_t i = 0; i < n_good; ++i) {
        idx1[i]     = good[i].queryIdx + 1;   // 1-based rows of keypoints1
        idx2[i]     = good[i].trainIdx + 1;   // 1-based rows of keypoints2
        distance[i] = good[i].distance;       // RoMa: 1 - certainty
        inlier[i]   = inlier_flags[i];
        if (inlier_flags[i]) ++n_inliers;
    }
    Rcpp::DataFrame matches_df = Rcpp::DataFrame::create(
        Rcpp::Named("idx1") = idx1, Rcpp::Named("idx2") = idx2,
        Rcpp::Named("distance") = distance, Rcpp::Named("inlier") = inlier);

    const double runtime = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    Rcpp::List stats = Rcpp::List::create(
        Rcpp::Named("n_keypoints1") = static_cast<int>(kp1.size()),
        Rcpp::Named("n_keypoints2") = static_cast<int>(kp2.size()),
        Rcpp::Named("n_matches")    = static_cast<int>(n_good),
        Rcpp::Named("n_inliers")    = n_inliers,
        Rcpp::Named("inlier_ratio") = n_good > 0 ? static_cast<double>(n_inliers) / n_good : NA_REAL,
        Rcpp::Named("runtime_sec")  = runtime);

    return Rcpp::List::create(
        Rcpp::Named("homography") = H.empty() ? Rcpp::RObject(R_NilValue) : Rcpp::RObject(matToRmat(H)),
        Rcpp::Named("affine")     = A.empty() ? Rcpp::RObject(R_NilValue) : Rcpp::RObject(matToRmat(A)),
        Rcpp::Named("keypoints1") = keypointsToDf(kp1),
        Rcpp::Named("keypoints2") = keypointsToDf(kp2),
        Rcpp::Named("matches")    = matches_df,
        Rcpp::Named("stats")      = stats);
}
