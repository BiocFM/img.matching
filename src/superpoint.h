#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct SuperPointParameters
{
    int max_num_keypoints     = 20000;
    int nms_radius            = 4;      
    float detection_threshold = 0.005f;
    int remove_borders        = 4;      
    int max_pixels            = 30000000;
};

struct SIFTParameters
{ 
  int sift_nfeatures=20000;
  int ransac_pixel_threshold = 5;
  float ransac_confidence = 0.95;
  int ransac_maxIters=2000;
};

void detectSuperPoint(const cv::Mat& image, const std::string& model_path,
                      const SuperPointParameters& params,
                      std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
