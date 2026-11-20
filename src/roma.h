#pragma once
#include <opencv2/opencv.hpp>
#include <cstdint>
#include <string>
#include <vector>

struct RomaParameters
{
    int      num_matches         = 10000;
    float    certainty_threshold = 0.05f;
    uint32_t seed                = 42;
};

void detectRoma(const cv::Mat& image1, const cv::Mat& image2, const std::string& model_path,
                const RomaParameters& params,
                std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2,
                std::vector<cv::DMatch>& matches);
