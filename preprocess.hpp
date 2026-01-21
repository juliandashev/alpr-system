#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace preprocessing {
    constexpr int GAUSSIAN_FILTER_SIZE = 5;
    constexpr int ADAPTIVE_THRESH_BLOCK_SIZE = 19;
    constexpr int ADAPTIVE_THRESH_WEIGHT = 9;
}

void preprocess(cv::Mat& original, cv::Mat& grayscale, cv::Mat& threshold);
cv::Mat extract_value(cv::Mat& original);
cv::Mat maximize_contrast(cv::Mat& grayscale);
