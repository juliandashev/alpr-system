#pragma once

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace debug {
    void set_output_dir(const std::string& dir);
    void set_enabled(bool enabled);
    bool is_enabled();

    void save_image(const cv::Mat& img, const std::string& stage, const std::string& name);
    void save_image(const cv::Mat& img, const std::string& stage, int index, const std::string& name);
}
