#pragma once

#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class PossiblePlate {
public:
    cv::Mat image;
    cv::Mat grayscale;
    cv::Mat threshold;
    cv::RotatedRect location;
    std::string chars;

    static bool sort_by_char_count_desc(const PossiblePlate& left, const PossiblePlate& right) {
        return left.chars.length() > right.chars.length();
    }
};
