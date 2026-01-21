#pragma once

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "possible_plate.hpp"
#include "possible_char.hpp"

namespace plate_detection {
    constexpr double WIDTH_PADDING_FACTOR = 1.3;
    constexpr double HEIGHT_PADDING_FACTOR = 1.5;

    // Color filtering thresholds
    constexpr double MIN_BACKGROUND_CONTRAST = 50.0;  // Minimum contrast between char and background
    constexpr double MAX_COLOR_STDDEV = 80.0;         // Maximum color variance within a character
    constexpr double MAX_GROUP_COLOR_VARIANCE = 60.0; // Maximum color variance within a character group
}

std::vector<PossiblePlate> detect_plates_in_scene(cv::Mat& scene);
std::vector<PossibleChar> find_possible_chars_in_scene(cv::Mat& color_image, cv::Mat& threshold);
PossiblePlate extract_plate(cv::Mat& original, std::vector<PossibleChar>& chars);

// Color filtering functions
bool passes_color_filter(const PossibleChar& character);
bool group_has_consistent_color(const std::vector<PossibleChar>& group);
std::vector<std::vector<PossibleChar>> filter_groups_by_color(std::vector<std::vector<PossibleChar>>& groups);
