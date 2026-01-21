#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <string>
#include <vector>

#include "possible_char.hpp"
#include "possible_plate.hpp"

namespace char_detection {
// Processing constants
constexpr double PLATE_SCALE_FACTOR = 1.6;

// Minimum character dimensions
constexpr int MIN_PIXEL_WIDTH = 2;
constexpr int MIN_PIXEL_HEIGHT = 8;
constexpr int MIN_PIXEL_AREA = 80;
constexpr double MIN_ASPECT_RATIO = 0.25;
constexpr double MAX_ASPECT_RATIO = 1.0;

// Character grouping parameters (tuned for 8-character plates)
constexpr double MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3;
constexpr double MAX_DIAG_SIZE_MULTIPLE_AWAY =
    10.0;  // Increased from 5.0 for longer plates
constexpr double MAX_CHANGE_IN_AREA = 0.6;  // Increased from 0.5
constexpr double MAX_CHANGE_IN_WIDTH = 0.8;
constexpr double MAX_CHANGE_IN_HEIGHT =
    0.4;  // Increased from 0.2 (was too strict)
constexpr double MAX_ANGLE_BETWEEN_CHARS = 15.0;  // Increased from 12.0
constexpr int MIN_MATCHING_CHARS = 3;

// Character recognition
constexpr int RESIZED_CHAR_WIDTH = 20;
constexpr int RESIZED_CHAR_HEIGHT = 30;
constexpr int MIN_CONTOUR_AREA = 100;
}  // namespace char_detection

extern cv::Ptr<cv::ml::KNearest> knn;

bool load_knn_data_and_train_knn();

std::vector<PossiblePlate> detect_chars_in_plates(
    std::vector<PossiblePlate>& plates);
std::vector<PossibleChar> find_possible_chars_in_plate(cv::Mat& color_image,
                                                       cv::Mat& threshold);
bool is_possible_char(const PossibleChar& character);

// Color filtering for plate characters
std::vector<PossibleChar> filter_chars_by_color(
    std::vector<PossibleChar>& chars);

std::vector<std::vector<PossibleChar>> find_matching_char_groups(
    const std::vector<PossibleChar>& chars);
std::vector<PossibleChar> find_matching_chars(
    const PossibleChar& character, const std::vector<PossibleChar>& chars);

double distance_between_chars(const PossibleChar& first,
                              const PossibleChar& second);
double angle_between_chars(const PossibleChar& first,
                           const PossibleChar& second);

std::vector<PossibleChar> remove_overlapping_chars(
    std::vector<PossibleChar>& chars);
std::vector<PossibleChar> remove_chars_with_large_gaps(
    std::vector<PossibleChar>& chars);
std::string recognize_chars_in_plate(cv::Mat& threshold,
                                     std::vector<PossibleChar>& chars);
