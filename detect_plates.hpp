#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "detect_chars.hpp"
#include "main.hpp"
#include "possible_char.hpp"
#include "possible_plate.hpp"
#include "preprocess.hpp"

constexpr double PLATE_WIDTH_PADDING_FACTOR = 1.3;
constexpr double PLATE_HEIGHT_PADDING_FACTOR = 1.5;

std::vector<PossiblePlate> detect_plates(cv::Mat& original_image);
std::vector<PossibleChar> find_possible_plates(cv::Mat& threshold);

PossiblePlate extract_plate(cv::Mat& original_image,
                            std::vector<PossibleChar>& matching_chars);