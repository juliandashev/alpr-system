#include <iostream>

#include "detect_plates.hpp"
#include "detect_chars.hpp"
#include "preprocess.hpp"
#include "debug.hpp"

static cv::Mat draw_contours_on_image(const cv::Mat& image, const std::vector<std::vector<cv::Point>>& contours, cv::Scalar color = cv::Scalar(0, 255, 0)) {
    cv::Mat result;
    if (image.channels() == 1) {
        cv::cvtColor(image, result, cv::COLOR_GRAY2BGR);
    } else {
        result = image.clone();
    }
    cv::drawContours(result, contours, -1, color, 1);
    return result;
}

static cv::Mat draw_chars_on_image(const cv::Mat& image, const std::vector<PossibleChar>& chars, cv::Scalar color = cv::Scalar(0, 255, 0)) {
    cv::Mat result;
    if (image.channels() == 1) {
        cv::cvtColor(image, result, cv::COLOR_GRAY2BGR);
    } else {
        result = image.clone();
    }

    for (const auto& c : chars) {
        cv::rectangle(result, c.bounding_rect, color, 2);
    }
    return result;
}

static cv::Mat draw_chars_with_color_info(const cv::Mat& image, const std::vector<PossibleChar>& chars) {
    cv::Mat result;
    if (image.channels() == 1) {
        cv::cvtColor(image, result, cv::COLOR_GRAY2BGR);
    } else {
        result = image.clone();
    }

    for (const auto& c : chars) {
        // Color based on whether it passes color filter
        bool passes = c.background_contrast >= plate_detection::MIN_BACKGROUND_CONTRAST &&
                      c.color_stddev <= plate_detection::MAX_COLOR_STDDEV;
        cv::Scalar color = passes ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::rectangle(result, c.bounding_rect, color, 2);
    }
    return result;
}

static cv::Mat draw_groups_on_image(const cv::Mat& image, const std::vector<std::vector<PossibleChar>>& groups) {
    cv::Mat result;
    if (image.channels() == 1) {
        cv::cvtColor(image, result, cv::COLOR_GRAY2BGR);
    } else {
        result = image.clone();
    }

    std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 255, 0),   // Green
        cv::Scalar(255, 0, 0),   // Blue
        cv::Scalar(0, 0, 255),   // Red
        cv::Scalar(255, 255, 0), // Cyan
        cv::Scalar(255, 0, 255), // Magenta
        cv::Scalar(0, 255, 255)  // Yellow
    };

    int group_idx = 0;
    for (const auto& group : groups) {
        cv::Scalar color = colors[group_idx % colors.size()];
        for (const auto& c : group) {
            cv::rectangle(result, c.bounding_rect, color, 2);
        }
        ++group_idx;
    }
    return result;
}

std::vector<PossiblePlate> detect_plates_in_scene(cv::Mat& scene) {
    std::vector<PossiblePlate> plates;

    cv::Mat grayscale, threshold;
    preprocess(scene, grayscale, threshold);

    // Step 1: Find all contours
    auto thresh_copy = threshold.clone();
    std::vector<std::vector<cv::Point>> all_contours;
    cv::findContours(thresh_copy, all_contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    debug::save_image(draw_contours_on_image(scene, all_contours), "2_plate_detect", "1_all_contours");

    // Step 2: Filter to character-like contours
    auto possible_chars = find_possible_chars_in_scene(scene, threshold);
    debug::save_image(draw_chars_on_image(scene, possible_chars), "2_plate_detect", "2_char_like_contours");

    // Step 3: Show color properties (green=passes, red=fails)
    debug::save_image(draw_chars_with_color_info(scene, possible_chars), "2_plate_detect", "3_color_properties");

    // Step 4: Group characters by proximity/alignment
    auto char_groups = find_matching_char_groups(possible_chars);
    debug::save_image(draw_groups_on_image(scene, char_groups), "2_plate_detect", "4_char_groups");

    // Step 5: Filter groups by color consistency
    auto filtered_groups = filter_groups_by_color(char_groups);
    debug::save_image(draw_groups_on_image(scene, filtered_groups), "2_plate_detect", "5_color_filtered_groups");

    // Step 6: Extract plate regions
    for (size_t i = 0; i < filtered_groups.size(); ++i) {
        auto plate = extract_plate(scene, filtered_groups[i]);
        if (!plate.image.empty()) {
            plates.push_back(plate);
            debug::save_image(plate.image, "2_plate_detect", static_cast<int>(i), "6_extracted_plate");
        }
    }

    std::cout << "\n" << plates.size() << " possible plates found\n";

    return plates;
}

std::vector<PossibleChar> find_possible_chars_in_scene(cv::Mat& color_image, cv::Mat& threshold) {
    std::vector<PossibleChar> possible_chars;
    auto thresh_copy = threshold.clone();

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh_copy, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    for (auto& contour : contours) {
        PossibleChar character(contour);
        if (is_possible_char(character)) {
            character.compute_color_properties(color_image, threshold);
            possible_chars.push_back(character);
        }
    }

    return possible_chars;
}

bool passes_color_filter(const PossibleChar& character) {
    return character.background_contrast >= plate_detection::MIN_BACKGROUND_CONTRAST &&
           character.color_stddev <= plate_detection::MAX_COLOR_STDDEV;
}

bool group_has_consistent_color(const std::vector<PossibleChar>& group) {
    if (group.size() < 2) {
        return true;
    }

    cv::Scalar group_mean(0, 0, 0);
    for (const auto& c : group) {
        group_mean[0] += c.mean_color[0];
        group_mean[1] += c.mean_color[1];
        group_mean[2] += c.mean_color[2];
    }
    group_mean[0] /= group.size();
    group_mean[1] /= group.size();
    group_mean[2] /= group.size();

    double total_variance = 0.0;
    for (const auto& c : group) {
        double b_diff = c.mean_color[0] - group_mean[0];
        double g_diff = c.mean_color[1] - group_mean[1];
        double r_diff = c.mean_color[2] - group_mean[2];
        total_variance += std::sqrt(b_diff * b_diff + g_diff * g_diff + r_diff * r_diff);
    }

    return (total_variance / group.size()) <= plate_detection::MAX_GROUP_COLOR_VARIANCE;
}

std::vector<std::vector<PossibleChar>> filter_groups_by_color(std::vector<std::vector<PossibleChar>>& groups) {
    std::vector<std::vector<PossibleChar>> filtered;

    for (auto& group : groups) {
        std::vector<PossibleChar> color_filtered_chars;
        for (const auto& c : group) {
            if (passes_color_filter(c)) {
                color_filtered_chars.push_back(c);
            }
        }

        if (color_filtered_chars.size() < 3) {
            continue;
        }

        if (group_has_consistent_color(color_filtered_chars)) {
            filtered.push_back(color_filtered_chars);
        }
    }

    return filtered;
}

PossiblePlate extract_plate(cv::Mat& original, std::vector<PossibleChar>& chars) {
    PossiblePlate plate;

    std::sort(chars.begin(), chars.end(), PossibleChar::sort_left_to_right);

    const auto& first = chars.front();
    const auto& last = chars.back();

    double center_x = (first.center_x + last.center_x) / 2.0;
    double center_y = (first.center_y + last.center_y) / 2.0;
    cv::Point2d center(center_x, center_y);

    int width = static_cast<int>((last.bounding_rect.x + last.bounding_rect.width - first.bounding_rect.x)
                                  * plate_detection::WIDTH_PADDING_FACTOR);

    double total_height = 0.0;
    for (const auto& c : chars) {
        total_height += c.bounding_rect.height;
    }
    double avg_height = total_height / chars.size();
    int height = static_cast<int>(avg_height * plate_detection::HEIGHT_PADDING_FACTOR);

    double opposite = last.center_y - first.center_y;
    double hypotenuse = distance_between_chars(first, last);
    double angle_rad = std::asin(opposite / hypotenuse);
    double angle_deg = angle_rad * (180.0 / CV_PI);

    plate.location = cv::RotatedRect(center, cv::Size2f(static_cast<float>(width), static_cast<float>(height)),
                                     static_cast<float>(angle_deg));

    auto rotation_matrix = cv::getRotationMatrix2D(center, angle_deg, 1.0);

    cv::Mat rotated;
    cv::warpAffine(original, rotated, rotation_matrix, original.size());

    cv::Mat cropped;
    cv::getRectSubPix(rotated, plate.location.size, plate.location.center, cropped);

    plate.image = cropped;

    return plate;
}
