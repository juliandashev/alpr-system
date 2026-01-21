#include "detect_chars.hpp"
#include "detect_plates.hpp"
#include "preprocess.hpp"
#include "debug.hpp"

cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();

static cv::Mat draw_chars_on_threshold(const cv::Mat& threshold, const std::vector<PossibleChar>& chars, cv::Scalar color = cv::Scalar(0, 255, 0)) {
    cv::Mat result;
    cv::cvtColor(threshold, result, cv::COLOR_GRAY2BGR);

    for (const auto& c : chars) {
        cv::rectangle(result, c.bounding_rect, color, 2);
    }
    return result;
}

static cv::Mat draw_chars_with_color_info(const cv::Mat& threshold, const std::vector<PossibleChar>& chars) {
    cv::Mat result;
    cv::cvtColor(threshold, result, cv::COLOR_GRAY2BGR);

    for (const auto& c : chars) {
        bool passes = c.background_contrast >= plate_detection::MIN_BACKGROUND_CONTRAST &&
                      c.color_stddev <= plate_detection::MAX_COLOR_STDDEV;
        cv::Scalar color = passes ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::rectangle(result, c.bounding_rect, color, 2);
    }
    return result;
}

static cv::Mat draw_contours_on_threshold(const cv::Mat& threshold, const std::vector<std::vector<cv::Point>>& contours) {
    cv::Mat result;
    cv::cvtColor(threshold, result, cv::COLOR_GRAY2BGR);
    cv::drawContours(result, contours, -1, cv::Scalar(0, 255, 0), 1);
    return result;
}

static cv::Mat draw_groups_on_threshold(const cv::Mat& threshold, const std::vector<std::vector<PossibleChar>>& groups) {
    cv::Mat result;
    cv::cvtColor(threshold, result, cv::COLOR_GRAY2BGR);

    std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255),
        cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255)
    };

    int idx = 0;
    for (const auto& group : groups) {
        cv::Scalar color = colors[idx % colors.size()];
        for (const auto& c : group) {
            cv::rectangle(result, c.bounding_rect, color, 2);
        }
        ++idx;
    }
    return result;
}

bool load_knn_data_and_train_knn() {
    cv::Mat classifications;
    cv::FileStorage classifications_file("classifications.xml", cv::FileStorage::READ);

    if (!classifications_file.isOpened()) {
        std::cout << "Error: unable to open training classifications file\n\n";
        return false;
    }

    classifications_file["classifications"] >> classifications;
    classifications_file.release();

    cv::Mat training_images;
    cv::FileStorage images_file("images.xml", cv::FileStorage::READ);

    if (!images_file.isOpened()) {
        std::cout << "Error: unable to open training images file\n\n";
        return false;
    }

    images_file["images"] >> training_images;
    images_file.release();

    knn->setDefaultK(1);
    knn->train(training_images, cv::ml::ROW_SAMPLE, classifications);

    return true;
}

std::vector<PossiblePlate> detect_chars_in_plates(std::vector<PossiblePlate>& plates) {
    if (plates.empty()) {
        return plates;
    }

    int plate_idx = 0;
    for (auto& plate : plates) {
        preprocess(plate.image, plate.grayscale, plate.threshold);

        cv::Mat resized_color;
        cv::resize(plate.image, resized_color, cv::Size(),
                   char_detection::PLATE_SCALE_FACTOR, char_detection::PLATE_SCALE_FACTOR);
        cv::resize(plate.threshold, plate.threshold, cv::Size(),
                   char_detection::PLATE_SCALE_FACTOR, char_detection::PLATE_SCALE_FACTOR);
        cv::threshold(plate.threshold, plate.threshold, 0.0, 255.0,
                      cv::THRESH_BINARY | cv::THRESH_OTSU);

        debug::save_image(resized_color, "3_char_detect", plate_idx, "1_plate_color");
        debug::save_image(plate.threshold, "3_char_detect", plate_idx, "2_plate_threshold");

        // Find all contours first
        auto thresh_copy = plate.threshold.clone();
        std::vector<std::vector<cv::Point>> all_contours;
        cv::findContours(thresh_copy, all_contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        debug::save_image(draw_contours_on_threshold(plate.threshold, all_contours), "3_char_detect", plate_idx, "3_all_contours");

        // Filter to character-like contours
        auto possible_chars = find_possible_chars_in_plate(resized_color, plate.threshold);
        debug::save_image(draw_chars_on_threshold(plate.threshold, possible_chars), "3_char_detect", plate_idx, "4_char_like_filtered");

        // Show color properties (green=passes, red=fails)
        debug::save_image(draw_chars_with_color_info(plate.threshold, possible_chars), "3_char_detect", plate_idx, "5_color_properties");

        // Color filtering
        auto color_filtered_chars = filter_chars_by_color(possible_chars);
        debug::save_image(draw_chars_on_threshold(plate.threshold, color_filtered_chars), "3_char_detect", plate_idx, "6_color_filtered");

        // Group by proximity/alignment
        auto char_groups = find_matching_char_groups(color_filtered_chars);
        debug::save_image(draw_groups_on_threshold(plate.threshold, char_groups), "3_char_detect", plate_idx, "7_char_groups");

        if (char_groups.empty()) {
            plate.chars = "";
            ++plate_idx;
            continue;
        }

        for (auto& group : char_groups) {
            std::sort(group.begin(), group.end(), PossibleChar::sort_left_to_right);
            group = remove_overlapping_chars(group);
            group = remove_chars_with_large_gaps(group);
        }

        auto longest_group = std::max_element(char_groups.begin(), char_groups.end(),
            [](const auto& a, const auto& b) { return a.size() < b.size(); });

        debug::save_image(draw_chars_on_threshold(plate.threshold, *longest_group), "3_char_detect", plate_idx, "8_final_chars");

        plate.chars = recognize_chars_in_plate(plate.threshold, *longest_group);
        ++plate_idx;
    }

    return plates;
}

std::vector<PossibleChar> find_possible_chars_in_plate(cv::Mat& color_image, cv::Mat& threshold) {
    std::vector<PossibleChar> possible_chars;

    auto thresh_copy = threshold.clone();

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh_copy, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    for (auto& contour : contours) {
        PossibleChar character(contour);
        if (is_possible_char(character)) {
            // Compute color properties for filtering
            character.compute_color_properties(color_image, threshold);
            possible_chars.push_back(character);
        }
    }

    return possible_chars;
}

bool is_possible_char(const PossibleChar& character) {
    return character.bounding_rect.area() > char_detection::MIN_PIXEL_AREA &&
           character.bounding_rect.width > char_detection::MIN_PIXEL_WIDTH &&
           character.bounding_rect.height > char_detection::MIN_PIXEL_HEIGHT &&
           character.aspect_ratio > char_detection::MIN_ASPECT_RATIO &&
           character.aspect_ratio < char_detection::MAX_ASPECT_RATIO;
}

std::vector<PossibleChar> filter_chars_by_color(std::vector<PossibleChar>& chars) {
    std::vector<PossibleChar> filtered;

    for (const auto& c : chars) {
        // Check contrast with background
        if (c.background_contrast < plate_detection::MIN_BACKGROUND_CONTRAST) {
            continue;
        }
        // Check color uniformity within character
        if (c.color_stddev > plate_detection::MAX_COLOR_STDDEV) {
            continue;
        }
        filtered.push_back(c);
    }

    return filtered;
}

std::vector<std::vector<PossibleChar>> find_matching_char_groups(const std::vector<PossibleChar>& chars) {
    std::vector<std::vector<PossibleChar>> groups;

    for (const auto& character : chars) {
        auto matches = find_matching_chars(character, chars);
        matches.push_back(character);

        if (matches.size() < char_detection::MIN_MATCHING_CHARS) {
            continue;
        }

        groups.push_back(matches);

        std::vector<PossibleChar> remaining;
        for (const auto& c : chars) {
            if (std::find(matches.begin(), matches.end(), c) == matches.end()) {
                remaining.push_back(c);
            }
        }

        auto recursive_groups = find_matching_char_groups(remaining);
        for (auto& group : recursive_groups) {
            groups.push_back(group);
        }

        break;
    }

    return groups;
}

std::vector<PossibleChar> find_matching_chars(const PossibleChar& character, const std::vector<PossibleChar>& chars) {
    std::vector<PossibleChar> matches;

    for (const auto& other : chars) {
        if (other == character) {
            continue;
        }

        auto distance = distance_between_chars(character, other);
        auto angle = angle_between_chars(character, other);
        auto area_change = static_cast<double>(std::abs(other.bounding_rect.area() - character.bounding_rect.area()))
                           / static_cast<double>(character.bounding_rect.area());
        auto width_change = static_cast<double>(std::abs(other.bounding_rect.width - character.bounding_rect.width))
                            / static_cast<double>(character.bounding_rect.width);
        auto height_change = static_cast<double>(std::abs(other.bounding_rect.height - character.bounding_rect.height))
                             / static_cast<double>(character.bounding_rect.height);

        if (distance < (character.diagonal_size * char_detection::MAX_DIAG_SIZE_MULTIPLE_AWAY) &&
            angle < char_detection::MAX_ANGLE_BETWEEN_CHARS &&
            area_change < char_detection::MAX_CHANGE_IN_AREA &&
            width_change < char_detection::MAX_CHANGE_IN_WIDTH &&
            height_change < char_detection::MAX_CHANGE_IN_HEIGHT) {
            matches.push_back(other);
        }
    }

    return matches;
}

double distance_between_chars(const PossibleChar& first, const PossibleChar& second) {
    int dx = std::abs(first.center_x - second.center_x);
    int dy = std::abs(first.center_y - second.center_y);
    return std::sqrt(std::pow(dx, 2) + std::pow(dy, 2));
}

double angle_between_chars(const PossibleChar& first, const PossibleChar& second) {
    double adjacent = std::abs(first.center_x - second.center_x);
    double opposite = std::abs(first.center_y - second.center_y);
    double angle_rad = std::atan(opposite / adjacent);
    return angle_rad * (180.0 / CV_PI);
}

std::vector<PossibleChar> remove_overlapping_chars(std::vector<PossibleChar>& chars) {
    std::vector<PossibleChar> result(chars);

    for (const auto& current : chars) {
        for (const auto& other : chars) {
            if (current == other) {
                continue;
            }

            if (distance_between_chars(current, other) < (current.diagonal_size * char_detection::MIN_DIAG_SIZE_MULTIPLE_AWAY)) {
                const auto& to_remove = (current.bounding_rect.area() < other.bounding_rect.area()) ? current : other;
                auto it = std::find(result.begin(), result.end(), to_remove);
                if (it != result.end()) {
                    result.erase(it);
                }
            }
        }
    }

    return result;
}

std::vector<PossibleChar> remove_chars_with_large_gaps(std::vector<PossibleChar>& chars) {
    if (chars.size() < 3) {
        return chars;
    }

    std::vector<int> gaps;
    for (size_t i = 1; i < chars.size(); ++i) {
        gaps.push_back(chars[i].center_x - chars[i - 1].center_x);
    }

    std::vector<int> sorted_gaps = gaps;
    std::sort(sorted_gaps.begin(), sorted_gaps.end());
    int baseline_gap = sorted_gaps[sorted_gaps.size() / 4];

    constexpr double GAP_THRESHOLD_MULTIPLIER = 3.0;
    int large_gap_index = -1;
    int max_gap_ratio = 0;

    for (size_t i = 0; i < gaps.size(); ++i) {
        if (baseline_gap > 0) {
            int ratio = gaps[i] / baseline_gap;
            if (ratio >= GAP_THRESHOLD_MULTIPLIER && ratio > max_gap_ratio) {
                max_gap_ratio = ratio;
                large_gap_index = static_cast<int>(i);
            }
        }
    }

    if (large_gap_index == -1) {
        return chars;
    }

    std::vector<PossibleChar> left_group(chars.begin(), chars.begin() + large_gap_index + 1);
    std::vector<PossibleChar> right_group(chars.begin() + large_gap_index + 1, chars.end());

    return (left_group.size() >= right_group.size()) ? left_group : right_group;
}

std::string recognize_chars_in_plate(cv::Mat& threshold, std::vector<PossibleChar>& chars) {
    std::string result;

    std::sort(chars.begin(), chars.end(), PossibleChar::sort_left_to_right);

    for (const auto& character : chars) {
        auto roi = threshold(character.bounding_rect).clone();

        cv::Mat resized;
        cv::resize(roi, resized, cv::Size(char_detection::RESIZED_CHAR_WIDTH, char_detection::RESIZED_CHAR_HEIGHT));

        cv::Mat float_roi;
        resized.convertTo(float_roi, CV_32FC1);

        auto flattened = float_roi.reshape(1, 1);

        cv::Mat detected_char(0, 0, CV_32F);
        knn->findNearest(flattened, 1, detected_char);

        auto char_value = static_cast<char>(static_cast<int>(detected_char.at<float>(0, 0)));
        result += char_value;
    }

    return result;
}
