#pragma once

#include <filesystem>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "detect_plates.hpp"
#include "possible_plate.hpp"
#include "detect_chars.hpp"

namespace fs = std::filesystem;

namespace colors {
    const cv::Scalar BLACK{0.0, 0.0, 0.0};
    const cv::Scalar WHITE{255.0, 255.0, 255.0};
    const cv::Scalar YELLOW{0.0, 255.0, 255.0};
    const cv::Scalar GREEN{0.0, 255.0, 0.0};
    const cv::Scalar RED{0.0, 0.0, 255.0};
}

inline void ensure_output_directory_exists(const std::string& dir = "out") {
    const fs::path out_dir{dir};
    if (!fs::exists(out_dir)) {
        fs::create_directory(out_dir);
    }
}

inline void draw_red_rectangle_around_plate(cv::Mat& scene, PossiblePlate& plate) {
    cv::Point2f vertices[4];
    plate.location.points(vertices);

    for (int i = 0; i < 4; ++i) {
        cv::line(scene, vertices[i], vertices[(i + 1) % 4], colors::RED, 2);
    }
}

inline void write_license_plate_chars_on_image(cv::Mat& scene, PossiblePlate& plate) {
    constexpr int FONT = cv::FONT_HERSHEY_SIMPLEX;

    double font_scale = static_cast<double>(plate.image.rows) / 30.0;
    int font_thickness = static_cast<int>(std::round(font_scale * 1.5));
    int baseline = 0;

    auto text_size = cv::getTextSize(plate.chars, FONT, font_scale, font_thickness, &baseline);

    cv::Point text_center;
    text_center.x = static_cast<int>(plate.location.center.x);

    if (plate.location.center.y < (scene.rows * 0.75)) {
        text_center.y = static_cast<int>(std::round(plate.location.center.y) +
                                         std::round(static_cast<double>(plate.image.rows) * 1.6));
    } else {
        text_center.y = static_cast<int>(std::round(plate.location.center.y) -
                                         std::round(static_cast<double>(plate.image.rows) * 1.6));
    }

    cv::Point text_origin;
    text_origin.x = text_center.x - (text_size.width / 2);
    text_origin.y = text_center.y + (text_size.height / 2);

    cv::putText(scene, plate.chars, text_origin, FONT, font_scale, colors::YELLOW, font_thickness);
}
