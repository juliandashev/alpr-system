#include "preprocess.hpp"
#include "debug.hpp"

void preprocess(cv::Mat& original, cv::Mat& grayscale, cv::Mat& threshold) {
    grayscale = extract_value(original);
    debug::save_image(grayscale, "1_preprocess", "1_hsv_value");

    auto max_contrast = maximize_contrast(grayscale);
    debug::save_image(max_contrast, "1_preprocess", "2_contrast_enhanced");

    cv::Mat blurred;
    cv::GaussianBlur(max_contrast, blurred,
                     cv::Size(preprocessing::GAUSSIAN_FILTER_SIZE, preprocessing::GAUSSIAN_FILTER_SIZE), 0);
    debug::save_image(blurred, "1_preprocess", "3_gaussian_blur");

    cv::adaptiveThreshold(blurred, threshold, 255.0,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV,
                          preprocessing::ADAPTIVE_THRESH_BLOCK_SIZE, preprocessing::ADAPTIVE_THRESH_WEIGHT);
    debug::save_image(threshold, "1_preprocess", "4_threshold");
}

cv::Mat extract_value(cv::Mat& original) {
    cv::Mat hsv;
    cv::cvtColor(original, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);

    return channels[2];
}

cv::Mat maximize_contrast(cv::Mat& grayscale) {
    auto kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    cv::Mat top_hat, black_hat;
    cv::morphologyEx(grayscale, top_hat, cv::MORPH_TOPHAT, kernel);
    cv::morphologyEx(grayscale, black_hat, cv::MORPH_BLACKHAT, kernel);

    return grayscale + top_hat - black_hat;
}
