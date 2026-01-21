#include "possible_char.hpp"

PossibleChar::PossibleChar(std::vector<cv::Point> contour)
    : contour{std::move(contour)},
      bounding_rect{cv::boundingRect(this->contour)},
      center_x{bounding_rect.x + bounding_rect.width / 2},
      center_y{bounding_rect.y + bounding_rect.height / 2},
      diagonal_size{std::sqrt(std::pow(bounding_rect.width, 2) +
                              std::pow(bounding_rect.height, 2))},
      aspect_ratio{static_cast<double>(bounding_rect.width) /
                   static_cast<double>(bounding_rect.height)},
      mean_color{0, 0, 0, 0},
      color_stddev{0.0},
      background_contrast{0.0} {}

void PossibleChar::compute_color_properties(const cv::Mat& color_image,
                                            const cv::Mat& threshold_mask) {
  // Ensure bounding rect is within image bounds
  cv::Rect safe_rect =
      bounding_rect & cv::Rect(0, 0, color_image.cols, color_image.rows);
  if (safe_rect.width <= 0 || safe_rect.height <= 0) {
    return;
  }

  // Extract ROI from color image and threshold mask
  cv::Mat color_roi = color_image(safe_rect);
  cv::Mat mask_roi = threshold_mask(safe_rect);

  // Create mask for character pixels (white pixels in threshold = character)
  cv::Mat char_mask;
  cv::threshold(mask_roi, char_mask, 127, 255, cv::THRESH_BINARY);

  // Create mask for background pixels (inverse)
  cv::Mat bg_mask;
  cv::bitwise_not(char_mask, bg_mask);

  // Compute mean and stddev of character pixels
  cv::Scalar char_mean, char_stddev;
  cv::meanStdDev(color_roi, char_mean, char_stddev, char_mask);

  mean_color = char_mean;
  // Average stddev across BGR channels
  color_stddev = (char_stddev[0] + char_stddev[1] + char_stddev[2]) / 3.0;

  // Compute mean of background pixels
  cv::Scalar bg_mean, bg_stddev;
  cv::meanStdDev(color_roi, bg_mean, bg_stddev, bg_mask);

  // Compute contrast as Euclidean distance in BGR space
  double b_diff = char_mean[0] - bg_mean[0];
  double g_diff = char_mean[1] - bg_mean[1];
  double r_diff = char_mean[2] - bg_mean[2];
  background_contrast =
      std::sqrt(b_diff * b_diff + g_diff * g_diff + r_diff * r_diff);
}
