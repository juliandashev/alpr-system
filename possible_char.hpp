#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class PossibleChar {
 public:
  std::vector<cv::Point> contour;
  cv::Rect bounding_rect;
  int center_x;
  int center_y;
  double diagonal_size;
  double aspect_ratio;

  // Color properties for filtering
  cv::Scalar mean_color;  // Mean BGR color of the character pixels
  double
      color_stddev;  // Color variance within character (lower = more uniform)
  double background_contrast;  // Contrast with surrounding background

  explicit PossibleChar(std::vector<cv::Point> contour);

  // Compute color properties from a color image
  void compute_color_properties(const cv::Mat& color_image,
                                const cv::Mat& threshold_mask);

  static bool sort_left_to_right(const PossibleChar& left,
                                 const PossibleChar& right) {
    return left.center_x < right.center_x;
  }

  bool operator==(const PossibleChar& other) const {
    return contour == other.contour;
  }

  bool operator!=(const PossibleChar& other) const { return !(*this == other); }
};
