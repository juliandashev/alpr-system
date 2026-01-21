#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class PossibleChar {
 public:
  PossibleChar(std::vector<cv::Point> _contour);

  std::vector<cv::Point> contour;

  cv::Rect boundingRect;

  int centerX;
  int centerY;

  double diagonalSize;
  double aspectRatio;

  // Sort chars from left to right
  static bool is_left_to_right(const PossibleChar& left,
                               const PossibleChar& right) {
    return (left.centerX < right.centerX);
  }

  // Operator overloads

  bool operator==(const PossibleChar& other) const {
    return (this->contour == other.contour);
  }

  bool operator!=(const PossibleChar& other) const {
    return (this->contour != other.contour);
  }
};