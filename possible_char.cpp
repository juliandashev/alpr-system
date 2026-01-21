#include "possible_char.hpp"

PossibleChar::PossibleChar(std::vector<cv::Point> _contour) {
  contour = _contour;

  boundingRect = cv::boundingRect(contour);

  centerX = (boundingRect.x + boundingRect.x + boundingRect.width) / 2;
  centerY = (boundingRect.y + boundingRect.y + boundingRect.height) / 2;
  diagonalSize = sqrt(pow(boundingRect.width, 2) + pow(boundingRect.height, 2));
  aspectRatio = (float)boundingRect.width / (float)boundingRect.height;
}