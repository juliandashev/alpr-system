#include "detect_plates.hpp"

std::vector<PossiblePlate> detect_plates(cv::Mat& original) {
  std::vector<PossiblePlate> possible_plates;

  cv::Mat gray;
  cv::Mat threshold;
  cv::Mat contours(original.size(), CV_8UC3, SCALAR_BLACK);

  cv::RNG rng;

  cv::destroyAllWindows();

  // preprocess to get grayscale and threshold images
  preprocess(original, gray, threshold);

  // find all contours -> include those that could be chars
  std::vector<PossibleChar> possible_chars = find_possible_plates(threshold);

  // given a vector of all possible chars, find groups of matching chars
  // in the next steps each group of matching chars will attempt to be
  // recognized as a plate
  std::vector<std::vector<PossibleChar>> possible_char_matrix =
      possible_char_matrix(possible_chars);

  for (auto& matching_chars : possible_char_matrix) {
    PossiblePlate current_plate = extract_plate(original, matching_chars);

    if (!current_plate.imgPlate.empty()) {
      // add to vector of possible plates
      possible_plates.push_back(current_plate);
    }
  }

  std::cout << std::endl
            << possible_plates.size() << " possible plates found" << std::endl;

  return possible_plates;
}

std::vector<PossibleChar> find_possible_plates(cv::Mat& threshold) {
  std::vector<PossibleChar> possible_chars;

  cv::Mat contours(threshold.size(), CV_8UC3, SCALAR_BLACK);
  cv::Mat threshold_copy = threshold.clone();

  std::vector<std::vector<cv::Point>> contours;

  // Find contours
  cv::findContours(threshold_copy, contours, CV_RETR_LIST,
                   CV_CHAIN_APPROX_SIMPLE);

  // init possible char counter
  int count = 0;
  for (unsigned int i = 0; i < contours.size(); i++) {  // for each contour
    PossibleChar current(contours[i]);

    if (is_possible_char(current)) {
      count++;
      possible_chars.push_back(current);
    }
  }

  return possible_chars;
}

PossiblePlate extract_plate(cv::Mat& original_image,
                            std::vector<PossibleChar>& matching_chars) {
  PossiblePlate plate;
  int last_char_index = static_cast<int>(matching_chars.size() - 1);
  double total_char_heights = 0;

  // sort chars from left to right based on x position
  std::sort(matching_chars.begin(), matching_chars.end(),
            PossibleChar::is_left_to_right);

  // calculate the center point of the plate
  double plate_center_x =
      static_cast<double>(matching_chars[0].centerX +
                          matching_chars[last_char_index].centerX) /
      2.0;
  double plate_center_y =
      static_cast<double>(matching_chars[0].centerY +
                          matching_chars[last_char_index].centerY) /
      2.0;

  cv::Point2d center(plate_center_x, plate_center_y);

  // calculate plate width and height
  int plate_width =
      static_cast<int>((matching_chars[last_char_index].boundingRect.x +
                        matching_chars[last_char_index].boundingRect.width -
                        matching_chars[0].boundingRect.x) *
                       PLATE_WIDTH_PADDING_FACTOR);

  for (auto& current_char : matching_chars) {
    total_char_heights += current_char.boundingRect.height;
  }

  double avg_char_height =
      static_cast<double>(total_char_heights / (last_char_index + 1));

  int plate_height =
      static_cast<int>(avg_char_height * PLATE_HEIGHT_PADDING_FACTOR);

  // calculate correction angle of plate region
  double opposite =
      matching_chars[last_char_index].centerY - matching_chars[0].centerY;
  double hypotenuse = distance_between_chars(matching_chars[0],
                                             matching_chars[last_char_index]);
  double correction_angle_rad = asin(opposite / hypotenuse);
  double correction_angle_deg = correction_angle_rad * (180.0 / CV_PI);

  // assign rotated rect member variable of possible plate
  possiblePlate.plate_location =
      cv::RotatedRect(center,
                      cv::Size2f(static_cast<float>(plate_width),
                                 static_cast<float>(plate_height)),
                      static_cast<float>(correction_angle_deg));

  cv::Mat rotation_matrix;
  cv::Mat rotated;
  cv::Mat cropped;

  rotation_matrix = cv::getRotationMatrix2D(
      center, correction_angle_deg,
      1.0);  // get the rotation matrix for our calculated correction angle

  cv::warpAffine(original_image, rotated, rotation_matrix,
                 original_image.size());

  // crop out the actual plate portion of the rotated image
  cv::getRectSubPix(rotated, plate.plate_location.size,
                    plate.plate_location.center, cropped);

  // copy the cropped plate image into the applicable
  // member variable of the possible plate
  plate.imgPlate = cropped;

  return plate;
}