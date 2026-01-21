#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

int main(int argc, char** argv) {
  std::string image_path =
      (argc > 1) ? std::string(argv[1]) : "data/image1.jpg";
  std::string output_image_path =
      (argc > 2) ? std::string(argv[2]) : "out/output_image.jpg";

  // Read input image
  cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);

  // 2. Grayscale
  cv::Mat imgGray;
  cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

  std::cout << "DONE.\n";
  return 0;
}
