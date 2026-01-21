#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>

#include "debug.hpp"
#include "detect_chars.hpp"
#include "detect_plates.hpp"
#include "preprocess.hpp"
#include "utils.hpp"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0]
              << " <image_path> [output_dir] [--debug]\n";
    return EXIT_FAILURE;
  }

  std::string image_path = argv[1];
  std::string output_dir = "out";
  bool debug_mode = false;

  // Parse arguments
  for (int i = 2; i < argc; ++i) {
    if (std::strcmp(argv[i], "--debug") == 0 ||
        std::strcmp(argv[i], "-d") == 0) {
      debug_mode = true;
    } else {
      output_dir = argv[i];
    }
  }

  // Extract base name from image path (no extension)
  namespace fs = std::filesystem;
  std::string image_basename = fs::path(image_path).stem().string();

  // Clean output directory of previous outputs
  if (fs::exists(output_dir)) {
    for (const auto& entry : fs::directory_iterator(output_dir)) {
      if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
        fs::remove(entry.path());
      }
    }
    // Also clean debug subdirectory
    std::string debug_dir = output_dir + "/debug";
    if (fs::exists(debug_dir)) {
      fs::remove_all(debug_dir);
    }
  }

  // Set up debug output
  debug::set_output_dir(output_dir);
  debug::set_enabled(debug_mode);

  if (debug_mode) {
    std::cout << "Debug mode enabled - saving pipeline images to " << output_dir
              << "/debug/\n";
  }

  // Load image
  cv::Mat scene = cv::imread(image_path);
  if (scene.empty()) {
    std::cout << "Could not open image: " << image_path << "\n";
    return EXIT_FAILURE;
  }

  std::cout << "Loaded image: " << image_path << "\n";
  debug::save_image(scene, "0_input", "original");

  // Load KNN training data
  if (!load_knn_data_and_train_knn()) {
    std::cout << "Error: Could not load KNN training data\n";
    std::cout << "Make sure classifications.xml and images.xml exist\n";
    return EXIT_FAILURE;
  }

  // Detect plates in scene
  auto plates = detect_plates_in_scene(scene);

  if (plates.empty()) {
    std::cout << "No plates detected\n";
    return EXIT_SUCCESS;
  }

  // Detect and recognize characters in each plate
  auto plates_with_chars = detect_chars_in_plates(plates);
  constexpr double MIN_PLATE_ASPECT_RATIO = 2.0;
  constexpr double MAX_PLATE_ASPECT_RATIO = 7.0;
  constexpr size_t MIN_CHAR_COUNT = 6;

  std::vector<PossiblePlate*> valid_plates;
  for (auto& plate : plates_with_chars) {
    if (plate.chars.length() < MIN_CHAR_COUNT) {
      continue;
    }

    float width = plate.location.size.width;
    float height = plate.location.size.height;
    if (height > width) {
      std::swap(width, height);
    }
    double aspect_ratio =
        (height > 0) ? static_cast<double>(width) / height : 0;

    if (aspect_ratio >= MIN_PLATE_ASPECT_RATIO &&
        aspect_ratio <= MAX_PLATE_ASPECT_RATIO) {
      valid_plates.push_back(&plate);
    }
  }

  if (valid_plates.empty()) {
    std::cout << "No characters recognized\n";
    return EXIT_SUCCESS;
  }

  // Sort by character count (descending) for consistent output
  std::sort(valid_plates.begin(), valid_plates.end(),
            [](const PossiblePlate* a, const PossiblePlate* b) {
              return a->chars.length() > b->chars.length();
            });

  // Draw result on image for ALL valid plates
  ensure_output_directory_exists(output_dir);

  std::cout << "\nRecognized " << valid_plates.size() << " plate(s):\n";
  for (size_t i = 0; i < valid_plates.size(); ++i) {
    auto& plate = *valid_plates[i];
    std::cout << "  Plate " << (i + 1) << ": " << plate.chars << "\n";

    draw_red_rectangle_around_plate(scene, plate);
    write_license_plate_chars_on_image(scene, plate);

    // Save individual plate crops
    std::string plate_filename = output_dir + "/" + image_basename + "_plate_" +
                                 std::to_string(i + 1) + ".jpg";
    cv::imwrite(plate_filename, plate.image);
  }

  // Save outputs
  cv::imwrite(output_dir + "/" + image_basename + "_output.jpg", scene);

  std::cout << "Saved results to " << output_dir << "/\n";

  return EXIT_SUCCESS;
}
