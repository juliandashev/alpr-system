#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <random>
#include <set>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

namespace fs = std::filesystem;

// Constants matching the main ALPR system
constexpr int RESIZED_CHAR_WIDTH = 20;
constexpr int RESIZED_CHAR_HEIGHT = 30;

// License plate configuration
constexpr int LICENSE_PLATE_LENGTH = 8;
constexpr int MIN_SAMPLES_PER_CHAR = 10;

// Required characters for license plate recognition (0-9 and A-Z)
const std::string REQUIRED_DIGITS = "0123456789";
const std::string REQUIRED_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

std::set<char> get_required_characters() {
    std::set<char> required;
    for (char c : REQUIRED_DIGITS) required.insert(c);
    for (char c : REQUIRED_LETTERS) required.insert(c);
    return required;
}

// Supported image extensions
const std::vector<std::string> IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"
};

bool is_image_file(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return std::find(IMAGE_EXTENSIONS.begin(), IMAGE_EXTENSIONS.end(), ext) != IMAGE_EXTENSIONS.end();
}

int get_ascii_code(const std::string& label) {
    if (label.length() == 1) {
        char c = std::toupper(label[0]);
        if (c >= '0' && c <= '9') {
            return static_cast<int>(c);  // ASCII 48-57
        }
        if (c >= 'A' && c <= 'Z') {
            return static_cast<int>(c);  // ASCII 65-90
        }
    }
    return -1;  // Invalid label
}

cv::Mat process_image(const fs::path& image_path) {
    // Read image in grayscale
    cv::Mat img = cv::imread(image_path.string(), cv::IMREAD_GRAYSCALE);

    if (img.empty()) {
        return cv::Mat();
    }

    // Resize to standard dimensions
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(RESIZED_CHAR_WIDTH, RESIZED_CHAR_HEIGHT));

    // Apply Otsu threshold to get binary image
    cv::Mat threshold;
    cv::threshold(resized, threshold, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Ensure consistent polarity: characters should be WHITE (255) on BLACK (0) background
    // This matches how the main ALPR system processes plate images
    // Count white pixels - if more than half, the image is inverted (white background)
    int white_pixels = cv::countNonZero(threshold);
    int total_pixels = threshold.rows * threshold.cols;

    if (white_pixels > total_pixels / 2) {
        // Background is white, invert to get white chars on black background
        cv::bitwise_not(threshold, threshold);
    }

    // Convert to float and flatten
    cv::Mat float_img;
    threshold.convertTo(float_img, CV_32F);

    return float_img.reshape(1, 1);  // Flatten to 1 row
}

int main(int argc, char** argv) {
    std::cout << "   KNN Training Tool for ALPR System\n\n";

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <dataset_path> [output_dir]\n\n";
        std::cout << "Arguments:\n";
        std::cout << "  dataset_path  - Path to the training dataset folder\n";
        std::cout << "  output_dir    - Output directory for XML files (default: current directory)\n\n";
        std::cout << "Expected dataset structure:\n";
        std::cout << "  dataset/\n";
        std::cout << "    ├── 0/\n";
        std::cout << "    │   ├── img001.png\n";
        std::cout << "    │   └── ...\n";
        std::cout << "    ├── A/\n";
        std::cout << "    │   ├── img001.png\n";
        std::cout << "    │   └── ...\n";
        std::cout << "    └── ...\n";
        return 1;
    }

    fs::path dataset_path = argv[1];
    fs::path output_dir = (argc > 2) ? argv[2] : ".";

    // Validate paths
    if (!fs::exists(dataset_path) || !fs::is_directory(dataset_path)) {
        std::cerr << "Error: Dataset path does not exist or is not a directory: "
                  << dataset_path << "\n";
        return 1;
    }

    if (!fs::exists(output_dir)) {
        std::cout << "Creating output directory: " << output_dir << "\n";
        fs::create_directories(output_dir);
    }

    // Matrices to hold training data
    cv::Mat training_images;
    cv::Mat classifications;

    int total_samples = 0;
    int skipped_samples = 0;
    std::map<char, int> char_counts;

    std::cout << "Processing dataset from: " << dataset_path << "\n\n";

    // Iterate through each subdirectory (character class)
    std::vector<fs::directory_entry> dirs;
    for (const auto& entry : fs::directory_iterator(dataset_path)) {
        if (entry.is_directory()) {
            dirs.push_back(entry);
        }
    }

    // Sort directories for consistent ordering
    std::sort(dirs.begin(), dirs.end(), [](const auto& a, const auto& b) {
        return a.path().filename().string() < b.path().filename().string();
    });

    for (const auto& char_dir : dirs) {
        std::string label = char_dir.path().filename().string();
        int ascii_code = get_ascii_code(label);

        if (ascii_code < 0) {
            std::cout << "  Skipping invalid label folder: " << label << "\n";
            continue;
        }

        char display_char = static_cast<char>(ascii_code);
        int char_sample_count = 0;

        // Process all images in this character folder
        for (const auto& img_entry : fs::directory_iterator(char_dir.path())) {
            if (!img_entry.is_regular_file() || !is_image_file(img_entry.path())) {
                continue;
            }

            cv::Mat processed = process_image(img_entry.path());

            if (processed.empty()) {
                std::cerr << "    Warning: Could not process image: "
                          << img_entry.path() << "\n";
                skipped_samples++;
                continue;
            }

            // Add to training data
            training_images.push_back(processed);
            classifications.push_back(ascii_code);

            total_samples++;
            char_sample_count++;
        }

        if (char_sample_count > 0) {
            char_counts[display_char] = char_sample_count;
            std::cout << "  [" << display_char << "] Processed " << char_sample_count << " samples\n";
        }
    }

    std::cout << "\n----------------------------------------\n";
    std::cout << "Processing complete!\n";
    std::cout << "- Total samples: " << total_samples << "\n";
    std::cout << "- Skipped: " << skipped_samples << "\n";
    std::cout << "- Unique characters: " << char_counts.size() << "\n";
    std::cout << "----------------------------------------\n\n";

    // Validate coverage for 8-character license plates
    std::cout << "Validating for " << LICENSE_PLATE_LENGTH << "-character license plates...\n";

    std::set<char> required_chars = get_required_characters();
    std::vector<char> missing_chars;
    std::vector<char> low_sample_chars;

    for (char c : required_chars) {
        if (char_counts.find(c) == char_counts.end()) {
            missing_chars.push_back(c);
        } else if (char_counts[c] < MIN_SAMPLES_PER_CHAR) {
            low_sample_chars.push_back(c);
        }
    }

    // Report digits coverage
    int digits_found = 0;
    for (char c : REQUIRED_DIGITS) {
        if (char_counts.find(c) != char_counts.end()) digits_found++;
    }
    std::cout << "  Digits (0-9): " << digits_found << "/10\n";

    // Report letters coverage
    int letters_found = 0;
    for (char c : REQUIRED_LETTERS) {
        if (char_counts.find(c) != char_counts.end()) letters_found++;
    }
    std::cout << "  Letters (A-Z): " << letters_found << "/26\n";
    std::cout << "  Total coverage: " << char_counts.size() << "/36 characters\n";

    if (!missing_chars.empty()) {
        std::cout << "\n  ⚠ WARNING: Missing characters: ";
        for (char c : missing_chars) std::cout << c << " ";
        std::cout << "\n";
        std::cout << "  The model won't be able to recognize these characters!\n";
    }

    if (!low_sample_chars.empty()) {
        std::cout << "\n  ⚠ WARNING: Characters with < " << MIN_SAMPLES_PER_CHAR << " samples: ";
        for (char c : low_sample_chars) std::cout << c << "(" << char_counts[c] << ") ";
        std::cout << "\n";
        std::cout << "  Consider adding more samples for better accuracy.\n";
    }

    if (missing_chars.empty() && low_sample_chars.empty()) {
        std::cout << "\n  ✓ All 36 characters have sufficient training samples!\n";
    }

    std::cout << "----------------------------------------\n\n";

    if (total_samples == 0) {
        std::cerr << "Error: No valid training samples found!\n";
        std::cerr << "Make sure your dataset has the correct structure.\n";
        return 1;
    }

    // Convert classifications to proper format (CV_32S for integer labels)
    classifications.convertTo(classifications, CV_32S);

    // Save to XML files
    fs::path classifications_file = output_dir / "classifications.xml";
    fs::path images_file = output_dir / "images.xml";

    std::cout << "Saving training data...\n";

    // Save classifications
    cv::FileStorage fs_class(classifications_file.string(), cv::FileStorage::WRITE);
    if (!fs_class.isOpened()) {
        std::cerr << "Error: Could not create " << classifications_file << "\n";
        return 1;
    }
    fs_class << "classifications" << classifications;
    fs_class.release();
    std::cout << "  Saved: " << classifications_file << "\n";

    // Save images
    cv::FileStorage fs_img(images_file.string(), cv::FileStorage::WRITE);
    if (!fs_img.isOpened()) {
        std::cerr << "Error: Could not create " << images_file << "\n";
        return 1;
    }
    fs_img << "images" << training_images;
    fs_img.release();
    std::cout << "  Saved: " << images_file << "\n";

    // Optionally verify by training KNN
    std::cout << "\nVerifying training data...\n";

    cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
    knn->setDefaultK(1);

    cv::Mat training_float;
    training_images.convertTo(training_float, CV_32F);

    if (knn->train(training_float, cv::ml::ROW_SAMPLE, classifications)) {
        std::cout << "  KNN training verification: SUCCESS\n";

        // Quick accuracy test with random samples
        std::cout << "\nQuick self-test (testing on training data):\n";

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, total_samples - 1);

        int test_count = std::min(10, total_samples);
        int correct = 0;

        for (int i = 0; i < test_count; i++) {
            int idx = dis(gen);
            cv::Mat sample = training_float.row(idx);
            cv::Mat result;
            knn->findNearest(sample, 1, result);

            int predicted = static_cast<int>(result.at<float>(0, 0));
            int actual = classifications.at<int>(idx, 0);

            char pred_char = static_cast<char>(predicted);
            char actual_char = static_cast<char>(actual);

            if (predicted == actual) {
                correct++;
                std::cout << "  Sample " << idx << ": '" << actual_char
                          << "' -> '" << pred_char << "' ✓\n";
            } else {
                std::cout << "  Sample " << idx << ": '" << actual_char
                          << "' -> '" << pred_char << "' ✗\n";
            }
        }

        std::cout << "\n  Self-test accuracy: " << correct << "/" << test_count
                  << " (" << (100.0 * correct / test_count) << "%)\n";
    } else {
        std::cerr << "  KNN training verification: FAILED\n";
        return 1;
    }

    std::cout << "\n========================================\n";
    std::cout << "Training complete!\n";
    std::cout << "========================================\n";
    std::cout << "\nTrained for " << LICENSE_PLATE_LENGTH << "-character license plates\n";
    std::cout << "Characters supported: 0-9, A-Z (" << char_counts.size() << " unique)\n";
    std::cout << "\nGenerated files:\n";
    std::cout << "  - " << classifications_file << "\n";
    std::cout << "  - " << images_file << "\n";
    std::cout << "\nCopy these files to your ALPR project root directory.\n";

    return 0;
}
