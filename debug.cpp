#include "debug.hpp"
#include <filesystem>
#include <iostream>

namespace debug {

static std::string g_output_dir = "out";
static bool g_enabled = false;

void set_output_dir(const std::string& dir) {
    g_output_dir = dir;
}

void set_enabled(bool enabled) {
    g_enabled = enabled;
}

bool is_enabled() {
    return g_enabled;
}

void save_image(const cv::Mat& img, const std::string& stage, const std::string& name) {
    if (!g_enabled) return;

    namespace fs = std::filesystem;
    std::string debug_dir = g_output_dir + "/debug";
    fs::create_directories(debug_dir);

    std::string filename = debug_dir + "/" + stage + "_" + name + ".jpg";
    cv::imwrite(filename, img);
    std::cout << "  [debug] " << stage << "/" << name << "\n";
}

void save_image(const cv::Mat& img, const std::string& stage, int index, const std::string& name) {
    if (!g_enabled) return;

    namespace fs = std::filesystem;
    std::string debug_dir = g_output_dir + "/debug";
    fs::create_directories(debug_dir);

    std::string filename = debug_dir + "/" + stage + "_" + std::to_string(index) + "_" + name + ".jpg";
    cv::imwrite(filename, img);
    std::cout << "  [debug] " << stage << "/" << index << "_" << name << "\n";
}

}
