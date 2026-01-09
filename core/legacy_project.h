#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

struct LegacyProject {
    std::string name;
    uint32_t frame_width = 0;
    uint32_t frame_height = 0;
    uint32_t frame_width_x = 0;
    uint32_t frame_height_x = 0;
    uint32_t sprite_width = 0;
    uint32_t sprite_height = 0;
    std::vector<cv::Mat> frames;
    std::vector<cv::Mat> sprites;
    std::vector<std::string> frame_labels;
    std::vector<std::string> sprite_labels;
    std::vector<uint32_t> frame_durations;
    std::vector<uint32_t> section_firsts;
    std::vector<std::string> section_names;
};

bool LoadLegacyProject(const std::string& path,
                       const std::string& rp_path,
                       LegacyProject& out,
                       std::string* error);
