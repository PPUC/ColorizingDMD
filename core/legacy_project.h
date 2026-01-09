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
    std::vector<cv::Mat> comp_masks;
    std::vector<uint8_t> frame_comp_mask_ids;
    std::vector<cv::Mat> dynamic_masks;
    std::vector<uint8_t> frame_dynamic_mask_ids;
    std::vector<cv::Mat> frame_refs;
    std::vector<std::vector<uint16_t>> frame_dynamic_colors;
    std::vector<std::string> frame_labels;
    std::vector<std::string> sprite_labels;
    std::vector<uint32_t> frame_durations;
    std::vector<uint32_t> section_firsts;
    std::vector<std::string> section_names;
    uint32_t no_colors = 64;
};

bool LoadLegacyProject(const std::string& path,
                       const std::string& rp_path,
                       LegacyProject& out,
                       std::string* error);
