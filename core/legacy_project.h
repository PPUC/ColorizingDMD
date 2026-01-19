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
    std::vector<cv::Mat> frames_x;
    std::vector<uint8_t> frame_extra_flags;
    std::vector<cv::Mat> sprites;
    std::vector<cv::Mat> sprites_x;
    std::vector<cv::Mat> sprite_colored;
    std::vector<cv::Mat> sprite_colored_x;
    std::vector<cv::Mat> sprite_originals;
    std::vector<cv::Mat> sprite_masks_x;
    std::vector<cv::Mat> sprite_dynamic_masks;
    std::vector<cv::Mat> sprite_dynamic_masks_x;
    std::vector<std::vector<uint16_t>> sprite_dynamic_colors;
    std::vector<std::vector<uint16_t>> sprite_dynamic_colors_x;
    std::vector<uint8_t> sprite_extra_flags;
    std::vector<uint8_t> sprite_shape_modes;
    std::vector<uint16_t> sprite_det_areas;
    std::vector<uint32_t> sprite_det_dwords;
    std::vector<uint16_t> sprite_det_dword_pos;
    std::vector<uint8_t> frame_sprites;
    std::vector<uint16_t> frame_sprite_bboxes;
    std::vector<uint32_t> sprite_col_from_frame;
    std::vector<uint16_t> sprite_rects;
    std::vector<uint32_t> sprite_rect_mirror;
    std::vector<cv::Mat> background_frames;
    std::vector<cv::Mat> background_frames_x;
    std::vector<uint8_t> background_extra_flags;
    std::vector<uint16_t> background_ids;
    std::vector<cv::Mat> background_masks;
    std::vector<cv::Mat> background_masks_x;
    std::vector<cv::Mat> comp_masks;
    std::vector<uint8_t> frame_comp_mask_ids;
    std::vector<uint8_t> frame_shape_comp_modes;
    std::vector<uint16_t> frame_rotations;
    std::vector<uint16_t> frame_rotations_x;
    std::vector<cv::Mat> frame_dynamic_mask_maps;
    std::vector<cv::Mat> frame_dynamic_mask_maps_x;
    std::vector<cv::Mat> frame_refs;
    std::vector<std::vector<uint16_t>> frame_dynamic_colors;
    std::vector<std::string> frame_labels;
    std::vector<std::string> sprite_labels;
    std::vector<uint32_t> frame_durations;
    std::vector<uint32_t> section_firsts;
    std::vector<std::string> section_names;
    std::vector<uint16_t> palettes;
    std::vector<std::string> palette_names;
    std::vector<uint32_t> hash_codes;
    std::vector<uint8_t> active_frames;
    std::vector<uint32_t> trigger_ids;
    std::vector<uint16_t> reduced_palettes;
    std::vector<std::string> reduced_palette_names;
    uint8_t active_reduced_palette = 0;
    uint8_t preview_reduced_palette = 0;
    std::vector<uint32_t> active_col_sets;
    std::vector<char> mask_names;
    uint32_t draw_col_mode = 0;
    uint8_t draw_mode = 0;
    int32_t mask_sel_mode = 0;
    uint32_t fill_mode = 0;
    std::vector<uint16_t> edit_colors;
    uint32_t n_image_pos_saves = 0;
    std::vector<char> image_pos_names;
    std::vector<int32_t> image_pos_data;
    uint32_t is_imported = 0;
    uint32_t time_elapsed = 0;
    uint32_t is_pup_pack = 0;
    std::vector<char> pup_pack;
    std::vector<uint8_t> dynashadow_dir;
    std::vector<uint16_t> dynashadow_col;
    std::vector<uint8_t> dynashadow_dir_x;
    std::vector<uint16_t> dynashadow_col_x;
    uint32_t no_colors = 64;
};

bool LoadLegacyProjectFromConcentrate(const std::string& cromc_path,
                                      const std::string& rp_path,
                                      LegacyProject& out,
                                      std::string* error,
                                      bool skip_frame_images = false);
