#include "legacy_project_writer.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <vector>

#include "serum_constants.h"

namespace {
bool WriteExact(std::ofstream& file, const void* src, std::size_t size)
{
    return static_cast<bool>(file.write(reinterpret_cast<const char*>(src), static_cast<std::streamsize>(size)));
}

uint16_t BgrToRgb565(const cv::Vec3b& color)
{
    const uint8_t b = color[0];
    const uint8_t g = color[1];
    const uint8_t r = color[2];
    const uint16_t r5 = static_cast<uint16_t>(r >> 3);
    const uint16_t g6 = static_cast<uint16_t>(g >> 2);
    const uint16_t b5 = static_cast<uint16_t>(b >> 3);
    return static_cast<uint16_t>((r5 << 11) | (g6 << 5) | b5);
}

cv::Mat EnsureBgr(const cv::Mat& source, const cv::Size& target)
{
    if (source.empty()) {
        return cv::Mat();
    }
    cv::Mat bgr;
    if (source.channels() == 3) {
        bgr = source;
    } else if (source.channels() == 4) {
        cv::cvtColor(source, bgr, cv::COLOR_BGRA2BGR);
    } else if (source.channels() == 1) {
        cv::cvtColor(source, bgr, cv::COLOR_GRAY2BGR);
    } else {
        bgr = source.clone();
    }
    if (bgr.size() != target) {
        cv::Mat resized;
        cv::resize(bgr, resized, target, 0.0, 0.0, cv::INTER_NEAREST);
        return resized;
    }
    return bgr.clone();
}

cv::Mat BuildReferenceFrame(const cv::Mat& source, const cv::Size& target, uint32_t no_colors)
{
    if (source.empty()) {
        return cv::Mat();
    }
    cv::Mat bgr = EnsureBgr(source, target);
    if (bgr.empty()) {
        return cv::Mat();
    }
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat ref(gray.rows, gray.cols, CV_8UC1);
    const int levels = no_colors > 0 ? static_cast<int>(no_colors) : 64;
    for (int y = 0; y < gray.rows; ++y) {
        const uint8_t* src = gray.ptr<uint8_t>(y);
        uint8_t* dst = ref.ptr<uint8_t>(y);
        for (int x = 0; x < gray.cols; ++x) {
            const int value = static_cast<int>((src[x] * (levels - 1) + 127) / 255);
            dst[x] = static_cast<uint8_t>(std::clamp(value, 0, levels - 1));
        }
    }
    return ref;
}

std::string BaseName(const std::string& path)
{
    const std::size_t slash = path.find_last_of("/\\");
    const std::string file = (slash == std::string::npos) ? path : path.substr(slash + 1);
    const std::size_t dot = file.find_last_of('.');
    return (dot == std::string::npos) ? file : file.substr(0, dot);
}

void WritePaddedName(char* dst, std::size_t size, const std::string& name)
{
    std::fill(dst, dst + size, '\0');
    const std::size_t copy = std::min(name.size(), size - 1);
    std::copy_n(name.data(), copy, dst);
}

void WriteFixedString(std::vector<char>& buffer, std::size_t offset, std::size_t size, const std::string& value)
{
    const std::size_t copy = std::min(value.size(), size - 1);
    std::fill(buffer.begin() + offset, buffer.begin() + offset + size, '\0');
    std::copy_n(value.data(), copy, buffer.begin() + offset);
}
}

bool SaveLegacyProject(const std::string& crom_path,
                       const std::string& rp_path,
                       const LegacyProject& project,
                       std::string* error)
{
    if (project.frames.empty()) {
        if (error) {
            *error = "No frames to save";
        }
        return false;
    }

    const uint32_t frame_width = static_cast<uint32_t>(project.frames.front().cols);
    const uint32_t frame_height = static_cast<uint32_t>(project.frames.front().rows);
    if (frame_width == 0 || frame_height == 0) {
        if (error) {
            *error = "Invalid frame size";
        }
        return false;
    }

    const uint32_t frame_width_x = frame_width;
    const uint32_t frame_height_x = frame_height;
    const uint32_t n_frames = static_cast<uint32_t>(project.frames.size());
    const uint32_t n_sprites = static_cast<uint32_t>(project.sprites.size());
    const uint32_t no_colors = project.no_colors > 0 ? project.no_colors : 64;
    uint32_t n_comp_masks = 0;
    const uint16_t n_backgrounds = 0;
    const uint32_t length_header = 14 * sizeof(uint32_t);

    const std::size_t mask_pixels = static_cast<std::size_t>(frame_width) * frame_height;
    if (!project.comp_masks.empty() || !project.frame_comp_mask_ids.empty()) {
        const std::size_t max_masks = std::min(project.comp_masks.size(), static_cast<std::size_t>(MAX_MASKS));
        for (std::size_t i = 0; i < max_masks; ++i) {
            const cv::Mat& mask = project.comp_masks[i];
            if (!mask.empty() && mask.rows == static_cast<int>(frame_height) && mask.cols == static_cast<int>(frame_width)) {
                bool has_pixels = false;
                for (std::size_t j = 0; j < mask_pixels; ++j) {
                    if (mask.data[j]) {
                        has_pixels = true;
                        break;
                    }
                }
                if (has_pixels) {
                    n_comp_masks = static_cast<uint32_t>(std::max<std::size_t>(n_comp_masks, i + 1));
                }
            }
        }
        for (std::size_t i = 0; i < project.frame_comp_mask_ids.size(); ++i) {
            if (project.frame_comp_mask_ids[i] != 255) {
                n_comp_masks = static_cast<uint32_t>(std::max<std::size_t>(n_comp_masks,
                    static_cast<std::size_t>(project.frame_comp_mask_ids[i]) + 1));
            }
        }
    }

    std::vector<uint8_t> comp_masks(n_comp_masks * mask_pixels, 0);
    if (n_comp_masks > 0 && !project.comp_masks.empty()) {
        for (uint32_t i = 0; i < n_comp_masks && i < project.comp_masks.size(); ++i) {
            const cv::Mat& mask = project.comp_masks[i];
            if (!mask.empty() && mask.rows == static_cast<int>(frame_height) && mask.cols == static_cast<int>(frame_width)) {
                std::memcpy(comp_masks.data() + static_cast<std::size_t>(i) * mask_pixels, mask.data, mask_pixels);
            }
        }
    }

    std::ofstream crom(crom_path, std::ios::binary);
    if (!crom) {
        if (error) {
            *error = "Could not open .cROM for writing";
        }
        return false;
    }

    char name[64]{};
    WritePaddedName(name, sizeof(name), project.name.empty() ? BaseName(crom_path) : project.name);
    if (!WriteExact(crom, name, sizeof(name)) ||
        !WriteExact(crom, &length_header, sizeof(length_header)) ||
        !WriteExact(crom, &frame_width, sizeof(frame_width)) ||
        !WriteExact(crom, &frame_height, sizeof(frame_height)) ||
        !WriteExact(crom, &frame_width_x, sizeof(frame_width_x)) ||
        !WriteExact(crom, &frame_height_x, sizeof(frame_height_x)) ||
        !WriteExact(crom, &n_frames, sizeof(n_frames)) ||
        !WriteExact(crom, &no_colors, sizeof(no_colors)) ||
        !WriteExact(crom, &n_comp_masks, sizeof(n_comp_masks)) ||
        !WriteExact(crom, &n_sprites, sizeof(n_sprites)) ||
        !WriteExact(crom, &n_backgrounds, sizeof(n_backgrounds))) {
        if (error) {
            *error = "Failed to write .cROM header";
        }
        return false;
    }

    std::vector<uint32_t> hash_codes(n_frames, 0);
    std::vector<uint8_t> shape_comp(n_frames, 0);
    std::vector<uint8_t> comp_mask_id(n_frames, 255);
    std::vector<uint8_t> extra_frame(n_frames, 0);
    std::vector<uint16_t> frames_565(n_frames * frame_width * frame_height, 0);
    std::vector<uint16_t> frames_x_565(n_frames * frame_width_x * frame_height_x, 0);
    std::vector<uint8_t> dyna_masks(n_frames * frame_width * frame_height, 255);
    std::vector<uint8_t> dyna_masks_x(n_frames * frame_width_x * frame_height_x, 255);
    std::vector<uint16_t> dyna_cols(n_frames * MAX_DYNA_SETS_PER_FRAMEN * no_colors, 0);
    std::vector<uint16_t> dyna_cols_x(dyna_cols.size(), 0);
    std::vector<uint8_t> extra_sprite(n_sprites, 0);
    std::vector<uint8_t> frame_sprites(n_frames * MAX_SPRITES_PER_FRAME, 255);
    std::vector<uint8_t> sprite_original(n_sprites * MAX_SPRITE_WIDTH * MAX_SPRITE_HEIGHT, 255);
    std::vector<uint16_t> sprite_colored(n_sprites * MAX_SPRITE_WIDTH * MAX_SPRITE_HEIGHT, 0);
    std::vector<uint8_t> sprite_mask_x(sprite_original.size(), 255);
    std::vector<uint16_t> sprite_colored_x(sprite_colored.size(), 0);

    if (!project.frame_comp_mask_ids.empty()) {
        for (std::size_t i = 0; i < std::min(project.frame_comp_mask_ids.size(),
                                             static_cast<std::size_t>(comp_mask_id.size())); ++i) {
            comp_mask_id[i] = project.frame_comp_mask_ids[i];
        }
    }
    if (!project.frame_shape_comp_modes.empty()) {
        for (std::size_t i = 0; i < std::min(project.frame_shape_comp_modes.size(),
                                             static_cast<std::size_t>(shape_comp.size())); ++i) {
            shape_comp[i] = project.frame_shape_comp_modes[i];
        }
    }

    for (uint32_t index = 0; index < n_frames; ++index) {
        cv::Mat frame = EnsureBgr(project.frames[index], cv::Size(frame_width, frame_height));
        const std::size_t offset = static_cast<std::size_t>(index) * frame_width * frame_height;
        for (uint32_t y = 0; y < frame_height; ++y) {
            const cv::Vec3b* row = frame.ptr<cv::Vec3b>(static_cast<int>(y));
            for (uint32_t x = 0; x < frame_width; ++x) {
                frames_565[offset + y * frame_width + x] = BgrToRgb565(row[x]);
            }
        }
        if (index < project.frame_dynamic_mask_ids.size()) {
            const uint8_t dyn_id = project.frame_dynamic_mask_ids[index];
            if (dyn_id < MAX_DYNA_SETS_PER_FRAMEN && dyn_id < project.dynamic_masks.size()) {
                const cv::Mat& mask = project.dynamic_masks[dyn_id];
                if (!mask.empty() && mask.rows == static_cast<int>(frame_height) && mask.cols == static_cast<int>(frame_width)) {
                    for (uint32_t y = 0; y < frame_height; ++y) {
                        const uint8_t* mrow = mask.ptr<uint8_t>(static_cast<int>(y));
                        for (uint32_t x = 0; x < frame_width; ++x) {
                            if (mrow[x]) {
                                dyna_masks[offset + y * frame_width + x] = dyn_id;
                            }
                        }
                    }
                }
            }
        }
        if (index < project.frame_dynamic_colors.size()) {
            const std::vector<uint16_t>& colors = project.frame_dynamic_colors[index];
            const std::size_t colors_offset = static_cast<std::size_t>(index) * MAX_DYNA_SETS_PER_FRAMEN * no_colors;
            if (colors.size() >= MAX_DYNA_SETS_PER_FRAMEN * no_colors) {
                std::memcpy(dyna_cols.data() + colors_offset, colors.data(),
                            MAX_DYNA_SETS_PER_FRAMEN * no_colors * sizeof(uint16_t));
            }
        }
    }

    for (uint32_t index = 0; index < n_sprites; ++index) {
        cv::Mat sprite = EnsureBgr(project.sprites[index], cv::Size(MAX_SPRITE_WIDTH, MAX_SPRITE_HEIGHT));
        const std::size_t offset = static_cast<std::size_t>(index) * MAX_SPRITE_WIDTH * MAX_SPRITE_HEIGHT;
        for (uint32_t y = 0; y < MAX_SPRITE_HEIGHT; ++y) {
            const cv::Vec3b* row = sprite.ptr<cv::Vec3b>(static_cast<int>(y));
            for (uint32_t x = 0; x < MAX_SPRITE_WIDTH; ++x) {
                sprite_original[offset + y * MAX_SPRITE_WIDTH + x] = 0;
                sprite_colored[offset + y * MAX_SPRITE_WIDTH + x] = BgrToRgb565(row[x]);
            }
        }
    }

    if (!WriteExact(crom, hash_codes.data(), hash_codes.size() * sizeof(uint32_t)) ||
        !WriteExact(crom, shape_comp.data(), shape_comp.size()) ||
        !WriteExact(crom, comp_mask_id.data(), comp_mask_id.size()) ||
        !WriteExact(crom, comp_masks.data(), comp_masks.size()) ||
        !WriteExact(crom, extra_frame.data(), extra_frame.size()) ||
        !WriteExact(crom, frames_565.data(), frames_565.size() * sizeof(uint16_t)) ||
        !WriteExact(crom, frames_x_565.data(), frames_x_565.size() * sizeof(uint16_t)) ||
        !WriteExact(crom, dyna_masks.data(), dyna_masks.size()) ||
        !WriteExact(crom, dyna_masks_x.data(), dyna_masks_x.size()) ||
        !WriteExact(crom, dyna_cols.data(), dyna_cols.size() * sizeof(uint16_t)) ||
        !WriteExact(crom, dyna_cols_x.data(), dyna_cols_x.size() * sizeof(uint16_t)) ||
        !WriteExact(crom, extra_sprite.data(), extra_sprite.size()) ||
        !WriteExact(crom, frame_sprites.data(), frame_sprites.size()) ||
        !WriteExact(crom, sprite_original.data(), sprite_original.size()) ||
        !WriteExact(crom, sprite_colored.data(), sprite_colored.size() * sizeof(uint16_t)) ||
        !WriteExact(crom, sprite_mask_x.data(), sprite_mask_x.size()) ||
        !WriteExact(crom, sprite_colored_x.data(), sprite_colored_x.size() * sizeof(uint16_t))) {
        if (error) {
            *error = "Failed to write .cROM payload";
        }
        return false;
    }

    std::vector<uint8_t> active_frames(n_frames, 0);
    if (!WriteExact(crom, active_frames.data(), active_frames.size())) {
        if (error) {
            *error = "Failed to write .cROM active frames";
        }
        return false;
    }

    const std::size_t rotations_count = static_cast<std::size_t>(n_frames) * MAX_COLOR_ROTATIONN * MAX_LENGTH_COLOR_ROTATION;
    std::vector<uint16_t> rotations(rotations_count, 0);
    if (!WriteExact(crom, rotations.data(), rotations.size() * sizeof(uint16_t)) ||
        !WriteExact(crom, rotations.data(), rotations.size() * sizeof(uint16_t))) {
        if (error) {
            *error = "Failed to write .cROM rotations";
        }
        return false;
    }

    const std::size_t det_dwords_count = static_cast<std::size_t>(n_sprites) * MAX_SPRITE_DETECT_AREAS;
    std::vector<uint32_t> det_dwords(det_dwords_count, 0);
    std::vector<uint16_t> det_dword_pos(det_dwords_count, 0);
    std::vector<uint16_t> det_areas(det_dwords_count * 4, 0xffff);
    if (!WriteExact(crom, det_dwords.data(), det_dwords.size() * sizeof(uint32_t)) ||
        !WriteExact(crom, det_dword_pos.data(), det_dword_pos.size() * sizeof(uint16_t)) ||
        !WriteExact(crom, det_areas.data(), det_areas.size() * sizeof(uint16_t))) {
        if (error) {
            *error = "Failed to write .cROM sprite detection";
        }
        return false;
    }

    std::vector<uint32_t> trigger_ids(n_frames, 0xffffffffu);
    if (!WriteExact(crom, trigger_ids.data(), trigger_ids.size() * sizeof(uint32_t))) {
        if (error) {
            *error = "Failed to write .cROM trigger IDs";
        }
        return false;
    }

    std::vector<uint16_t> frame_sprite_bb(n_frames * MAX_SPRITES_PER_FRAME * 4, 0);
    for (uint32_t i = 0; i < n_frames; ++i) {
        for (uint32_t j = 0; j < MAX_SPRITES_PER_FRAME; ++j) {
            const std::size_t offset = (static_cast<std::size_t>(i) * MAX_SPRITES_PER_FRAME + j) * 4;
            frame_sprite_bb[offset + 2] = static_cast<uint16_t>(frame_width - 1);
            frame_sprite_bb[offset + 3] = static_cast<uint16_t>(frame_height - 1);
        }
    }
    if (!WriteExact(crom, frame_sprite_bb.data(), frame_sprite_bb.size() * sizeof(uint16_t))) {
        if (error) {
            *error = "Failed to write .cROM frame sprite bounds";
        }
        return false;
    }

    std::vector<uint16_t> background_ids(n_frames, 0xffff);
    std::vector<uint8_t> background_mask(n_frames * frame_width * frame_height, 0);
    std::vector<uint8_t> background_mask_x(n_frames * frame_width_x * frame_height_x, 0);
    if (!WriteExact(crom, background_ids.data(), background_ids.size() * sizeof(uint16_t)) ||
        !WriteExact(crom, background_mask.data(), background_mask.size()) ||
        !WriteExact(crom, background_mask_x.data(), background_mask_x.size())) {
        if (error) {
            *error = "Failed to write .cROM background info";
        }
        return false;
    }

    crom.close();

    if (rp_path.empty()) {
        return true;
    }

    std::ofstream crp(rp_path, std::ios::binary);
    if (!crp) {
        if (error) {
            *error = "Could not open .cRP for writing";
        }
        return false;
    }

    char rpname[64]{};
    WritePaddedName(rpname, sizeof(rpname), project.name.empty() ? BaseName(rp_path) : project.name);
    if (!WriteExact(crp, rpname, sizeof(rpname))) {
        if (error) {
            *error = "Failed to write .cRP header";
        }
        return false;
    }

    std::vector<uint8_t> o_frames(n_frames * frame_width * frame_height, 0);
    for (uint32_t index = 0; index < n_frames; ++index) {
        const std::size_t offset = static_cast<std::size_t>(index) * frame_width * frame_height;
        if (index < project.frame_refs.size()) {
            const cv::Mat& ref = project.frame_refs[index];
            if (!ref.empty() && ref.rows == static_cast<int>(frame_height) && ref.cols == static_cast<int>(frame_width)) {
                std::memcpy(o_frames.data() + offset, ref.data, frame_width * frame_height);
                continue;
            }
        }
        if (index < project.frames.size()) {
            cv::Mat ref = BuildReferenceFrame(project.frames[index], cv::Size(frame_width, frame_height), no_colors);
            if (!ref.empty()) {
                std::memcpy(o_frames.data() + offset, ref.data, frame_width * frame_height);
            }
        }
    }
    std::vector<uint32_t> active_col_set(MAX_COL_SETS, 0);
    std::vector<uint16_t> col_sets(MAX_COL_SETS * 16, 0);
    uint8_t ac_col_set = 0;
    uint8_t pre_col_set = 0;
    std::vector<char> name_col_set(MAX_COL_SETS * 64, 0);
    uint32_t draw_col_mode = 0;
    uint8_t draw_mode = 0;
    int32_t mask_sel_mode = 0;
    uint32_t fill_mode = 0;
    std::vector<char> mask_names(MAX_MASKS * SIZE_MASK_NAME, 0);

    uint32_t n_sections = static_cast<uint32_t>(project.section_firsts.size());
    n_sections = std::min(n_sections, static_cast<uint32_t>(MAX_SECTIONS));
    std::vector<uint32_t> section_firsts(MAX_SECTIONS, 0);
    for (uint32_t i = 0; i < n_sections; ++i) {
        section_firsts[i] = project.section_firsts[i];
    }
    std::vector<char> section_names(MAX_SECTIONS * SIZE_SECTION_NAMES, 0);
    for (uint32_t i = 0; i < n_sections && i < project.section_names.size(); ++i) {
        WriteFixedString(section_names, i * SIZE_SECTION_NAMES, SIZE_SECTION_NAMES, project.section_names[i]);
    }

    std::vector<char> sprite_names(MAX_SPRITES * SIZE_SECTION_NAMES, 0);
    for (uint32_t i = 0; i < n_sprites && i < project.sprite_labels.size() && i < MAX_SPRITES; ++i) {
        WriteFixedString(sprite_names, i * SIZE_SECTION_NAMES, SIZE_SECTION_NAMES, project.sprite_labels[i]);
    }

    std::vector<uint32_t> sprite_col_from_frame(MAX_SPRITES, 0);
    std::vector<uint32_t> frame_duration(n_frames, 30);
    if (!project.frame_durations.empty()) {
        for (uint32_t i = 0; i < n_frames && i < project.frame_durations.size(); ++i) {
            frame_duration[i] = project.frame_durations[i];
        }
    }

    if (!WriteExact(crp, o_frames.data(), o_frames.size()) ||
        !WriteExact(crp, active_col_set.data(), active_col_set.size() * sizeof(uint32_t)) ||
        !WriteExact(crp, col_sets.data(), col_sets.size() * sizeof(uint16_t)) ||
        !WriteExact(crp, &ac_col_set, sizeof(ac_col_set)) ||
        !WriteExact(crp, &pre_col_set, sizeof(pre_col_set)) ||
        !WriteExact(crp, name_col_set.data(), name_col_set.size()) ||
        !WriteExact(crp, &draw_col_mode, sizeof(draw_col_mode)) ||
        !WriteExact(crp, &draw_mode, sizeof(draw_mode)) ||
        !WriteExact(crp, &mask_sel_mode, sizeof(mask_sel_mode)) ||
        !WriteExact(crp, &fill_mode, sizeof(fill_mode)) ||
        !WriteExact(crp, mask_names.data(), mask_names.size()) ||
        !WriteExact(crp, &n_sections, sizeof(n_sections)) ||
        !WriteExact(crp, section_firsts.data(), section_firsts.size() * sizeof(uint32_t)) ||
        !WriteExact(crp, section_names.data(), section_names.size()) ||
        !WriteExact(crp, sprite_names.data(), sprite_names.size()) ||
        !WriteExact(crp, sprite_col_from_frame.data(), sprite_col_from_frame.size() * sizeof(uint32_t)) ||
        !WriteExact(crp, frame_duration.data(), frame_duration.size() * sizeof(uint32_t))) {
        if (error) {
            *error = "Failed to write .cRP core data";
        }
        return false;
    }

    std::vector<uint16_t> sprite_rect(4 * MAX_SPRITES, 0);
    std::vector<uint32_t> sprite_rect_mirror(2 * MAX_SPRITES, 0);
    std::vector<uint16_t> palette(N_PALETTES * 64, 0);
    std::vector<uint16_t> edit_colors(16, 0);
    uint32_t n_image_pos_saves = 0;
    std::vector<char> image_pos_name(N_IMAGE_POS_TO_SAVE * 64, 0);
    std::vector<int32_t> image_pos(N_IMAGE_POS_TO_SAVE * 16, 0);
    std::vector<char> pal_names(N_PALETTES * 64, 0);
    uint32_t is_imported = 0;
    uint32_t time_elapsed = 0;
    uint32_t is_pup_pack = 0;
    std::vector<char> pup_pack(sizeof(wchar_t) * 256, 0);

    if (!WriteExact(crp, sprite_rect.data(), sprite_rect.size() * sizeof(uint16_t)) ||
        !WriteExact(crp, sprite_rect_mirror.data(), sprite_rect_mirror.size() * sizeof(uint32_t)) ||
        !WriteExact(crp, palette.data(), palette.size() * sizeof(uint16_t)) ||
        !WriteExact(crp, edit_colors.data(), edit_colors.size() * sizeof(uint16_t)) ||
        !WriteExact(crp, &n_image_pos_saves, sizeof(n_image_pos_saves)) ||
        !WriteExact(crp, image_pos_name.data(), image_pos_name.size()) ||
        !WriteExact(crp, image_pos.data(), image_pos.size() * sizeof(int32_t)) ||
        !WriteExact(crp, pal_names.data(), pal_names.size()) ||
        !WriteExact(crp, &is_imported, sizeof(is_imported)) ||
        !WriteExact(crp, &time_elapsed, sizeof(time_elapsed)) ||
        !WriteExact(crp, &is_pup_pack, sizeof(is_pup_pack)) ||
        !WriteExact(crp, pup_pack.data(), pup_pack.size())) {
        if (error) {
            *error = "Failed to write .cRP tail data";
        }
        return false;
    }

    return true;
}
