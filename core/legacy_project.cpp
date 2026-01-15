#include "legacy_project.h"

#include <algorithm>
#include <cstring>
#include <fstream>

#include "serum_constants.h"

namespace {
bool ReadExact(std::ifstream& file, void* dst, std::size_t size)
{
    return static_cast<bool>(file.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(size)));
}

std::size_t BytesRemaining(std::ifstream& file)
{
    const std::streampos current = file.tellg();
    if (current < 0) {
        return 0;
    }
    file.seekg(0, std::ios::end);
    const std::streampos end = file.tellg();
    if (end < 0) {
        file.seekg(current);
        return 0;
    }
    file.seekg(current);
    const std::streamoff remaining = end - current;
    if (remaining <= 0) {
        return 0;
    }
    return static_cast<std::size_t>(remaining);
}

bool SkipExact(std::ifstream& file, std::size_t size)
{
    return static_cast<bool>(file.seekg(static_cast<std::streamoff>(size), std::ios::cur));
}

cv::Vec3b Rgb565ToBgr(uint16_t value)
{
    const uint8_t r5 = static_cast<uint8_t>((value >> 11) & 0x1f);
    const uint8_t g6 = static_cast<uint8_t>((value >> 5) & 0x3f);
    const uint8_t b5 = static_cast<uint8_t>(value & 0x1f);
    const uint8_t r8 = static_cast<uint8_t>((r5 << 3) | (r5 >> 2));
    const uint8_t g8 = static_cast<uint8_t>((g6 << 2) | (g6 >> 4));
    const uint8_t b8 = static_cast<uint8_t>((b5 << 3) | (b5 >> 2));
    return cv::Vec3b(b8, g8, r8);
}

std::string TrimName(const char* buffer, std::size_t size)
{
    std::size_t length = 0;
    while (length < size && buffer[length] != '\0') {
        ++length;
    }
    return std::string(buffer, length);
}

std::string TrimName(const std::vector<char>& buffer, std::size_t offset, std::size_t size)
{
    return TrimName(buffer.data() + offset, size);
}

bool LoadLegacyReferenceFrames(const std::string& path,
                               std::size_t pixel_count,
                               std::vector<uint8_t>& out,
                               std::string* error)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        if (error) {
            *error = "Could not open .cRP file";
        }
        return false;
    }
    char name[64]{};
    if (!ReadExact(file, name, sizeof(name))) {
        if (error) {
            *error = "Invalid .cRP file header";
        }
        return false;
    }
    out.resize(pixel_count);
    if (pixel_count > 0) {
        if (!ReadExact(file, out.data(), pixel_count)) {
            if (error) {
                *error = "Unexpected end of .cRP file";
            }
            return false;
        }
    }
    return true;
}

bool LoadLegacyMetadataFromRP(const std::string& path,
                              uint32_t frame_width,
                              uint32_t frame_height,
                              uint32_t n_frames,
                              uint32_t n_sprites,
                              LegacyProject& out,
                              std::vector<uint32_t>& frame_durations,
                              std::vector<std::string>& sprite_names,
                              std::vector<uint32_t>& section_firsts,
                              std::vector<std::string>& section_names,
                              std::string* error)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        if (error) {
            *error = "Could not open .cRP file";
        }
        return false;
    }

    char name[64]{};
    if (!ReadExact(file, name, sizeof(name))) {
        if (error) {
            *error = "Invalid .cRP file header";
        }
        return false;
    }

    const std::size_t frame_pixels = static_cast<std::size_t>(n_frames) * frame_width * frame_height;
    if (!SkipExact(file, frame_pixels)) {
        if (error) {
            *error = "Unexpected end of .cRP file";
        }
        return false;
    }

    constexpr std::size_t kBoolSize = 4;
    std::vector<uint32_t> active_col_set(MAX_COL_SETS, 0);
    if (!ReadExact(file, active_col_set.data(), active_col_set.size() * sizeof(uint32_t))) {
        if (error) {
            *error = "Unexpected end of .cRP file (col sets)";
        }
        return false;
    }

    out.reduced_palettes.resize(MAX_COL_SETS * 16);
    if (!ReadExact(file, out.reduced_palettes.data(),
                   out.reduced_palettes.size() * sizeof(uint16_t))) {
        if (error) {
            *error = "Unexpected end of .cRP file (color sets)";
        }
        return false;
    }

    if (!ReadExact(file, &out.active_reduced_palette, sizeof(uint8_t)) ||
        !ReadExact(file, &out.preview_reduced_palette, sizeof(uint8_t))) {
        if (error) {
            *error = "Unexpected end of .cRP file (color set indices)";
        }
        return false;
    }

    std::vector<char> name_col_set(MAX_COL_SETS * 64, 0);
    if (!ReadExact(file, name_col_set.data(), name_col_set.size())) {
        if (error) {
            *error = "Unexpected end of .cRP file (color set names)";
        }
        return false;
    }
    out.reduced_palette_names.clear();
    out.reduced_palette_names.reserve(MAX_COL_SETS);
    for (int i = 0; i < MAX_COL_SETS; ++i) {
        out.reduced_palette_names.push_back(TrimName(name_col_set, i * 64, 64));
    }

    uint32_t draw_col_mode = 0;
    uint8_t draw_mode = 0;
    int32_t mask_sel_mode = 0;
    uint32_t fill_mode = 0;
    if (!ReadExact(file, &draw_col_mode, sizeof(draw_col_mode)) ||
        !ReadExact(file, &draw_mode, sizeof(draw_mode)) ||
        !ReadExact(file, &mask_sel_mode, sizeof(mask_sel_mode)) ||
        !ReadExact(file, &fill_mode, sizeof(fill_mode))) {
        if (error) {
            *error = "Unexpected end of .cRP file (draw settings)";
        }
        return false;
    }

    if (!SkipExact(file, MAX_MASKS * SIZE_MASK_NAME)) {
        if (error) {
            *error = "Unexpected end of .cRP file (mask names)";
        }
        return false;
    }

    uint32_t n_sections = 0;
    if (!ReadExact(file, &n_sections, sizeof(n_sections))) {
        if (error) {
            *error = "Unexpected end of .cRP file (sections)";
        }
        return false;
    }

    std::vector<uint32_t> section_firsts_raw(MAX_SECTIONS);
    if (!ReadExact(file, section_firsts_raw.data(), MAX_SECTIONS * sizeof(uint32_t))) {
        if (error) {
            *error = "Unexpected end of .cRP file (section frames)";
        }
        return false;
    }

    std::vector<char> section_names_raw(MAX_SECTIONS * SIZE_SECTION_NAMES);
    if (!ReadExact(file, section_names_raw.data(), section_names_raw.size())) {
        if (error) {
            *error = "Unexpected end of .cRP file (section names)";
        }
        return false;
    }

    std::vector<char> sprite_names_raw(MAX_SPRITES * SIZE_SECTION_NAMES);
    if (!ReadExact(file, sprite_names_raw.data(), sprite_names_raw.size())) {
        if (error) {
            *error = "Unexpected end of .cRP file (sprite names)";
        }
        return false;
    }

    std::vector<uint32_t> sprite_col_from_frame(MAX_SPRITES, 0);
    if (!ReadExact(file, sprite_col_from_frame.data(), sprite_col_from_frame.size() * sizeof(uint32_t))) {
        if (error) {
            *error = "Unexpected end of .cRP file (sprite colors)";
        }
        return false;
    }

    frame_durations.resize(n_frames);
    if (!ReadExact(file, frame_durations.data(), n_frames * sizeof(uint32_t))) {
        if (error) {
            *error = "Unexpected end of .cRP file (frame durations)";
        }
        return false;
    }

    std::vector<uint16_t> sprite_rects(4 * MAX_SPRITES, 0);
    if (!ReadExact(file, sprite_rects.data(), sprite_rects.size() * sizeof(uint16_t))) {
        if (error) {
            *error = "Unexpected end of .cRP file (sprite rects)";
        }
        return false;
    }
    std::vector<uint32_t> sprite_rect_mirror(2 * MAX_SPRITES, 0);
    if (!ReadExact(file, sprite_rect_mirror.data(), sprite_rect_mirror.size() * sizeof(uint32_t))) {
        if (error) {
            *error = "Unexpected end of .cRP file (sprite rect mirrors)";
        }
        return false;
    }

    out.palettes.resize(N_PALETTES * 64);
    if (!ReadExact(file, out.palettes.data(), out.palettes.size() * sizeof(uint16_t))) {
        if (error) {
            *error = "Unexpected end of .cRP file (palette data)";
        }
        return false;
    }

    if (!SkipExact(file, 16 * sizeof(uint16_t))) {
        if (error) {
            *error = "Unexpected end of .cRP file (edit colors)";
        }
        return false;
    }

    uint32_t n_image_pos_saves = 0;
    if (!ReadExact(file, &n_image_pos_saves, sizeof(n_image_pos_saves))) {
        if (error) {
            *error = "Unexpected end of .cRP file (image positions)";
        }
        return false;
    }
    if (!SkipExact(file, N_IMAGE_POS_TO_SAVE * 64)) {
        if (error) {
            *error = "Unexpected end of .cRP file (image pos names)";
        }
        return false;
    }
    if (!SkipExact(file, N_IMAGE_POS_TO_SAVE * 16 * sizeof(int32_t))) {
        if (error) {
            *error = "Unexpected end of .cRP file (image pos data)";
        }
        return false;
    }

    std::vector<char> pal_names(N_PALETTES * 64, 0);
    if (!ReadExact(file, pal_names.data(), pal_names.size())) {
        if (error) {
            *error = "Unexpected end of .cRP file (palette names)";
        }
        return false;
    }
    out.palette_names.clear();
    out.palette_names.reserve(N_PALETTES);
    for (int i = 0; i < N_PALETTES; ++i) {
        out.palette_names.push_back(TrimName(pal_names, i * 64, 64));
    }

    uint32_t is_imported = 0;
    uint32_t time_elapsed = 0;
    uint32_t is_pup_pack = 0;
    if (!ReadExact(file, &is_imported, sizeof(is_imported)) ||
        !ReadExact(file, &time_elapsed, sizeof(time_elapsed)) ||
        !ReadExact(file, &is_pup_pack, sizeof(is_pup_pack))) {
        if (error) {
            *error = "Unexpected end of .cRP file (import metadata)";
        }
        return false;
    }
    if (!SkipExact(file, sizeof(wchar_t) * 256)) {
        if (error) {
            *error = "Unexpected end of .cRP file (pup pack)";
        }
        return false;
    }

    const uint32_t sections_to_use = std::min<uint32_t>(n_sections, MAX_SECTIONS);
    section_firsts.assign(section_firsts_raw.begin(), section_firsts_raw.begin() + sections_to_use);
    section_names.reserve(sections_to_use);
    for (uint32_t i = 0; i < sections_to_use; ++i) {
        section_names.push_back(TrimName(section_names_raw, i * SIZE_SECTION_NAMES, SIZE_SECTION_NAMES));
    }

    const uint32_t sprites_to_use = std::min<uint32_t>(n_sprites, MAX_SPRITES);
    sprite_names.reserve(sprites_to_use);
    for (uint32_t i = 0; i < sprites_to_use; ++i) {
        sprite_names.push_back(TrimName(sprite_names_raw, i * SIZE_SECTION_NAMES, SIZE_SECTION_NAMES));
    }
    out.sprite_col_from_frame = sprite_col_from_frame;
    out.sprite_rects = sprite_rects;
    out.sprite_rect_mirror = sprite_rect_mirror;

    return true;
}

cv::Mat BuildFrameImage(uint32_t width,
                        uint32_t height,
                        const uint16_t* frame_data,
                        const uint8_t* dyna_mask,
                        const uint16_t* dyna_cols,
                        const uint8_t* ref_frame,
                        uint32_t no_colors)
{
    cv::Mat image(static_cast<int>(height), static_cast<int>(width), CV_8UC3);
    for (uint32_t y = 0; y < height; ++y) {
        auto* row = image.ptr<cv::Vec3b>(static_cast<int>(y));
        for (uint32_t x = 0; x < width; ++x) {
            const std::size_t idx = static_cast<std::size_t>(y) * width + x;
            uint16_t value = frame_data[idx];
            if (dyna_mask && dyna_cols && ref_frame) {
                const uint8_t mask = dyna_mask[idx];
                if (mask != 255 && mask < MAX_DYNA_SETS_PER_FRAMEN) {
                    const uint8_t ref = ref_frame[idx];
                    if (ref < no_colors) {
                        const std::size_t offset =
                            static_cast<std::size_t>(mask) * no_colors + ref;
                        value = dyna_cols[offset];
                    }
                }
            }
            row[x] = Rgb565ToBgr(value);
        }
    }
    return image;
}

cv::Mat ScaleReferenceFrame(const uint8_t* ref_data,
                            uint32_t width,
                            uint32_t height,
                            uint32_t target_width,
                            uint32_t target_height)
{
    if (!ref_data || width == 0 || height == 0 || target_width == 0 || target_height == 0) {
        return cv::Mat();
    }
    cv::Mat source(static_cast<int>(height), static_cast<int>(width), CV_8UC1,
                   const_cast<uint8_t*>(ref_data));
    if (source.cols == static_cast<int>(target_width) && source.rows == static_cast<int>(target_height)) {
        return source.clone();
    }
    cv::Mat resized;
    cv::resize(source, resized, cv::Size(static_cast<int>(target_width), static_cast<int>(target_height)),
               0.0, 0.0, cv::INTER_NEAREST);
    return resized;
}

cv::Mat BuildSpriteImage(uint32_t width,
                         uint32_t height,
                         const uint16_t* sprite_colored,
                         const uint8_t* sprite_original,
                         const uint8_t* dyna_mask,
                         const uint16_t* dyna_cols,
                         uint32_t no_colors)
{
    cv::Mat image(static_cast<int>(height), static_cast<int>(width), CV_8UC3, cv::Scalar(0, 0, 0));
    for (uint32_t y = 0; y < height; ++y) {
        auto* row = image.ptr<cv::Vec3b>(static_cast<int>(y));
        for (uint32_t x = 0; x < width; ++x) {
            const std::size_t idx = static_cast<std::size_t>(y) * width + x;
            if (sprite_original && sprite_original[idx] == 255) {
                continue;
            }
            uint16_t value = sprite_colored ? sprite_colored[idx] : 0;
            if (dyna_mask && dyna_cols && sprite_original) {
                const uint8_t mask = dyna_mask[idx];
                if (mask != 255 && mask < MAX_DYNA_SETS_PER_SPRITE) {
                    const uint8_t ref = sprite_original[idx];
                    if (ref < no_colors) {
                        const std::size_t offset =
                            static_cast<std::size_t>(mask) * no_colors + ref;
                        value = dyna_cols[offset];
                    }
                }
            }
            row[x] = Rgb565ToBgr(value);
        }
    }
    return image;
}
}

bool LoadLegacyProject(const std::string& path,
                       const std::string& rp_path,
                       LegacyProject& out,
                       std::string* error)
{
    out = LegacyProject{};

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        if (error) {
            *error = "Could not open file";
        }
        return false;
    }

    char name[64]{};
    uint32_t length_header = 0;
    uint32_t frame_width = 0;
    uint32_t frame_height = 0;
    uint32_t frame_width_x = 0;
    uint32_t frame_height_x = 0;
    uint32_t n_frames = 0;
    uint32_t no_colors = 0;
    uint32_t n_comp_masks = 0;
    uint32_t n_sprites = 0;
    uint16_t n_backgrounds = 0;

    if (!ReadExact(file, name, sizeof(name)) ||
        !ReadExact(file, &length_header, sizeof(length_header)) ||
        !ReadExact(file, &frame_width, sizeof(frame_width)) ||
        !ReadExact(file, &frame_height, sizeof(frame_height))) {
        if (error) {
            *error = "Invalid or truncated file header";
        }
        return false;
    }

    const bool is_new_format = (length_header >= 14 * sizeof(uint32_t));
    if (!is_new_format) {
        if (error) {
            *error = "Legacy palette-based format is not supported in the Qt port yet.";
        }
        return false;
    }

    if (!ReadExact(file, &frame_width_x, sizeof(frame_width_x)) ||
        !ReadExact(file, &frame_height_x, sizeof(frame_height_x)) ||
        !ReadExact(file, &n_frames, sizeof(n_frames)) ||
        !ReadExact(file, &no_colors, sizeof(no_colors)) ||
        !ReadExact(file, &n_comp_masks, sizeof(n_comp_masks)) ||
        !ReadExact(file, &n_sprites, sizeof(n_sprites))) {
        if (error) {
            *error = "Invalid or truncated file header";
        }
        return false;
    }

    if (length_header >= 13 * sizeof(uint32_t)) {
        if (!ReadExact(file, &n_backgrounds, sizeof(n_backgrounds))) {
            if (error) {
                *error = "Invalid or truncated file header";
            }
            return false;
        }
    }

    std::vector<uint32_t> hash_codes(n_frames, 0);
    std::vector<uint8_t> shape_comp(n_frames, 0);
    std::vector<uint8_t> comp_mask_id(n_frames, 255);
    if (!ReadExact(file, hash_codes.data(), hash_codes.size() * sizeof(uint32_t)) ||
        !ReadExact(file, shape_comp.data(), shape_comp.size()) ||
        !ReadExact(file, comp_mask_id.data(), comp_mask_id.size())) {
        if (error) {
            *error = "Unexpected end of file (header data)";
        }
        return false;
    }

    const std::size_t comp_masks_bytes =
        static_cast<std::size_t>(n_comp_masks) * frame_width * frame_height;
    std::vector<uint8_t> comp_masks;
    if (comp_masks_bytes > 0) {
        comp_masks.resize(comp_masks_bytes);
        if (!ReadExact(file, comp_masks.data(), comp_masks_bytes)) {
            if (error) {
                *error = "Unexpected end of file (comparison masks)";
            }
            return false;
        }
    }

    std::vector<uint8_t> extra_frame(n_frames, 0);
    if (!ReadExact(file, extra_frame.data(), extra_frame.size())) {
        if (error) {
            *error = "Unexpected end of file (extra frame flags)";
        }
        return false;
    }

    const std::size_t frame_pixels = static_cast<std::size_t>(n_frames) * frame_width * frame_height;
    std::vector<uint16_t> frames_565(frame_pixels);
    if (frame_pixels > 0) {
        if (!ReadExact(file, frames_565.data(), frame_pixels * sizeof(uint16_t))) {
            if (error) {
                *error = "Unexpected end of file (frame data)";
            }
            return false;
        }
    }

    const std::size_t frame_pixels_x = static_cast<std::size_t>(n_frames) * frame_width_x * frame_height_x;
    std::vector<uint16_t> frames_x_565(frame_pixels_x);
    if (frame_pixels_x > 0) {
        if (!ReadExact(file, frames_x_565.data(), frame_pixels_x * sizeof(uint16_t))) {
            if (error) {
                *error = "Unexpected end of file (extra frame data)";
            }
            return false;
        }
    }

    const std::size_t dyna_masks_count = static_cast<std::size_t>(n_frames) * frame_width * frame_height;
    const std::size_t dyna_masks_x_count = static_cast<std::size_t>(n_frames) * frame_width_x * frame_height_x;
    const std::size_t dyna_cols_count = static_cast<std::size_t>(n_frames) * MAX_DYNA_SETS_PER_FRAMEN * no_colors;
    std::vector<uint8_t> dyna_masks(dyna_masks_count);
    std::vector<uint8_t> dyna_masks_x(dyna_masks_x_count);
    std::vector<uint16_t> dyna_cols(dyna_cols_count);
    std::vector<uint16_t> dyna_cols_x(dyna_cols_count);
    if (dyna_masks_count > 0) {
        if (!ReadExact(file, dyna_masks.data(), dyna_masks_count)) {
            if (error) {
                *error = "Unexpected end of file (dynamic masks)";
            }
            return false;
        }
    }
    if (dyna_masks_x_count > 0) {
        if (!ReadExact(file, dyna_masks_x.data(), dyna_masks_x_count)) {
            if (error) {
                *error = "Unexpected end of file (dynamic masks extra)";
            }
            return false;
        }
    }
    if (dyna_cols_count > 0) {
        if (!ReadExact(file, dyna_cols.data(), dyna_cols_count * sizeof(uint16_t))) {
            if (error) {
                *error = "Unexpected end of file (dynamic colors)";
            }
            return false;
        }
        if (dyna_cols_count > 0) {
            if (!ReadExact(file, dyna_cols_x.data(), dyna_cols_count * sizeof(uint16_t))) {
                if (error) {
                    *error = "Unexpected end of file (dynamic colors extra)";
                }
                return false;
            }
        }
    }

    std::vector<uint8_t> extra_sprite(n_sprites, 0);
    if (n_sprites > 0) {
        if (!ReadExact(file, extra_sprite.data(), extra_sprite.size())) {
            if (error) {
                *error = "Unexpected end of file (sprite flags)";
            }
            return false;
        }
    }

    const std::size_t frame_sprites_bytes = static_cast<std::size_t>(n_frames) * MAX_SPRITES_PER_FRAME;
    std::vector<uint8_t> frame_sprites(frame_sprites_bytes, 255);
    if (frame_sprites_bytes > 0) {
        if (!ReadExact(file, frame_sprites.data(), frame_sprites_bytes)) {
            if (error) {
                *error = "Unexpected end of file (frame sprite indices)";
            }
            return false;
        }
    }

    const std::size_t sprite_pixels = static_cast<std::size_t>(n_sprites) * MAX_SPRITE_WIDTH * MAX_SPRITE_HEIGHT;
    std::vector<uint8_t> sprite_original(sprite_pixels);
    if (sprite_pixels > 0) {
        if (!ReadExact(file, sprite_original.data(), sprite_pixels)) {
            if (error) {
                *error = "Unexpected end of file (sprite originals)";
            }
            return false;
        }
    }

    std::vector<uint16_t> sprites_565(sprite_pixels);
    if (sprite_pixels > 0) {
        if (!ReadExact(file, sprites_565.data(), sprite_pixels * sizeof(uint16_t))) {
            if (error) {
                *error = "Unexpected end of file (sprite colored data)";
            }
            return false;
        }
    }

    std::vector<uint8_t> sprite_mask_x(sprite_pixels, 255);
    std::vector<uint16_t> sprites_x_565(sprite_pixels, 0);
    if (sprite_pixels > 0) {
        if (!ReadExact(file, sprite_mask_x.data(), sprite_pixels)) {
            if (error) {
                *error = "Unexpected end of file (sprite extra masks)";
            }
            return false;
        }
        if (!ReadExact(file, sprites_x_565.data(), sprite_pixels * sizeof(uint16_t))) {
            if (error) {
                *error = "Unexpected end of file (sprite extra data)";
            }
            return false;
        }
    }

    if (!SkipExact(file, static_cast<std::size_t>(n_frames))) {
        if (error) {
            *error = "Unexpected end of file (active frame data)";
        }
        return false;
    }

    std::vector<uint16_t> sprite_dyna_cols;
    std::vector<uint8_t> sprite_dyna_masks;
    std::vector<uint8_t> background_flags;
    std::vector<uint16_t> background_frames_565;
    std::vector<uint16_t> background_frames_x_565;
    std::vector<uint16_t> background_ids;
    std::vector<uint8_t> background_masks;
    std::vector<uint8_t> background_masks_x;

    if (length_header >= 9 * sizeof(uint32_t)) {
        const std::size_t rotations_bytes =
            static_cast<std::size_t>(n_frames) * MAX_COLOR_ROTATIONN * MAX_LENGTH_COLOR_ROTATION * sizeof(uint16_t);
        const std::size_t rotations_count = rotations_bytes / sizeof(uint16_t);
        std::vector<uint16_t> rotations(rotations_count, 0);
        std::vector<uint16_t> rotations_x(rotations_count, 0);
        if ((rotations_bytes > 0 && !ReadExact(file, rotations.data(), rotations_bytes)) ||
            (rotations_bytes > 0 && !ReadExact(file, rotations_x.data(), rotations_bytes))) {
            if (error) {
                *error = "Unexpected end of file (rotations)";
            }
            return false;
        }
        out.frame_rotations = std::move(rotations);
        out.frame_rotations_x = std::move(rotations_x);
        if (length_header >= 10 * sizeof(uint32_t)) {
            const std::size_t det_dwords_bytes =
                static_cast<std::size_t>(n_sprites) * MAX_SPRITE_DETECT_AREAS * sizeof(uint32_t);
            const std::size_t det_dword_pos_bytes =
                static_cast<std::size_t>(n_sprites) * MAX_SPRITE_DETECT_AREAS * sizeof(uint16_t);
            const std::size_t det_areas_bytes =
                static_cast<std::size_t>(n_sprites) * MAX_SPRITE_DETECT_AREAS * 4 * sizeof(uint16_t);
            std::vector<uint32_t> det_dwords(det_dwords_bytes / sizeof(uint32_t), 0);
            std::vector<uint16_t> det_dword_pos(det_dword_pos_bytes / sizeof(uint16_t), 0);
            std::vector<uint16_t> det_areas(det_areas_bytes / sizeof(uint16_t), 0xffff);
            if ((det_dwords_bytes > 0 && !ReadExact(file, det_dwords.data(), det_dwords_bytes)) ||
                (det_dword_pos_bytes > 0 && !ReadExact(file, det_dword_pos.data(), det_dword_pos_bytes)) ||
                (det_areas_bytes > 0 && !ReadExact(file, det_areas.data(), det_areas_bytes))) {
                if (error) {
                    *error = "Unexpected end of file (sprite detection)";
                }
                return false;
            }
            out.sprite_det_dwords = std::move(det_dwords);
            out.sprite_det_dword_pos = std::move(det_dword_pos);
            out.sprite_det_areas = std::move(det_areas);
            if (length_header >= 11 * sizeof(uint32_t)) {
                const std::size_t trigger_bytes = static_cast<std::size_t>(n_frames) * sizeof(uint32_t);
                if (!SkipExact(file, trigger_bytes)) {
                    if (error) {
                        *error = "Unexpected end of file (trigger IDs)";
                    }
                    return false;
                }
                if (length_header >= 12 * sizeof(uint32_t)) {
                    const std::size_t frame_sprite_bb_bytes =
                        static_cast<std::size_t>(n_frames) * MAX_SPRITES_PER_FRAME * 4 * sizeof(uint16_t);
                    std::vector<uint16_t> frame_sprite_bb(frame_sprite_bb_bytes / sizeof(uint16_t), 0);
                    if (frame_sprite_bb_bytes > 0 &&
                        !ReadExact(file, frame_sprite_bb.data(), frame_sprite_bb_bytes)) {
                        if (error) {
                            *error = "Unexpected end of file (frame sprite bounding boxes)";
                        }
                        return false;
                    }
                    out.frame_sprite_bboxes = std::move(frame_sprite_bb);
                    if (length_header >= 13 * sizeof(uint32_t)) {
                        const std::size_t background_flags_bytes = static_cast<std::size_t>(n_backgrounds);
                        const std::size_t background_frames =
                            static_cast<std::size_t>(n_backgrounds) * frame_width * frame_height;
                        const std::size_t background_frames_x =
                            static_cast<std::size_t>(n_backgrounds) * frame_width_x * frame_height_x;
                        const std::size_t background_ids_bytes =
                            static_cast<std::size_t>(n_frames) * sizeof(uint16_t);
                        const std::size_t background_masks_bytes =
                            static_cast<std::size_t>(n_frames) * frame_width * frame_height;
                        const std::size_t background_masks_x_bytes =
                            static_cast<std::size_t>(n_frames) * frame_width_x * frame_height_x;

                        background_flags.resize(background_flags_bytes);
                        background_frames_565.resize(background_frames);
                        background_frames_x_565.resize(background_frames_x);
                        background_ids.resize(n_frames, 0xffff);
                        background_masks.resize(background_masks_bytes);
                        background_masks_x.resize(background_masks_x_bytes);
                        if ((background_flags_bytes > 0 &&
                             !ReadExact(file, background_flags.data(), background_flags_bytes)) ||
                            (background_frames > 0 &&
                             !ReadExact(file, background_frames_565.data(), background_frames * sizeof(uint16_t))) ||
                            (background_frames_x > 0 &&
                             !ReadExact(file, background_frames_x_565.data(), background_frames_x * sizeof(uint16_t))) ||
                            (background_ids_bytes > 0 &&
                             !ReadExact(file, background_ids.data(), background_ids_bytes)) ||
                            (background_masks_bytes > 0 &&
                             !ReadExact(file, background_masks.data(), background_masks_bytes)) ||
                            (background_masks_x_bytes > 0 &&
                             !ReadExact(file, background_masks_x.data(), background_masks_x_bytes))) {
                            if (error) {
                                *error = "Unexpected end of file (background data)";
                            }
                            return false;
                        }
                        if (length_header >= 15 * sizeof(uint32_t)) {
                            const std::size_t dyna_shadow_dirs =
                                static_cast<std::size_t>(n_frames) * MAX_DYNA_SETS_PER_FRAMEN;
                            const std::size_t dyna_shadow_cols =
                                static_cast<std::size_t>(n_frames) * MAX_DYNA_SETS_PER_FRAMEN * sizeof(uint16_t);
                            if (!SkipExact(file, dyna_shadow_dirs + dyna_shadow_cols +
                                                   dyna_shadow_dirs + dyna_shadow_cols)) {
                                if (error) {
                                    *error = "Unexpected end of file (dynamic shadows)";
                                }
                                return false;
                            }
                            if (length_header >= 18 * sizeof(uint32_t)) {
                                const std::size_t sprite_dyna_cols_bytes =
                                    static_cast<std::size_t>(n_sprites) * MAX_DYNA_SETS_PER_SPRITE * no_colors * sizeof(uint16_t);
                                const std::size_t sprite_masks = static_cast<std::size_t>(n_sprites) * MAX_SPRITE_WIDTH * MAX_SPRITE_HEIGHT;
                                std::vector<uint16_t> sprite_dyna_cols_x;
                                std::vector<uint8_t> sprite_dyna_masks_x;
                                const std::size_t sprite_dyna_bytes =
                                    sprite_dyna_cols_bytes * 2 + sprite_masks * 2;
                                const bool sprite_dyna_present = BytesRemaining(file) >= sprite_dyna_bytes;
                                if (sprite_dyna_present) {
                                    sprite_dyna_cols.resize(sprite_dyna_cols_bytes / sizeof(uint16_t));
                                    sprite_dyna_cols_x.resize(sprite_dyna_cols_bytes / sizeof(uint16_t));
                                    sprite_dyna_masks.resize(sprite_masks);
                                    sprite_dyna_masks_x.resize(sprite_masks);
                                    if (sprite_dyna_cols_bytes > 0) {
                                        if (!ReadExact(file, sprite_dyna_cols.data(), sprite_dyna_cols_bytes)) {
                                            if (error) {
                                                *error = "Unexpected end of file (sprite dyna colors)";
                                            }
                                            return false;
                                        }
                                        if (!ReadExact(file, sprite_dyna_cols_x.data(), sprite_dyna_cols_bytes)) {
                                            if (error) {
                                                *error = "Unexpected end of file (sprite dyna colors extra)";
                                            }
                                            return false;
                                        }
                                    }
                                    if (sprite_masks > 0) {
                                        if (!ReadExact(file, sprite_dyna_masks.data(), sprite_masks)) {
                                            if (error) {
                                                *error = "Unexpected end of file (sprite dyna masks)";
                                            }
                                            return false;
                                        }
                                        if (!ReadExact(file, sprite_dyna_masks_x.data(), sprite_masks)) {
                                            if (error) {
                                                *error = "Unexpected end of file (sprite dyna masks extra)";
                                            }
                                            return false;
                                        }
                                    }
                                }
                                out.sprite_dynamic_colors_x.clear();
                                out.sprite_dynamic_masks_x.clear();
                                out.sprite_dynamic_colors_x.reserve(n_sprites);
                                out.sprite_dynamic_masks_x.reserve(n_sprites);
                                if (!sprite_dyna_cols_x.empty()) {
                                    const std::size_t per_sprite = MAX_DYNA_SETS_PER_SPRITE * no_colors;
                                    for (uint32_t i = 0; i < n_sprites; ++i) {
                                        std::vector<uint16_t> colors(per_sprite, 0);
                                        const std::size_t offset = static_cast<std::size_t>(i) * per_sprite;
                                        if (offset + per_sprite <= sprite_dyna_cols_x.size()) {
                                            std::memcpy(colors.data(),
                                                        sprite_dyna_cols_x.data() + offset,
                                                        per_sprite * sizeof(uint16_t));
                                        }
                                        out.sprite_dynamic_colors_x.push_back(std::move(colors));
                                    }
                                }
                                if (!sprite_dyna_masks_x.empty()) {
                                    for (uint32_t i = 0; i < n_sprites; ++i) {
                                        cv::Mat mask(static_cast<int>(MAX_SPRITE_HEIGHT),
                                                     static_cast<int>(MAX_SPRITE_WIDTH),
                                                     CV_8UC1,
                                                     cv::Scalar(255));
                                        const std::size_t offset = static_cast<std::size_t>(i) *
                                            MAX_SPRITE_WIDTH * MAX_SPRITE_HEIGHT;
                                        if (offset + MAX_SPRITE_WIDTH * MAX_SPRITE_HEIGHT <= sprite_dyna_masks_x.size()) {
                                            std::memcpy(mask.data,
                                                        sprite_dyna_masks_x.data() + offset,
                                                        MAX_SPRITE_WIDTH * MAX_SPRITE_HEIGHT);
                                        }
                                        out.sprite_dynamic_masks_x.push_back(mask);
                                    }
                                }
                                if (length_header >= 19 * sizeof(uint32_t)) {
                                    if (sprite_dyna_present &&
                                        BytesRemaining(file) >= static_cast<std::size_t>(n_sprites)) {
                                        std::vector<uint8_t> sprite_shape_mode(n_sprites, 0);
                                        if (n_sprites > 0 &&
                                            !ReadExact(file, sprite_shape_mode.data(), sprite_shape_mode.size())) {
                                            if (error) {
                                                *error = "Unexpected end of file (sprite shape mode)";
                                            }
                                            return false;
                                        }
                                        out.sprite_shape_modes = std::move(sprite_shape_mode);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    std::vector<uint8_t> ref_frames;
    if (!rp_path.empty()) {
        if (!LoadLegacyReferenceFrames(rp_path, frame_pixels, ref_frames, error)) {
            return false;
        }
    }

    out.name = TrimName(name, sizeof(name));
    out.frame_width = frame_width;
    out.frame_height = frame_height;
    out.frame_width_x = frame_width_x;
    out.frame_height_x = frame_height_x;
    out.sprite_width = MAX_SPRITE_WIDTH;
    out.sprite_height = MAX_SPRITE_HEIGHT;
    out.no_colors = no_colors;

    out.frame_comp_mask_ids = comp_mask_id;
    out.frame_shape_comp_modes = shape_comp;
    out.comp_masks.resize(MAX_MASKS);
    if (n_comp_masks > 0 && !comp_masks.empty()) {
        const std::size_t mask_pixels = static_cast<std::size_t>(frame_width) * frame_height;
        const uint32_t masks_to_copy = std::min<uint32_t>(n_comp_masks, MAX_MASKS);
        for (uint32_t i = 0; i < masks_to_copy; ++i) {
            cv::Mat mask(static_cast<int>(frame_height), static_cast<int>(frame_width), CV_8UC1, cv::Scalar(0));
            const std::size_t offset = static_cast<std::size_t>(i) * mask_pixels;
            std::memcpy(mask.data, comp_masks.data() + offset, mask_pixels);
            out.comp_masks[i] = mask;
        }
    }

    out.frames.reserve(n_frames);
    out.frames_x.reserve(n_frames);
    out.frame_extra_flags = extra_frame;
    out.background_extra_flags = background_flags;
    out.background_ids = background_ids;
    if (out.background_ids.size() < n_frames) {
        out.background_ids.resize(n_frames, 0xffff);
    }
    out.background_masks.reserve(n_frames);
    out.background_masks_x.reserve(n_frames);
    out.background_frames.reserve(n_backgrounds);
    out.background_frames_x.reserve(n_backgrounds);
    out.frame_refs.reserve(n_frames);
    out.frame_dynamic_colors.reserve(n_frames);
    out.frame_dynamic_mask_maps.resize(n_frames);
    out.frame_dynamic_mask_maps_x.resize(n_frames);
    for (uint32_t index = 0; index < n_frames; ++index) {
        const std::size_t offset = static_cast<std::size_t>(index) * frame_width * frame_height;
        const uint16_t* frame_data = frames_565.data() + offset;
        const uint8_t* mask_data = dyna_masks.empty() ? nullptr : dyna_masks.data() + offset;
        const uint16_t* cols_data = dyna_cols.empty() ? nullptr :
            dyna_cols.data() + static_cast<std::size_t>(index) * MAX_DYNA_SETS_PER_FRAMEN * no_colors;
        const uint8_t* ref_data = ref_frames.empty() ? nullptr : ref_frames.data() + offset;
        out.frames.push_back(BuildFrameImage(frame_width,
                                             frame_height,
                                             frame_data,
                                             mask_data,
                                             cols_data,
                                             ref_data,
                                             no_colors));
        const uint8_t* mask_data_x = nullptr;
        if (!frames_x_565.empty() && frame_width_x > 0 && frame_height_x > 0 && index < extra_frame.size() &&
            extra_frame[index] != 0) {
            const std::size_t offset_x = static_cast<std::size_t>(index) * frame_width_x * frame_height_x;
            const uint16_t* frame_data_x = frames_x_565.data() + offset_x;
            mask_data_x = dyna_masks_x.empty() ? nullptr : dyna_masks_x.data() + offset_x;
            const uint16_t* cols_data_x = dyna_cols_x.empty() ? nullptr :
                dyna_cols_x.data() + static_cast<std::size_t>(index) * MAX_DYNA_SETS_PER_FRAMEN * no_colors;
            const uint8_t* ref_data_x = nullptr;
            cv::Mat scaled_ref_x;
            if (ref_data) {
                scaled_ref_x = ScaleReferenceFrame(ref_data,
                                                   frame_width,
                                                   frame_height,
                                                   frame_width_x,
                                                   frame_height_x);
                if (!scaled_ref_x.empty()) {
                    ref_data_x = scaled_ref_x.data;
                }
            }
            out.frames_x.push_back(BuildFrameImage(frame_width_x,
                                                   frame_height_x,
                                                   frame_data_x,
                                                   mask_data_x,
                                                   cols_data_x,
                                                   ref_data_x,
                                                   no_colors));
        } else {
            out.frames_x.emplace_back();
        }

        if (frame_width > 0 && frame_height > 0) {
            cv::Mat map(static_cast<int>(frame_height), static_cast<int>(frame_width), CV_8UC1, cv::Scalar(255));
            const std::size_t pixels = static_cast<std::size_t>(frame_width) * frame_height;
            if (mask_data) {
                std::memcpy(map.data, mask_data, pixels);
            }
            out.frame_dynamic_mask_maps[index] = map;
        }
        if (frame_width_x > 0 && frame_height_x > 0) {
            cv::Mat map_x(static_cast<int>(frame_height_x), static_cast<int>(frame_width_x), CV_8UC1, cv::Scalar(255));
            const std::size_t pixels_x = static_cast<std::size_t>(frame_width_x) * frame_height_x;
            if (mask_data_x) {
                std::memcpy(map_x.data, mask_data_x, pixels_x);
            }
            out.frame_dynamic_mask_maps_x[index] = map_x;
        }
        if (ref_data) {
            cv::Mat refMat(static_cast<int>(frame_height), static_cast<int>(frame_width), CV_8UC1);
            std::memcpy(refMat.data, ref_data, frame_width * frame_height);
            out.frame_refs.push_back(refMat);
        } else {
            out.frame_refs.emplace_back();
        }
        std::vector<uint16_t> colors(MAX_DYNA_SETS_PER_FRAMEN * no_colors, 0);
        if (cols_data) {
            std::memcpy(colors.data(), cols_data, colors.size() * sizeof(uint16_t));
        }
        out.frame_dynamic_colors.push_back(std::move(colors));
    }

    if (n_backgrounds > 0) {
        const std::size_t bg_pixels = static_cast<std::size_t>(frame_width) * frame_height;
        const std::size_t bg_pixels_x = static_cast<std::size_t>(frame_width_x) * frame_height_x;
        for (uint32_t bg = 0; bg < n_backgrounds; ++bg) {
            const std::size_t offset = static_cast<std::size_t>(bg) * bg_pixels;
            const uint16_t* bg_data = background_frames_565.empty()
                ? nullptr
                : background_frames_565.data() + offset;
            if (bg_data && bg_pixels > 0) {
                out.background_frames.push_back(BuildFrameImage(frame_width,
                                                                frame_height,
                                                                bg_data,
                                                                nullptr,
                                                                nullptr,
                                                                nullptr,
                                                                no_colors));
            } else {
                out.background_frames.emplace_back();
            }

            const std::size_t offset_x = static_cast<std::size_t>(bg) * bg_pixels_x;
            const uint16_t* bg_data_x = background_frames_x_565.empty()
                ? nullptr
                : background_frames_x_565.data() + offset_x;
            if (bg_data_x && bg_pixels_x > 0) {
                out.background_frames_x.push_back(BuildFrameImage(frame_width_x,
                                                                  frame_height_x,
                                                                  bg_data_x,
                                                                  nullptr,
                                                                  nullptr,
                                                                  nullptr,
                                                                  no_colors));
            } else {
                out.background_frames_x.emplace_back();
            }
        }
    }
    if (!background_masks.empty()) {
        const std::size_t bg_pixels = static_cast<std::size_t>(frame_width) * frame_height;
        for (uint32_t i = 0; i < n_frames; ++i) {
            cv::Mat mask(static_cast<int>(frame_height), static_cast<int>(frame_width), CV_8UC1, cv::Scalar(0));
            const std::size_t offset = static_cast<std::size_t>(i) * bg_pixels;
            std::memcpy(mask.data, background_masks.data() + offset, bg_pixels);
            out.background_masks.push_back(mask);
        }
    } else {
        out.background_masks.resize(n_frames);
    }
    if (!background_masks_x.empty()) {
        const std::size_t bg_pixels_x = static_cast<std::size_t>(frame_width_x) * frame_height_x;
        for (uint32_t i = 0; i < n_frames; ++i) {
            cv::Mat mask(static_cast<int>(frame_height_x), static_cast<int>(frame_width_x), CV_8UC1, cv::Scalar(0));
            const std::size_t offset = static_cast<std::size_t>(i) * bg_pixels_x;
            std::memcpy(mask.data, background_masks_x.data() + offset, bg_pixels_x);
            out.background_masks_x.push_back(mask);
        }
    } else {
        out.background_masks_x.resize(n_frames);
    }

    out.sprite_extra_flags = extra_sprite;
    out.frame_sprites = frame_sprites;

    out.sprites.reserve(n_sprites);
    out.sprites_x.reserve(n_sprites);
    out.sprite_colored.reserve(n_sprites);
    out.sprite_colored_x.reserve(n_sprites);
    out.sprite_originals.reserve(n_sprites);
    out.sprite_masks_x.reserve(n_sprites);
    out.sprite_dynamic_colors.reserve(n_sprites);
    out.sprite_dynamic_masks.reserve(n_sprites);
    for (uint32_t index = 0; index < n_sprites; ++index) {
        const bool has_extra_sprite = index < extra_sprite.size() && extra_sprite[index] != 0;
        const std::size_t offset = static_cast<std::size_t>(index) * MAX_SPRITE_WIDTH * MAX_SPRITE_HEIGHT;
        const uint16_t* sprite_colored = sprites_565.data() + offset;
        const uint8_t* sprite_orig = sprite_original.data() + offset;
        const uint8_t* sprite_mask = sprite_dyna_masks.empty() ? nullptr : sprite_dyna_masks.data() + offset;
        const uint16_t* sprite_cols = sprite_dyna_cols.empty() ? nullptr :
            sprite_dyna_cols.data() + static_cast<std::size_t>(index) * MAX_DYNA_SETS_PER_SPRITE * no_colors;
        cv::Mat sprite_original_mat(static_cast<int>(MAX_SPRITE_HEIGHT),
                                    static_cast<int>(MAX_SPRITE_WIDTH),
                                    CV_8UC1,
                                    cv::Scalar(255));
        if (!sprite_original.empty()) {
            std::memcpy(sprite_original_mat.data, sprite_orig,
                        MAX_SPRITE_WIDTH * MAX_SPRITE_HEIGHT);
        }
        out.sprite_originals.push_back(sprite_original_mat.clone());

        if (has_extra_sprite) {
            cv::Mat sprite_mask_x_mat(static_cast<int>(MAX_SPRITE_HEIGHT),
                                      static_cast<int>(MAX_SPRITE_WIDTH),
                                      CV_8UC1,
                                      cv::Scalar(255));
            if (!sprite_mask_x.empty()) {
                std::memcpy(sprite_mask_x_mat.data, sprite_mask_x.data() + offset,
                            MAX_SPRITE_WIDTH * MAX_SPRITE_HEIGHT);
            }
            out.sprite_masks_x.push_back(sprite_mask_x_mat.clone());
        } else {
            out.sprite_masks_x.emplace_back();
        }

        out.sprite_colored.push_back(BuildSpriteImage(MAX_SPRITE_WIDTH,
                                                      MAX_SPRITE_HEIGHT,
                                                      sprite_colored,
                                                      sprite_orig,
                                                      nullptr,
                                                      nullptr,
                                                      no_colors));
        out.sprites.push_back(BuildSpriteImage(MAX_SPRITE_WIDTH,
                                               MAX_SPRITE_HEIGHT,
                                               sprite_colored,
                                               sprite_orig,
                                               sprite_mask,
                                               sprite_cols,
                                               no_colors));

        if (has_extra_sprite) {
            const uint8_t* sprite_mask_x_ptr = sprite_mask_x.empty()
                ? nullptr
                : sprite_mask_x.data() + offset;
            const uint16_t* sprite_colored_x = sprites_x_565.empty() ? nullptr : sprites_x_565.data() + offset;
            out.sprite_colored_x.push_back(BuildSpriteImage(MAX_SPRITE_WIDTH,
                                                            MAX_SPRITE_HEIGHT,
                                                            sprite_colored_x,
                                                            sprite_mask_x_ptr,
                                                            nullptr,
                                                            nullptr,
                                                            no_colors));
            const cv::Mat* sprite_dyn_mask_x = (!out.sprite_dynamic_masks_x.empty() &&
                index < out.sprite_dynamic_masks_x.size())
                ? &out.sprite_dynamic_masks_x[index]
                : nullptr;
            const std::vector<uint16_t>* sprite_dyn_cols_x = (!out.sprite_dynamic_colors_x.empty() &&
                index < out.sprite_dynamic_colors_x.size())
                ? &out.sprite_dynamic_colors_x[index]
                : nullptr;
            out.sprites_x.push_back(BuildSpriteImage(MAX_SPRITE_WIDTH,
                                                     MAX_SPRITE_HEIGHT,
                                                     sprite_colored_x,
                                                     sprite_mask_x_ptr,
                                                     sprite_dyn_mask_x ? sprite_dyn_mask_x->data : nullptr,
                                                     sprite_dyn_cols_x && !sprite_dyn_cols_x->empty()
                                                        ? sprite_dyn_cols_x->data()
                                                        : nullptr,
                                                     no_colors));
        } else {
            out.sprite_colored_x.emplace_back();
            out.sprites_x.emplace_back();
            if (index < out.sprite_dynamic_masks_x.size()) {
                out.sprite_dynamic_masks_x[index] = cv::Mat();
            }
            if (index < out.sprite_dynamic_colors_x.size()) {
                out.sprite_dynamic_colors_x[index].clear();
            }
        }

        std::vector<uint16_t> colors(MAX_DYNA_SETS_PER_SPRITE * no_colors, 0);
        if (sprite_cols) {
            std::memcpy(colors.data(), sprite_cols, colors.size() * sizeof(uint16_t));
        }
        out.sprite_dynamic_colors.push_back(std::move(colors));
        cv::Mat dyna_mask(static_cast<int>(MAX_SPRITE_HEIGHT),
                          static_cast<int>(MAX_SPRITE_WIDTH),
                          CV_8UC1,
                          cv::Scalar(255));
        if (sprite_mask) {
            std::memcpy(dyna_mask.data, sprite_mask,
                        MAX_SPRITE_WIDTH * MAX_SPRITE_HEIGHT);
        }
        out.sprite_dynamic_masks.push_back(dyna_mask);
    }
    if (out.sprite_shape_modes.empty() && n_sprites > 0) {
        out.sprite_shape_modes.resize(n_sprites, 0);
    }

    if (!rp_path.empty()) {
        std::vector<uint32_t> section_firsts;
        std::vector<std::string> section_names;
        std::vector<std::string> sprite_names;
        std::vector<uint32_t> frame_durations;
        std::string meta_error;
        if (LoadLegacyMetadataFromRP(rp_path,
                                     frame_width,
                                     frame_height,
                                     n_frames,
                                     n_sprites,
                                     out,
                                     frame_durations,
                                     sprite_names,
                                     section_firsts,
                                     section_names,
                                     &meta_error)) {
            out.frame_durations = frame_durations;
            out.sprite_labels = sprite_names;
            out.section_firsts = section_firsts;
            out.section_names = section_names;

            if (out.sprite_labels.size() < n_sprites) {
                out.sprite_labels.resize(n_sprites);
            }
            for (uint32_t i = 0; i < n_sprites; ++i) {
                if (out.sprite_labels[i].empty()) {
                    out.sprite_labels[i] = "Sprite " + std::to_string(i);
                } else {
                    out.sprite_labels[i] = "Sprite " + std::to_string(i) + " - " + out.sprite_labels[i];
                }
            }
        }
    }

    if (out.sprite_labels.size() < n_sprites) {
        out.sprite_labels.reserve(n_sprites);
        for (uint32_t i = 0; i < n_sprites; ++i) {
            out.sprite_labels.push_back("Sprite " + std::to_string(i));
        }
    }

    return true;
}
