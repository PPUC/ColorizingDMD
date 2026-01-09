#include "legacy_project.h"

#include <algorithm>
#include <fstream>

#include "serum_constants.h"

namespace {
bool ReadExact(std::ifstream& file, void* dst, std::size_t size)
{
    return static_cast<bool>(file.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(size)));
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
    if (!SkipExact(file, MAX_COL_SETS * kBoolSize)) {
        if (error) {
            *error = "Unexpected end of .cRP file (col sets)";
        }
        return false;
    }

    const std::size_t col_sets_bytes = MAX_COL_SETS * 16 * sizeof(uint16_t);
    if (!SkipExact(file, col_sets_bytes)) {
        if (error) {
            *error = "Unexpected end of .cRP file (color sets)";
        }
        return false;
    }

    if (!SkipExact(file, sizeof(uint8_t) + sizeof(uint8_t))) {
        if (error) {
            *error = "Unexpected end of .cRP file (color set indices)";
        }
        return false;
    }

    if (!SkipExact(file, MAX_COL_SETS * 64)) {
        if (error) {
            *error = "Unexpected end of .cRP file (color set names)";
        }
        return false;
    }

    if (!SkipExact(file, sizeof(uint32_t) + sizeof(uint8_t) + sizeof(int32_t) + kBoolSize)) {
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

    if (!SkipExact(file, MAX_SPRITES * sizeof(uint32_t))) {
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

    const std::size_t hash_bytes = static_cast<std::size_t>(n_frames) * sizeof(uint32_t);
    const std::size_t shape_bytes = static_cast<std::size_t>(n_frames);
    const std::size_t comp_id_bytes = static_cast<std::size_t>(n_frames);
    if (!SkipExact(file, hash_bytes + shape_bytes + comp_id_bytes)) {
        if (error) {
            *error = "Unexpected end of file (header data)";
        }
        return false;
    }

    const std::size_t comp_masks_bytes =
        static_cast<std::size_t>(n_comp_masks) * frame_width * frame_height;
    if (!SkipExact(file, comp_masks_bytes)) {
        if (error) {
            *error = "Unexpected end of file (comparison masks)";
        }
        return false;
    }

    if (!SkipExact(file, static_cast<std::size_t>(n_frames))) {
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
    if (!SkipExact(file, frame_pixels_x * sizeof(uint16_t))) {
        if (error) {
            *error = "Unexpected end of file (extra frame data)";
        }
        return false;
    }

    const std::size_t dyna_masks_count = static_cast<std::size_t>(n_frames) * frame_width * frame_height;
    const std::size_t dyna_masks_x_count = static_cast<std::size_t>(n_frames) * frame_width_x * frame_height_x;
    const std::size_t dyna_cols_count = static_cast<std::size_t>(n_frames) * MAX_DYNA_SETS_PER_FRAMEN * no_colors;
    std::vector<uint8_t> dyna_masks(dyna_masks_count);
    std::vector<uint16_t> dyna_cols(dyna_cols_count);
    if (dyna_masks_count > 0) {
        if (!ReadExact(file, dyna_masks.data(), dyna_masks_count)) {
            if (error) {
                *error = "Unexpected end of file (dynamic masks)";
            }
            return false;
        }
    }
    if (!SkipExact(file, dyna_masks_x_count)) {
        if (error) {
            *error = "Unexpected end of file (dynamic masks extra)";
        }
        return false;
    }
    if (dyna_cols_count > 0) {
        if (!ReadExact(file, dyna_cols.data(), dyna_cols_count * sizeof(uint16_t))) {
            if (error) {
                *error = "Unexpected end of file (dynamic colors)";
            }
            return false;
        }
        if (!SkipExact(file, dyna_cols_count * sizeof(uint16_t))) {
            if (error) {
                *error = "Unexpected end of file (dynamic colors extra)";
            }
            return false;
        }
    }

    if (!SkipExact(file, static_cast<std::size_t>(n_sprites))) {
        if (error) {
            *error = "Unexpected end of file (sprite flags)";
        }
        return false;
    }

    const std::size_t frame_sprites_bytes = static_cast<std::size_t>(n_frames) * MAX_SPRITES_PER_FRAME;
    if (!SkipExact(file, frame_sprites_bytes)) {
        if (error) {
            *error = "Unexpected end of file (frame sprite indices)";
        }
        return false;
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

    const std::size_t sprite_masks_x_bytes = sprite_pixels;
    const std::size_t sprite_colored_x_bytes = sprite_pixels * sizeof(uint16_t);
    if (!SkipExact(file, sprite_masks_x_bytes + sprite_colored_x_bytes)) {
        if (error) {
            *error = "Unexpected end of file (sprite extra data)";
        }
        return false;
    }

    if (!SkipExact(file, static_cast<std::size_t>(n_frames))) {
        if (error) {
            *error = "Unexpected end of file (active frame data)";
        }
        return false;
    }

    std::vector<uint16_t> sprite_dyna_cols;
    std::vector<uint8_t> sprite_dyna_masks;

    if (length_header >= 9 * sizeof(uint32_t)) {
        const std::size_t rotations_bytes =
            static_cast<std::size_t>(n_frames) * MAX_COLOR_ROTATIONN * MAX_LENGTH_COLOR_ROTATION * sizeof(uint16_t);
        if (!SkipExact(file, rotations_bytes + rotations_bytes)) {
            if (error) {
                *error = "Unexpected end of file (rotations)";
            }
            return false;
        }
        if (length_header >= 10 * sizeof(uint32_t)) {
            const std::size_t det_dwords_bytes =
                static_cast<std::size_t>(n_sprites) * MAX_SPRITE_DETECT_AREAS * sizeof(uint32_t);
            const std::size_t det_dword_pos_bytes =
                static_cast<std::size_t>(n_sprites) * MAX_SPRITE_DETECT_AREAS * sizeof(uint16_t);
            const std::size_t det_areas_bytes =
                static_cast<std::size_t>(n_sprites) * MAX_SPRITE_DETECT_AREAS * 4 * sizeof(uint16_t);
            if (!SkipExact(file, det_dwords_bytes + det_dword_pos_bytes + det_areas_bytes)) {
                if (error) {
                    *error = "Unexpected end of file (sprite detection)";
                }
                return false;
            }
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
                    if (!SkipExact(file, frame_sprite_bb_bytes)) {
                        if (error) {
                            *error = "Unexpected end of file (frame sprite bounding boxes)";
                        }
                        return false;
                    }
                    if (length_header >= 13 * sizeof(uint32_t)) {
                        const std::size_t background_flags = static_cast<std::size_t>(n_backgrounds);
                        const std::size_t background_frames =
                            static_cast<std::size_t>(n_backgrounds) * frame_width * frame_height * sizeof(uint16_t);
                        const std::size_t background_frames_x =
                            static_cast<std::size_t>(n_backgrounds) * frame_width_x * frame_height_x * sizeof(uint16_t);
                        const std::size_t background_ids =
                            static_cast<std::size_t>(n_frames) * sizeof(uint16_t);
                        const std::size_t background_masks =
                            static_cast<std::size_t>(n_frames) * frame_width * frame_height;
                        const std::size_t background_masks_x =
                            static_cast<std::size_t>(n_frames) * frame_width_x * frame_height_x;
                        if (!SkipExact(file,
                                       background_flags + background_frames + background_frames_x +
                                       background_ids + background_masks + background_masks_x)) {
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
                                sprite_dyna_cols.resize(sprite_dyna_cols_bytes / sizeof(uint16_t));
                                sprite_dyna_masks.resize(sprite_masks);
                                if (sprite_dyna_cols_bytes > 0) {
                                    if (!ReadExact(file, sprite_dyna_cols.data(), sprite_dyna_cols_bytes)) {
                                        if (error) {
                                            *error = "Unexpected end of file (sprite dyna colors)";
                                        }
                                        return false;
                                    }
                                }
                                if (!SkipExact(file, sprite_dyna_cols_bytes)) {
                                    if (error) {
                                        *error = "Unexpected end of file (sprite dyna colors extra)";
                                    }
                                    return false;
                                }
                                if (sprite_masks > 0) {
                                    if (!ReadExact(file, sprite_dyna_masks.data(), sprite_masks)) {
                                        if (error) {
                                            *error = "Unexpected end of file (sprite dyna masks)";
                                        }
                                        return false;
                                    }
                                }
                                if (!SkipExact(file, sprite_masks)) {
                                    if (error) {
                                        *error = "Unexpected end of file (sprite dyna masks extra)";
                                    }
                                    return false;
                                }
                                if (length_header >= 19 * sizeof(uint32_t)) {
                                    if (!SkipExact(file, static_cast<std::size_t>(n_sprites))) {
                                        if (error) {
                                            *error = "Unexpected end of file (sprite shape mode)";
                                        }
                                        return false;
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

    out.frames.reserve(n_frames);
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
    }

    out.sprites.reserve(n_sprites);
    for (uint32_t index = 0; index < n_sprites; ++index) {
        const std::size_t offset = static_cast<std::size_t>(index) * MAX_SPRITE_WIDTH * MAX_SPRITE_HEIGHT;
        const uint16_t* sprite_colored = sprites_565.data() + offset;
        const uint8_t* sprite_orig = sprite_original.data() + offset;
        const uint8_t* sprite_mask = sprite_dyna_masks.empty() ? nullptr : sprite_dyna_masks.data() + offset;
        const uint16_t* sprite_cols = sprite_dyna_cols.empty() ? nullptr :
            sprite_dyna_cols.data() + static_cast<std::size_t>(index) * MAX_DYNA_SETS_PER_SPRITE * no_colors;
        out.sprites.push_back(BuildSpriteImage(MAX_SPRITE_WIDTH,
                                               MAX_SPRITE_HEIGHT,
                                               sprite_colored,
                                               sprite_orig,
                                               sprite_mask,
                                               sprite_cols,
                                               no_colors));
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
