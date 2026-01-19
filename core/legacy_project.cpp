#include "legacy_project.h"

#include <algorithm>
#include <cstring>
#include <fstream>

#include "SerumData.h"
#include "serum.h"
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

bool FileExists(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    return static_cast<bool>(file);
}

bool ReadLegacyHeader(const std::string& path,
                      uint32_t& width,
                      uint32_t& height,
                      uint32_t& n_frames)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }

    char name[64]{};
    uint32_t length_header = 0;
    uint32_t frame_width = 0;
    uint32_t frame_height = 0;
    uint32_t frame_width_x = 0;
    uint32_t frame_height_x = 0;
    uint32_t frames = 0;

    if (!ReadExact(file, name, sizeof(name)) ||
        !ReadExact(file, &length_header, sizeof(length_header)) ||
        !ReadExact(file, &frame_width, sizeof(frame_width)) ||
        !ReadExact(file, &frame_height, sizeof(frame_height))) {
        return false;
    }

    if (length_header >= 14 * sizeof(uint32_t)) {
        if (!ReadExact(file, &frame_width_x, sizeof(frame_width_x)) ||
            !ReadExact(file, &frame_height_x, sizeof(frame_height_x)) ||
            !ReadExact(file, &frames, sizeof(frames))) {
            return false;
        }
    } else {
        return false;
    }

    if (frame_width == 0 || frame_height == 0 || frames == 0) {
        return false;
    }

    width = frame_width;
    height = frame_height;
    n_frames = frames;
    return true;
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

template <typename T>
void CopySparseVector(const SparseVector<T>& source,
                      std::size_t element_size,
                      uint32_t count,
                      std::vector<T>& out)
{
    out.clear();
    if (count == 0 || element_size == 0) {
        return;
    }
    out.resize(static_cast<std::size_t>(count) * element_size);
    auto& mutable_source = const_cast<SparseVector<T>&>(source);
    mutable_source.reserve(element_size);
    for (uint32_t index = 0; index < count; ++index) {
        const T* data = mutable_source[index];
        std::memcpy(out.data() + static_cast<std::size_t>(index) * element_size,
                    data,
                    element_size * sizeof(T));
    }
}

template <typename T>
std::size_t ElementSizeOr(const SparseVector<T>& source, std::size_t fallback)
{
    return source.elementCount() > 0 ? source.elementCount() : fallback;
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
    out.active_col_sets = active_col_set;

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
    out.draw_col_mode = draw_col_mode;
    out.draw_mode = draw_mode;
    out.mask_sel_mode = mask_sel_mode;
    out.fill_mode = fill_mode;

    std::vector<char> mask_names(MAX_MASKS * SIZE_MASK_NAME, 0);
    if (!ReadExact(file, mask_names.data(), mask_names.size())) {
        if (error) {
            *error = "Unexpected end of .cRP file (mask names)";
        }
        return false;
    }
    out.mask_names = mask_names;

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

    std::vector<uint16_t> edit_colors(16, 0);
    if (!ReadExact(file, edit_colors.data(), edit_colors.size() * sizeof(uint16_t))) {
        if (error) {
            *error = "Unexpected end of .cRP file (edit colors)";
        }
        return false;
    }
    out.edit_colors = edit_colors;

    uint32_t n_image_pos_saves = 0;
    if (!ReadExact(file, &n_image_pos_saves, sizeof(n_image_pos_saves))) {
        if (error) {
            *error = "Unexpected end of .cRP file (image positions)";
        }
        return false;
    }
    out.n_image_pos_saves = n_image_pos_saves;
    std::vector<char> image_pos_names(N_IMAGE_POS_TO_SAVE * 64, 0);
    if (!ReadExact(file, image_pos_names.data(), image_pos_names.size())) {
        if (error) {
            *error = "Unexpected end of .cRP file (image pos names)";
        }
        return false;
    }
    out.image_pos_names = image_pos_names;
    std::vector<int32_t> image_pos_data(N_IMAGE_POS_TO_SAVE * 16, 0);
    if (!ReadExact(file, image_pos_data.data(), image_pos_data.size() * sizeof(int32_t))) {
        if (error) {
            *error = "Unexpected end of .cRP file (image pos data)";
        }
        return false;
    }
    out.image_pos_data = image_pos_data;

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
    out.is_imported = is_imported;
    out.time_elapsed = time_elapsed;
    out.is_pup_pack = is_pup_pack;
    std::vector<char> pup_pack(sizeof(wchar_t) * 256, 0);
    if (!ReadExact(file, pup_pack.data(), pup_pack.size())) {
        if (error) {
            *error = "Unexpected end of .cRP file (pup pack)";
        }
        return false;
    }
    out.pup_pack = pup_pack;

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

bool LoadLegacyProjectFromConcentrate(const std::string& cromc_path,
                                      const std::string& rp_path,
                                      LegacyProject& out,
                                      std::string* error,
                                      bool skip_frame_images)
{
    out = LegacyProject{};

    SerumData data;
    const uint8_t flags = FLAG_REQUEST_32P_FRAMES | FLAG_REQUEST_64P_FRAMES;
    if (!data.LoadFromFile(cromc_path.c_str(), flags)) {
        if (error) {
            *error = "Failed to load .cROMc file";
        }
        return false;
    }
    if (data.SerumVersion != SERUM_V2) {
        if (error) {
            *error = "Only Serum V2 cROMc files are supported";
        }
        return false;
    }

    const uint32_t frame_width = data.fwidth;
    const uint32_t frame_height = data.fheight;
    const uint32_t frame_width_x = data.fwidth_extra;
    const uint32_t frame_height_x = data.fheight_extra;
    const uint32_t n_frames = data.nframes;
    const uint32_t no_colors = data.nocolors;
    const uint32_t n_comp_masks = data.ncompmasks;
    const uint32_t n_sprites = data.nsprites;
    const uint16_t n_backgrounds = data.nbackgrounds;

    if ((frame_width == 0) || (frame_height == 0) || (n_frames == 0) ||
        (no_colors == 0)) {
        if (error) {
            *error = "Invalid cROMc header data";
        }
        return false;
    }

    std::vector<uint32_t> hash_codes;
    CopySparseVector(data.hashcodes,
                     ElementSizeOr(data.hashcodes, 1u),
                     n_frames,
                     hash_codes);

    std::vector<uint8_t> shape_comp;
    CopySparseVector(data.shapecompmode,
                     ElementSizeOr(data.shapecompmode, 1u),
                     n_frames,
                     shape_comp);

    std::vector<uint8_t> comp_mask_id;
    CopySparseVector(data.compmaskID,
                     ElementSizeOr(data.compmaskID, 1u),
                     n_frames,
                     comp_mask_id);

    std::vector<uint8_t> comp_masks;
    const std::size_t comp_mask_pixels = (data.is256x64
        ? static_cast<std::size_t>(256 * 64)
        : static_cast<std::size_t>(frame_width) * frame_height);
    CopySparseVector(data.compmasks,
                     ElementSizeOr(data.compmasks, comp_mask_pixels),
                     n_comp_masks,
                     comp_masks);

    std::vector<uint8_t> extra_frame;
    CopySparseVector(data.isextraframe,
                     ElementSizeOr(data.isextraframe, 1u),
                     n_frames,
                     extra_frame);

    std::vector<uint16_t> frames_565;
    std::vector<uint16_t> frames_x_565;
    std::vector<uint8_t> dyna_masks;
    std::vector<uint8_t> dyna_masks_x;
    std::vector<uint16_t> dyna_cols;
    std::vector<uint16_t> dyna_cols_x;
    const std::size_t frame_dyna_cols = MAX_DYNA_SETS_PER_FRAMEN * no_colors;
    if (!skip_frame_images) {
        CopySparseVector(data.cframes_v2,
                         ElementSizeOr(data.cframes_v2,
                                       static_cast<std::size_t>(frame_width) * frame_height),
                         n_frames,
                         frames_565);

        CopySparseVector(data.cframes_v2_extra,
                         ElementSizeOr(data.cframes_v2_extra,
                                       static_cast<std::size_t>(frame_width_x) * frame_height_x),
                         n_frames,
                         frames_x_565);

        CopySparseVector(data.dynamasks,
                         ElementSizeOr(data.dynamasks,
                                       static_cast<std::size_t>(frame_width) * frame_height),
                         n_frames,
                         dyna_masks);

        CopySparseVector(data.dynamasks_extra,
                         ElementSizeOr(data.dynamasks_extra,
                                       static_cast<std::size_t>(frame_width_x) * frame_height_x),
                         n_frames,
                         dyna_masks_x);

        CopySparseVector(data.dyna4cols_v2,
                         ElementSizeOr(data.dyna4cols_v2, frame_dyna_cols),
                         n_frames,
                         dyna_cols);

        CopySparseVector(data.dyna4cols_v2_extra,
                         ElementSizeOr(data.dyna4cols_v2_extra, frame_dyna_cols),
                         n_frames,
                         dyna_cols_x);
    }

    std::vector<uint8_t> frame_sprites;
    CopySparseVector(data.framesprites,
                     ElementSizeOr(data.framesprites, MAX_SPRITES_PER_FRAME),
                     n_frames,
                     frame_sprites);

    std::vector<uint8_t> sprite_original;
    const std::size_t sprite_pixels =
        static_cast<std::size_t>(MAX_SPRITE_WIDTH) * MAX_SPRITE_HEIGHT;
    CopySparseVector(data.spriteoriginal,
                     ElementSizeOr(data.spriteoriginal, sprite_pixels),
                     n_sprites,
                     sprite_original);

    std::vector<uint16_t> sprites_565;
    CopySparseVector(data.spritecolored,
                     ElementSizeOr(data.spritecolored, sprite_pixels),
                     n_sprites,
                     sprites_565);

    std::vector<uint8_t> sprite_mask_x;
    CopySparseVector(data.spritemask_extra,
                     ElementSizeOr(data.spritemask_extra, sprite_pixels),
                     n_sprites,
                     sprite_mask_x);

    std::vector<uint16_t> sprites_x_565;
    CopySparseVector(data.spritecolored_extra,
                     ElementSizeOr(data.spritecolored_extra, sprite_pixels),
                     n_sprites,
                     sprites_x_565);

    std::vector<uint8_t> extra_sprite;
    CopySparseVector(data.isextrasprite,
                     ElementSizeOr(data.isextrasprite, 1u),
                     n_sprites,
                     extra_sprite);

    std::vector<uint8_t> sprite_shape_mode;
    CopySparseVector(data.sprshapemode,
                     ElementSizeOr(data.sprshapemode, 1u),
                     n_sprites,
                     sprite_shape_mode);

    std::vector<uint16_t> det_areas;
    CopySparseVector(data.spritedetareas,
                     ElementSizeOr(data.spritedetareas, MAX_SPRITE_DETECT_AREAS * 4),
                     n_sprites,
                     det_areas);

    std::vector<uint32_t> det_dwords;
    CopySparseVector(data.spritedetdwords,
                     ElementSizeOr(data.spritedetdwords, MAX_SPRITE_DETECT_AREAS),
                     n_sprites,
                     det_dwords);

    std::vector<uint16_t> det_dword_pos;
    CopySparseVector(data.spritedetdwordpos,
                     ElementSizeOr(data.spritedetdwordpos, MAX_SPRITE_DETECT_AREAS),
                     n_sprites,
                     det_dword_pos);

    std::vector<uint32_t> trigger_ids;
    CopySparseVector(data.triggerIDs,
                     ElementSizeOr(data.triggerIDs, 1u),
                     n_frames,
                     trigger_ids);

    std::vector<uint16_t> frame_sprite_bboxes;
    CopySparseVector(data.framespriteBB,
                     ElementSizeOr(data.framespriteBB, MAX_SPRITES_PER_FRAME * 4),
                     n_frames,
                     frame_sprite_bboxes);

    std::vector<uint16_t> color_rotations;
    CopySparseVector(data.colorrotations_v2,
                     ElementSizeOr(data.colorrotations_v2,
                                   MAX_LENGTH_COLOR_ROTATION * MAX_COLOR_ROTATIONN),
                     n_frames,
                     color_rotations);

    std::vector<uint16_t> color_rotations_x;
    CopySparseVector(data.colorrotations_v2_extra,
                     ElementSizeOr(data.colorrotations_v2_extra,
                                   MAX_LENGTH_COLOR_ROTATION * MAX_COLOR_ROTATIONN),
                     n_frames,
                     color_rotations_x);

    std::vector<uint8_t> background_flags;
    CopySparseVector(data.isextrabackground,
                     ElementSizeOr(data.isextrabackground, 1u),
                     n_backgrounds,
                     background_flags);

    std::vector<uint16_t> background_frames_565;
    const std::size_t bg_pixels =
        static_cast<std::size_t>(frame_width) * frame_height;
    CopySparseVector(data.backgroundframes_v2,
                     ElementSizeOr(data.backgroundframes_v2, bg_pixels),
                     n_backgrounds,
                     background_frames_565);

    std::vector<uint16_t> background_frames_x_565;
    const std::size_t bg_pixels_x =
        static_cast<std::size_t>(frame_width_x) * frame_height_x;
    CopySparseVector(data.backgroundframes_v2_extra,
                     ElementSizeOr(data.backgroundframes_v2_extra, bg_pixels_x),
                     n_backgrounds,
                     background_frames_x_565);

    std::vector<uint16_t> background_ids;
    CopySparseVector(data.backgroundIDs,
                     ElementSizeOr(data.backgroundIDs, 1u),
                     n_frames,
                     background_ids);

    std::vector<uint8_t> background_masks;
    CopySparseVector(data.backgroundmask,
                     ElementSizeOr(data.backgroundmask, bg_pixels),
                     n_frames,
                     background_masks);

    std::vector<uint8_t> background_masks_x;
    CopySparseVector(data.backgroundmask_extra,
                     ElementSizeOr(data.backgroundmask_extra, bg_pixels_x),
                     n_frames,
                     background_masks_x);

    std::vector<uint8_t> dynashadow_dir;
    CopySparseVector(data.dynashadowsdir,
                     ElementSizeOr(data.dynashadowsdir, MAX_DYNA_SETS_PER_FRAMEN),
                     n_frames,
                     dynashadow_dir);

    std::vector<uint16_t> dynashadow_col;
    CopySparseVector(data.dynashadowscol,
                     ElementSizeOr(data.dynashadowscol, MAX_DYNA_SETS_PER_FRAMEN),
                     n_frames,
                     dynashadow_col);

    std::vector<uint8_t> dynashadow_dir_x;
    CopySparseVector(data.dynashadowsdir_extra,
                     ElementSizeOr(data.dynashadowsdir_extra, MAX_DYNA_SETS_PER_FRAMEN),
                     n_frames,
                     dynashadow_dir_x);

    std::vector<uint16_t> dynashadow_col_x;
    CopySparseVector(data.dynashadowscol_extra,
                     ElementSizeOr(data.dynashadowscol_extra, MAX_DYNA_SETS_PER_FRAMEN),
                     n_frames,
                     dynashadow_col_x);

    const std::size_t sprite_dyna_cols_count =
        ElementSizeOr(data.dynasprite4cols,
                      static_cast<std::size_t>(MAX_DYNA_SETS_PER_SPRITE) * no_colors);
    const std::size_t sprite_dyna_cols_per_sprite =
        no_colors == 0 ? 0 : (sprite_dyna_cols_count / no_colors) * no_colors;
    std::vector<uint16_t> sprite_dyna_cols;
    CopySparseVector(data.dynasprite4cols, sprite_dyna_cols_count, n_sprites, sprite_dyna_cols);

    std::vector<uint16_t> sprite_dyna_cols_x;
    CopySparseVector(data.dynasprite4cols_extra,
                     ElementSizeOr(data.dynasprite4cols_extra, sprite_dyna_cols_count),
                     n_sprites,
                     sprite_dyna_cols_x);

    std::vector<uint8_t> sprite_dyna_masks;
    CopySparseVector(data.dynaspritemasks,
                     ElementSizeOr(data.dynaspritemasks, sprite_pixels),
                     n_sprites,
                     sprite_dyna_masks);

    std::vector<uint8_t> sprite_dyna_masks_x;
    CopySparseVector(data.dynaspritemasks_extra,
                     ElementSizeOr(data.dynaspritemasks_extra, sprite_pixels),
                     n_sprites,
                     sprite_dyna_masks_x);

    std::vector<uint32_t> sprite_col_from_frame(MAX_SPRITES, 0);
    std::vector<uint16_t> sprite_rects(4 * MAX_SPRITES, 0);
    std::vector<uint32_t> sprite_rect_mirror(2 * MAX_SPRITES, 0);

    std::vector<uint8_t> ref_frames;
    if (!rp_path.empty()) {
        const std::size_t frame_pixels =
            static_cast<std::size_t>(n_frames) * frame_width * frame_height;
        if (!LoadLegacyReferenceFrames(rp_path, frame_pixels, ref_frames, error)) {
            return false;
        }
    }

    out.name = data.rname[0] ? TrimName(data.rname, sizeof(data.rname)) : "";
    out.frame_width = frame_width;
    out.frame_height = frame_height;
    out.frame_width_x = frame_width_x;
    out.frame_height_x = frame_height_x;
    out.sprite_width = MAX_SPRITE_WIDTH;
    out.sprite_height = MAX_SPRITE_HEIGHT;
    out.no_colors = no_colors;

    out.hash_codes = hash_codes;
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
        const uint8_t* ref_data = ref_frames.empty() ? nullptr : ref_frames.data() + offset;
        if (!skip_frame_images) {
            const uint16_t* frame_data = frames_565.data() + offset;
            const uint8_t* mask_data = dyna_masks.empty() ? nullptr : dyna_masks.data() + offset;
            const uint16_t* cols_data = dyna_cols.empty() ? nullptr :
                dyna_cols.data() + static_cast<std::size_t>(index) * MAX_DYNA_SETS_PER_FRAMEN * no_colors;
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
                cv::Mat map_x(static_cast<int>(frame_height_x),
                              static_cast<int>(frame_width_x),
                              CV_8UC1,
                              cv::Scalar(255));
                const std::size_t pixels_x = static_cast<std::size_t>(frame_width_x) * frame_height_x;
                if (mask_data_x) {
                    std::memcpy(map_x.data, mask_data_x, pixels_x);
                }
                out.frame_dynamic_mask_maps_x[index] = map_x;
            }
            std::vector<uint16_t> colors(MAX_DYNA_SETS_PER_FRAMEN * no_colors, 0);
            if (cols_data) {
                std::memcpy(colors.data(), cols_data, colors.size() * sizeof(uint16_t));
            }
            out.frame_dynamic_colors.push_back(std::move(colors));
        } else {
            out.frames.emplace_back();
            out.frames_x.emplace_back();
            out.frame_dynamic_colors.emplace_back();
        }
        if (ref_data) {
            cv::Mat refMat(static_cast<int>(frame_height), static_cast<int>(frame_width), CV_8UC1);
            std::memcpy(refMat.data, ref_data, frame_width * frame_height);
            out.frame_refs.push_back(refMat);
        } else {
            out.frame_refs.emplace_back();
        }
    }

    if (n_backgrounds > 0) {
        const std::size_t pixels = static_cast<std::size_t>(frame_width) * frame_height;
        const std::size_t pixels_x = static_cast<std::size_t>(frame_width_x) * frame_height_x;
        for (uint32_t bg = 0; bg < n_backgrounds; ++bg) {
            const std::size_t offset = static_cast<std::size_t>(bg) * pixels;
            const uint16_t* bg_data = background_frames_565.empty()
                ? nullptr
                : background_frames_565.data() + offset;
            if (bg_data && pixels > 0) {
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

            const std::size_t offset_x = static_cast<std::size_t>(bg) * pixels_x;
            const uint16_t* bg_data_x = background_frames_x_565.empty()
                ? nullptr
                : background_frames_x_565.data() + offset_x;
            if (bg_data_x && pixels_x > 0) {
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
        const std::size_t pixels = static_cast<std::size_t>(frame_width) * frame_height;
        for (uint32_t i = 0; i < n_frames; ++i) {
            cv::Mat mask(static_cast<int>(frame_height), static_cast<int>(frame_width), CV_8UC1, cv::Scalar(0));
            const std::size_t offset = static_cast<std::size_t>(i) * pixels;
            std::memcpy(mask.data, background_masks.data() + offset, pixels);
            out.background_masks.push_back(mask);
        }
    } else {
        out.background_masks.resize(n_frames);
    }
    if (!background_masks_x.empty()) {
        const std::size_t pixels_x = static_cast<std::size_t>(frame_width_x) * frame_height_x;
        for (uint32_t i = 0; i < n_frames; ++i) {
            cv::Mat mask(static_cast<int>(frame_height_x), static_cast<int>(frame_width_x), CV_8UC1, cv::Scalar(0));
            const std::size_t offset = static_cast<std::size_t>(i) * pixels_x;
            std::memcpy(mask.data, background_masks_x.data() + offset, pixels_x);
            out.background_masks_x.push_back(mask);
        }
    } else {
        out.background_masks_x.resize(n_frames);
    }

    out.sprite_det_areas = det_areas;
    out.sprite_det_dwords = det_dwords;
    out.sprite_det_dword_pos = det_dword_pos;
    out.frame_sprite_bboxes = frame_sprite_bboxes;
    out.sprite_col_from_frame = sprite_col_from_frame;
    out.sprite_rects = sprite_rects;
    out.sprite_rect_mirror = sprite_rect_mirror;
    out.frame_rotations = color_rotations;
    out.frame_rotations_x = color_rotations_x;
    out.hash_codes = hash_codes;
    CopySparseVector(data.activeframes,
                     ElementSizeOr(data.activeframes, 1u),
                     n_frames,
                     out.active_frames);
    out.trigger_ids = trigger_ids;
    out.dynashadow_dir = dynashadow_dir;
    out.dynashadow_col = dynashadow_col;
    out.dynashadow_dir_x = dynashadow_dir_x;
    out.dynashadow_col_x = dynashadow_col_x;

    out.sprite_dynamic_colors_x.clear();
    out.sprite_dynamic_masks_x.clear();
    out.sprite_dynamic_colors_x.reserve(n_sprites);
    out.sprite_dynamic_masks_x.reserve(n_sprites);
    if (!sprite_dyna_cols_x.empty()) {
        const std::size_t per_sprite = sprite_dyna_cols_per_sprite;
        for (uint32_t i = 0; i < n_sprites; ++i) {
            const std::size_t offset = static_cast<std::size_t>(i) * per_sprite;
            if (offset + per_sprite <= sprite_dyna_cols_x.size()) {
                std::vector<uint16_t> colors(per_sprite, 0);
                std::memcpy(colors.data(), sprite_dyna_cols_x.data() + offset,
                            colors.size() * sizeof(uint16_t));
                out.sprite_dynamic_colors_x.push_back(std::move(colors));
            } else {
                out.sprite_dynamic_colors_x.emplace_back();
            }
        }
    }
    if (!sprite_dyna_masks_x.empty()) {
        for (uint32_t i = 0; i < n_sprites; ++i) {
            const std::size_t offset = static_cast<std::size_t>(i) * sprite_pixels;
            cv::Mat mask(static_cast<int>(MAX_SPRITE_HEIGHT),
                         static_cast<int>(MAX_SPRITE_WIDTH),
                         CV_8UC1,
                         cv::Scalar(255));
            if (offset + sprite_pixels <= sprite_dyna_masks_x.size()) {
                std::memcpy(mask.data, sprite_dyna_masks_x.data() + offset,
                            sprite_pixels);
            }
            out.sprite_dynamic_masks_x.push_back(mask);
        }
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
        const std::size_t offset = static_cast<std::size_t>(index) * sprite_pixels;
        const uint16_t* sprite_colored = sprites_565.data() + offset;
        const uint8_t* sprite_orig = sprite_original.data() + offset;
        const uint8_t* sprite_mask = sprite_dyna_masks.empty() ? nullptr : sprite_dyna_masks.data() + offset;
        const uint16_t* sprite_cols = sprite_dyna_cols.empty() ? nullptr :
            sprite_dyna_cols.data() + static_cast<std::size_t>(index) * sprite_dyna_cols_per_sprite;
        cv::Mat sprite_original_mat(static_cast<int>(MAX_SPRITE_HEIGHT),
                                    static_cast<int>(MAX_SPRITE_WIDTH),
                                    CV_8UC1,
                                    cv::Scalar(255));
        if (!sprite_original.empty()) {
            std::memcpy(sprite_original_mat.data, sprite_orig,
                        sprite_pixels);
        }
        out.sprite_originals.push_back(sprite_original_mat.clone());

        if (has_extra_sprite) {
            cv::Mat sprite_mask_x_mat(static_cast<int>(MAX_SPRITE_HEIGHT),
                                      static_cast<int>(MAX_SPRITE_WIDTH),
                                      CV_8UC1,
                                      cv::Scalar(255));
            if (!sprite_mask_x.empty()) {
                std::memcpy(sprite_mask_x_mat.data, sprite_mask_x.data() + offset,
                            sprite_pixels);
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

        std::vector<uint16_t> colors(sprite_dyna_cols_per_sprite, 0);
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
                        sprite_pixels);
        }
        out.sprite_dynamic_masks.push_back(dyna_mask);
    }
    if (!sprite_shape_mode.empty()) {
        out.sprite_shape_modes = sprite_shape_mode;
    }
    if (out.sprite_shape_modes.size() < n_sprites) {
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
