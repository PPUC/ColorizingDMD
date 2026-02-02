#include "legacy_project_writer.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <vector>

#include "SerumData.h"
#include "serum.h"
#include "serum-version.h"
#include "serum_constants.h"

namespace {
template <typename T, typename U = T>
void MergeSparseFromSeed(SparseVector<T>& target,
                         const SparseVector<T>& seed,
                         std::size_t element_size_override = 0,
                         SparseVector<U>* parent = nullptr)
{
    const std::size_t element_size =
        element_size_override > 0 ? element_size_override : seed.elementCount();
    if (element_size == 0) {
        return;
    }
    const std::vector<uint32_t> ids = seed.elementIds();
    for (const uint32_t id : ids) {
        if (target.hasData(id)) {
            continue;
        }
        const T* values = seed[id];
        if (target.isIndexStorage()) {
            target.setIndex(id, values, element_size);
        } else {
            target.set(id, values, element_size, parent);
        }
    }
}

void MergeSerumDataFromSeed(SerumData& data, const SerumData& seed)
{
    MergeSparseFromSeed(data.hashcodes, seed.hashcodes, 1);
    MergeSparseFromSeed(data.shapecompmode, seed.shapecompmode, 1);
    MergeSparseFromSeed(data.compmaskID, seed.compmaskID, 1);
    MergeSparseFromSeed(data.movrctID, seed.movrctID, 1);
    MergeSparseFromSeed(data.isextraframe, seed.isextraframe, 1);
    MergeSparseFromSeed(data.isextrasprite, seed.isextrasprite, 1);
    MergeSparseFromSeed(data.isextrabackground, seed.isextrabackground, 1);
    MergeSparseFromSeed(data.activeframes, seed.activeframes, 1);
    MergeSparseFromSeed(data.triggerIDs, seed.triggerIDs, 1);
    MergeSparseFromSeed(data.backgroundIDs, seed.backgroundIDs, 1);
    MergeSparseFromSeed(data.sprshapemode, seed.sprshapemode, 1);

    MergeSparseFromSeed(data.compmasks, seed.compmasks);
    MergeSparseFromSeed(data.movrcts, seed.movrcts);
    MergeSparseFromSeed(data.cpal, seed.cpal);
    MergeSparseFromSeed(data.cframes, seed.cframes);
    MergeSparseFromSeed(data.cframes_v2, seed.cframes_v2);
    MergeSparseFromSeed(data.dynamasks, seed.dynamasks);
    MergeSparseFromSeed(data.dyna4cols, seed.dyna4cols);
    MergeSparseFromSeed(data.dyna4cols_v2, seed.dyna4cols_v2);
    MergeSparseFromSeed(data.framesprites, seed.framesprites);
    MergeSparseFromSeed(data.spritedescriptionso, seed.spritedescriptionso);
    MergeSparseFromSeed(data.spritedescriptionsc, seed.spritedescriptionsc);
    MergeSparseFromSeed(data.spriteoriginal, seed.spriteoriginal);
    MergeSparseFromSeed(data.spritecolored, seed.spritecolored);
    MergeSparseFromSeed(data.colorrotations, seed.colorrotations);
    MergeSparseFromSeed(data.colorrotations_v2, seed.colorrotations_v2);
    MergeSparseFromSeed(data.spritedetdwords, seed.spritedetdwords);
    MergeSparseFromSeed(data.spritedetdwordpos, seed.spritedetdwordpos);
    MergeSparseFromSeed(data.spritedetareas, seed.spritedetareas);
    MergeSparseFromSeed(data.backgroundframes, seed.backgroundframes);
    MergeSparseFromSeed(data.backgroundframes_v2, seed.backgroundframes_v2);
    MergeSparseFromSeed(data.backgroundBB, seed.backgroundBB, 0, &data.backgroundIDs);
    MergeSparseFromSeed(data.backgroundmask, seed.backgroundmask, 0, &data.backgroundIDs);
    MergeSparseFromSeed(data.dynashadowsdir, seed.dynashadowsdir);
    MergeSparseFromSeed(data.dynashadowscol, seed.dynashadowscol);
    MergeSparseFromSeed(data.dynasprite4cols, seed.dynasprite4cols);
    MergeSparseFromSeed(data.dynaspritemasks, seed.dynaspritemasks);

    MergeSparseFromSeed(data.cframes_v2_extra, seed.cframes_v2_extra, 0, &data.isextraframe);
    MergeSparseFromSeed(data.dynamasks_extra, seed.dynamasks_extra, 0, &data.isextraframe);
    MergeSparseFromSeed(data.dyna4cols_v2_extra, seed.dyna4cols_v2_extra, 0, &data.isextraframe);
    MergeSparseFromSeed(data.colorrotations_v2_extra, seed.colorrotations_v2_extra, 0, &data.isextraframe);
    MergeSparseFromSeed(data.dynashadowsdir_extra, seed.dynashadowsdir_extra, 0, &data.isextraframe);
    MergeSparseFromSeed(data.dynashadowscol_extra, seed.dynashadowscol_extra, 0, &data.isextraframe);
    MergeSparseFromSeed(data.dynasprite4cols_extra, seed.dynasprite4cols_extra, 0, &data.isextraframe);
    MergeSparseFromSeed(data.dynaspritemasks_extra, seed.dynaspritemasks_extra, 0, &data.isextraframe);
    MergeSparseFromSeed(data.spritemask_extra, seed.spritemask_extra, 0, &data.isextrasprite);
    MergeSparseFromSeed(data.spritecolored_extra, seed.spritecolored_extra, 0, &data.isextrasprite);
    MergeSparseFromSeed(data.framespriteBB, seed.framespriteBB, 0, &data.framesprites);
    MergeSparseFromSeed(data.backgroundframes_v2_extra, seed.backgroundframes_v2_extra, 0,
                        &data.isextrabackground);
    MergeSparseFromSeed(data.backgroundmask_extra, seed.backgroundmask_extra, 0, &data.backgroundIDs);
}
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

void CopyRgb565FromBgr(const cv::Mat& image, const cv::Size& target, std::vector<uint16_t>& out)
{
    out.clear();
    if (target.width <= 0 || target.height <= 0) {
        return;
    }
    cv::Mat bgr = EnsureBgr(image, target);
    if (bgr.empty()) {
        out.resize(static_cast<std::size_t>(target.width) * target.height, 0);
        return;
    }
    out.resize(static_cast<std::size_t>(target.width) * target.height, 0);
    for (int y = 0; y < bgr.rows; ++y) {
        const cv::Vec3b* src = bgr.ptr<cv::Vec3b>(y);
        uint16_t* dst = out.data() + static_cast<std::size_t>(y) * bgr.cols;
        for (int x = 0; x < bgr.cols; ++x) {
            dst[x] = BgrToRgb565(src[x]);
        }
    }
}

void CopyUint8FromMat(const cv::Mat& image, const cv::Size& target, std::vector<uint8_t>& out, uint8_t fill)
{
    out.clear();
    if (target.width <= 0 || target.height <= 0) {
        return;
    }
    if (image.empty()) {
        out.resize(static_cast<std::size_t>(target.width) * target.height, fill);
        return;
    }
    cv::Mat gray;
    if (image.channels() == 1) {
        gray = image;
    } else {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }
    if (gray.size() != target) {
        cv::Mat resized;
        cv::resize(gray, resized, target, 0.0, 0.0, cv::INTER_NEAREST);
        gray = resized;
    }
    out.resize(static_cast<std::size_t>(target.width) * target.height, fill);
    if (!gray.empty()) {
        std::memcpy(out.data(), gray.data, out.size());
    }
}
}

bool SaveLegacyProjectRp(const std::string& rp_path,
                         const LegacyProject& project,
                         std::string* error)
{
    if (rp_path.empty()) {
        return true;
    }

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

    const uint32_t n_frames = static_cast<uint32_t>(project.frames.size());
    const uint32_t n_sprites = static_cast<uint32_t>(project.sprites.size());
    const uint32_t no_colors = project.no_colors > 0 ? project.no_colors : 64;

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
    uint32_t draw_col_mode = project.draw_col_mode;
    uint8_t draw_mode = project.draw_mode;
    int32_t mask_sel_mode = project.mask_sel_mode;
    uint32_t fill_mode = project.fill_mode;
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
    if (!project.sprite_col_from_frame.empty()) {
        const std::size_t copy = std::min(project.sprite_col_from_frame.size(),
                                          sprite_col_from_frame.size());
        std::copy_n(project.sprite_col_from_frame.begin(), copy, sprite_col_from_frame.begin());
    }
    std::vector<uint32_t> frame_duration(n_frames, 30);
    if (!project.frame_durations.empty()) {
        for (uint32_t i = 0; i < n_frames && i < project.frame_durations.size(); ++i) {
            frame_duration[i] = project.frame_durations[i];
        }
    }

    if (!project.active_col_sets.empty() &&
        project.active_col_sets.size() >= static_cast<std::size_t>(MAX_COL_SETS)) {
        std::copy_n(project.active_col_sets.begin(), MAX_COL_SETS, active_col_set.begin());
    }
    if (!project.reduced_palettes.empty() &&
        project.reduced_palettes.size() >= static_cast<std::size_t>(MAX_COL_SETS * 16)) {
        col_sets.assign(project.reduced_palettes.begin(),
                        project.reduced_palettes.begin() + static_cast<std::size_t>(MAX_COL_SETS * 16));
        if (project.active_col_sets.empty()) {
            std::fill(active_col_set.begin(), active_col_set.end(), 1);
        }
    }
    if (!project.reduced_palette_names.empty()) {
        for (uint32_t i = 0; i < MAX_COL_SETS && i < project.reduced_palette_names.size(); ++i) {
            WriteFixedString(name_col_set, i * 64, 64, project.reduced_palette_names[i]);
        }
    }
    ac_col_set = project.active_reduced_palette;
    pre_col_set = project.preview_reduced_palette;
    if (!project.mask_names.empty()) {
        const std::size_t copy = std::min(project.mask_names.size(), mask_names.size());
        std::copy_n(project.mask_names.begin(), copy, mask_names.begin());
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
    if (!project.sprite_rects.empty()) {
        const std::size_t copy = std::min(project.sprite_rects.size(), sprite_rect.size());
        std::copy_n(project.sprite_rects.begin(), copy, sprite_rect.begin());
    }
    if (!project.sprite_rect_mirror.empty()) {
        const std::size_t copy = std::min(project.sprite_rect_mirror.size(), sprite_rect_mirror.size());
        std::copy_n(project.sprite_rect_mirror.begin(), copy, sprite_rect_mirror.begin());
    }
    std::vector<uint16_t> palette(N_PALETTES * 64, 0);
    std::vector<uint16_t> edit_colors(16, 0);
    uint32_t n_image_pos_saves = project.n_image_pos_saves;
    std::vector<char> image_pos_name(N_IMAGE_POS_TO_SAVE * 64, 0);
    std::vector<int32_t> image_pos(N_IMAGE_POS_TO_SAVE * 16, 0);
    std::vector<char> pal_names(N_PALETTES * 64, 0);
    uint32_t is_imported = project.is_imported;
    uint32_t time_elapsed = project.time_elapsed;
    uint32_t is_pup_pack = project.is_pup_pack;
    std::vector<char> pup_pack(sizeof(wchar_t) * 256, 0);

    if (!project.palettes.empty() &&
        project.palettes.size() >= static_cast<std::size_t>(N_PALETTES * 64)) {
        palette.assign(project.palettes.begin(),
                       project.palettes.begin() + static_cast<std::size_t>(N_PALETTES * 64));
    }
    if (!project.palette_names.empty()) {
        for (uint32_t i = 0; i < N_PALETTES && i < project.palette_names.size(); ++i) {
            WriteFixedString(pal_names, i * 64, 64, project.palette_names[i]);
        }
    }
    if (!project.edit_colors.empty()) {
        const std::size_t copy = std::min(project.edit_colors.size(), edit_colors.size());
        std::copy_n(project.edit_colors.begin(), copy, edit_colors.begin());
    }
    if (!project.image_pos_names.empty()) {
        const std::size_t copy = std::min(project.image_pos_names.size(), image_pos_name.size());
        std::copy_n(project.image_pos_names.begin(), copy, image_pos_name.begin());
    }
    if (!project.image_pos_data.empty()) {
        const std::size_t copy = std::min(project.image_pos_data.size(), image_pos.size());
        std::copy_n(project.image_pos_data.begin(), copy, image_pos.begin());
    }
    if (!project.pup_pack.empty()) {
        const std::size_t copy = std::min(project.pup_pack.size(), pup_pack.size());
        std::copy_n(project.pup_pack.begin(), copy, pup_pack.begin());
    }

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

bool BuildConcentrateData(const LegacyProject& project,
                          SerumData& data,
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

    uint32_t frame_width_x = project.frame_width_x > 0 ? project.frame_width_x : frame_width;
    uint32_t frame_height_x = project.frame_height_x > 0 ? project.frame_height_x : frame_height;
    const uint32_t n_frames = static_cast<uint32_t>(project.frames.size());
    const uint32_t n_sprites = static_cast<uint32_t>(project.sprites.size());
    const uint32_t no_colors = project.no_colors > 0 ? project.no_colors : 64;
    const uint16_t n_backgrounds = static_cast<uint16_t>(project.background_frames.size());

    if (!project.frames_x.empty()) {
        for (const auto& frame_x : project.frames_x) {
            if (!frame_x.empty()) {
                frame_width_x = static_cast<uint32_t>(frame_x.cols);
                frame_height_x = static_cast<uint32_t>(frame_x.rows);
                break;
            }
        }
    }
    if (!project.background_frames_x.empty()) {
        for (const auto& bg_x : project.background_frames_x) {
            if (!bg_x.empty()) {
                frame_width_x = static_cast<uint32_t>(bg_x.cols);
                frame_height_x = static_cast<uint32_t>(bg_x.rows);
                break;
            }
        }
    }

    uint32_t n_comp_masks = 0;
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

    data.Clear();
    data.SerumVersion = SERUM_V2;
    data.concentrateFileVersion = SERUM_CONCENTRATE_VERSION;
    std::fill(std::begin(data.rname), std::end(data.rname), '\0');
    const std::string baseName = project.name;
    const std::size_t nameCopy = std::min(baseName.size(), sizeof(data.rname) - 1);
    std::memcpy(data.rname, baseName.data(), nameCopy);
    data.fwidth = frame_width;
    data.fheight = frame_height;
    data.fwidth_extra = frame_width_x;
    data.fheight_extra = frame_height_x;
    data.nframes = n_frames;
    data.nocolors = no_colors;
    data.nccolors = no_colors;
    data.ncompmasks = n_comp_masks;
    data.nmovmasks = 0;
    data.nsprites = n_sprites;
    data.nbackgrounds = n_backgrounds;
    data.is256x64 = (frame_width == 256 && frame_height == 64);

    for (uint32_t i = 0; i < n_frames; ++i) {
        const uint32_t hash = (i < project.hash_codes.size()) ? project.hash_codes[i] : 0;
        data.hashcodes.setIndex(i, &hash, 1);
        const uint8_t shape = (i < project.frame_shape_comp_modes.size()) ? project.frame_shape_comp_modes[i] : 0;
        data.shapecompmode.set(i, &shape, 1);
        const uint8_t mask_id = (i < project.frame_comp_mask_ids.size()) ? project.frame_comp_mask_ids[i] : 255;
        data.compmaskID.set(i, &mask_id, 1);
        const uint8_t extra = (i < project.frame_extra_flags.size()) ? project.frame_extra_flags[i] : 0;
        data.isextraframe.setIndex(i, &extra, 1);
    }

    const std::size_t comp_mask_pixels = data.is256x64
        ? static_cast<std::size_t>(256 * 64)
        : mask_pixels;
    if (n_comp_masks > 0 && !project.comp_masks.empty()) {
        for (uint32_t i = 0; i < n_comp_masks && i < project.comp_masks.size(); ++i) {
            const cv::Mat& mask = project.comp_masks[i];
            if (!mask.empty() && mask.rows == static_cast<int>(frame_height) && mask.cols == static_cast<int>(frame_width)) {
                data.compmasks.set(i, mask.data, comp_mask_pixels);
            }
        }
    }

    for (uint32_t i = 0; i < n_frames; ++i) {
        std::vector<uint16_t> frame565;
        CopyRgb565FromBgr(i < project.frames.size() ? project.frames[i] : cv::Mat(),
                          cv::Size(static_cast<int>(frame_width), static_cast<int>(frame_height)),
                          frame565);
        data.cframes_v2.set(i, frame565.data(), frame565.size());

        const bool has_extra = (i < project.frame_extra_flags.size() && project.frame_extra_flags[i] != 0);
        if (has_extra) {
            std::vector<uint16_t> frame565x;
            const cv::Mat& src = (i < project.frames_x.size()) ? project.frames_x[i] : cv::Mat();
            CopyRgb565FromBgr(src,
                              cv::Size(static_cast<int>(frame_width_x), static_cast<int>(frame_height_x)),
                              frame565x);
            data.cframes_v2_extra.set(i, frame565x.data(), frame565x.size(), &data.isextraframe);
        }

        std::vector<uint8_t> dyn_mask;
        CopyUint8FromMat(i < project.frame_dynamic_mask_maps.size() ? project.frame_dynamic_mask_maps[i] : cv::Mat(),
                         cv::Size(static_cast<int>(frame_width), static_cast<int>(frame_height)),
                         dyn_mask,
                         255);
        data.dynamasks.set(i, dyn_mask.data(), dyn_mask.size());

        std::vector<uint8_t> dyn_mask_x;
        if (has_extra) {
            CopyUint8FromMat(i < project.frame_dynamic_mask_maps_x.size() ? project.frame_dynamic_mask_maps_x[i] : cv::Mat(),
                             cv::Size(static_cast<int>(frame_width_x), static_cast<int>(frame_height_x)),
                             dyn_mask_x,
                             255);
            data.dynamasks_extra.set(i, dyn_mask_x.data(), dyn_mask_x.size(), &data.isextraframe);
        }

        std::vector<uint16_t> dyna_cols(MAX_DYNA_SETS_PER_FRAMEN * no_colors, 0);
        if (i < project.frame_dynamic_colors.size() &&
            project.frame_dynamic_colors[i].size() >= dyna_cols.size()) {
            std::copy_n(project.frame_dynamic_colors[i].begin(), dyna_cols.size(), dyna_cols.begin());
        }
        data.dyna4cols_v2.set(i, dyna_cols.data(), dyna_cols.size());

        if (has_extra) {
            std::vector<uint16_t> dyna_cols_x(MAX_DYNA_SETS_PER_FRAMEN * no_colors, 0);
            if (i < project.frame_dynamic_colors.size() &&
                project.frame_dynamic_colors[i].size() >= dyna_cols_x.size()) {
                std::copy_n(project.frame_dynamic_colors[i].begin(), dyna_cols_x.size(), dyna_cols_x.begin());
            }
            data.dyna4cols_v2_extra.set(i, dyna_cols_x.data(), dyna_cols_x.size(), &data.isextraframe);
        }

        const uint8_t active = (i < project.active_frames.size()) ? project.active_frames[i] : 0;
        data.activeframes.set(i, &active, 1);

        const uint32_t trigger = (i < project.trigger_ids.size()) ? project.trigger_ids[i] : 0xffffffffu;
        data.triggerIDs.set(i, &trigger, 1);

        const std::size_t rotation_block = MAX_LENGTH_COLOR_ROTATION * MAX_COLOR_ROTATIONN;
        if (project.frame_rotations.size() >= (static_cast<std::size_t>(i) + 1) * rotation_block) {
            const uint16_t* rotations = project.frame_rotations.data() +
                static_cast<std::size_t>(i) * rotation_block;
            data.colorrotations_v2.set(i, rotations, rotation_block);
        }
        if (has_extra && project.frame_rotations_x.size() >= (static_cast<std::size_t>(i) + 1) * rotation_block) {
            const uint16_t* rotations_x = project.frame_rotations_x.data() +
                static_cast<std::size_t>(i) * rotation_block;
            data.colorrotations_v2_extra.set(i, rotations_x, rotation_block, &data.isextraframe);
        }
    }

    if (n_frames > 0) {
        const std::size_t per_frame = MAX_SPRITES_PER_FRAME;
        for (uint32_t i = 0; i < n_frames; ++i) {
            std::vector<uint8_t> sprites(per_frame, 255);
            const std::size_t offset = static_cast<std::size_t>(i) * per_frame;
            if (offset + per_frame <= project.frame_sprites.size()) {
                std::copy_n(project.frame_sprites.begin() + offset, per_frame, sprites.begin());
            }
            data.framesprites.set(i, sprites.data(), sprites.size());

            std::vector<uint16_t> bboxes(per_frame * 4, 0);
            const std::size_t bbox_offset = static_cast<std::size_t>(i) * per_frame * 4;
            if (bbox_offset + bboxes.size() <= project.frame_sprite_bboxes.size()) {
                std::copy_n(project.frame_sprite_bboxes.begin() + bbox_offset, bboxes.size(), bboxes.begin());
            }
            data.framespriteBB.set(i, bboxes.data(), bboxes.size(), &data.framesprites);
        }
    }

    const std::size_t sprite_pixels =
        static_cast<std::size_t>(MAX_SPRITE_WIDTH) * MAX_SPRITE_HEIGHT;
    const std::size_t dynasprite_cols_per_sprite =
        static_cast<std::size_t>(MAX_DYNA_SETS_PER_SPRITE) * no_colors;
    for (uint32_t i = 0; i < n_sprites; ++i) {
        const uint8_t extra = (i < project.sprite_extra_flags.size()) ? project.sprite_extra_flags[i] : 0;
        data.isextrasprite.setIndex(i, &extra, 1);

        std::vector<uint8_t> orig(sprite_pixels, 255);
        if (i < project.sprite_originals.size() && !project.sprite_originals[i].empty()) {
            std::memcpy(orig.data(), project.sprite_originals[i].data, sprite_pixels);
        }
        data.spriteoriginal.set(i, orig.data(), orig.size());

        std::vector<uint16_t> colored;
        const cv::Mat& colored_src = (i < project.sprite_colored.size()) ? project.sprite_colored[i] : cv::Mat();
        CopyRgb565FromBgr(colored_src,
                          cv::Size(static_cast<int>(MAX_SPRITE_WIDTH), static_cast<int>(MAX_SPRITE_HEIGHT)),
                          colored);
        data.spritecolored.set(i, colored.data(), colored.size());

        std::vector<uint16_t> colored_extra;
        if (extra) {
            const cv::Mat& colored_x = (i < project.sprite_colored_x.size()) ? project.sprite_colored_x[i] : cv::Mat();
            CopyRgb565FromBgr(colored_x,
                              cv::Size(static_cast<int>(MAX_SPRITE_WIDTH), static_cast<int>(MAX_SPRITE_HEIGHT)),
                              colored_extra);
            data.spritecolored_extra.set(i, colored_extra.data(), colored_extra.size(), &data.isextrasprite);
        }

        if (extra) {
            std::vector<uint8_t> mask_extra(sprite_pixels, 255);
            if (i < project.sprite_masks_x.size() && !project.sprite_masks_x[i].empty()) {
                std::memcpy(mask_extra.data(), project.sprite_masks_x[i].data, sprite_pixels);
            }
            data.spritemask_extra.set(i, mask_extra.data(), mask_extra.size(), &data.isextrasprite);
        }

        std::vector<uint8_t> dyn_mask(sprite_pixels, 255);
        if (i < project.sprite_dynamic_masks.size() && !project.sprite_dynamic_masks[i].empty()) {
            std::memcpy(dyn_mask.data(), project.sprite_dynamic_masks[i].data, sprite_pixels);
        }
        data.dynaspritemasks.set(i, dyn_mask.data(), dyn_mask.size());

        std::vector<uint8_t> dyn_mask_x(sprite_pixels, 255);
        if (extra && i < project.sprite_dynamic_masks_x.size() && !project.sprite_dynamic_masks_x[i].empty()) {
            std::memcpy(dyn_mask_x.data(), project.sprite_dynamic_masks_x[i].data, sprite_pixels);
            data.dynaspritemasks_extra.set(i, dyn_mask_x.data(), dyn_mask_x.size(), &data.isextrasprite);
        }

        std::vector<uint16_t> dyn_cols(dynasprite_cols_per_sprite, 0);
        if (i < project.sprite_dynamic_colors.size() &&
            project.sprite_dynamic_colors[i].size() >= dyn_cols.size()) {
            std::copy_n(project.sprite_dynamic_colors[i].begin(), dyn_cols.size(), dyn_cols.begin());
        }
        data.dynasprite4cols.set(i, dyn_cols.data(), dyn_cols.size());

        if (extra) {
            std::vector<uint16_t> dyn_cols_x(dynasprite_cols_per_sprite, 0);
            if (i < project.sprite_dynamic_colors_x.size() &&
                project.sprite_dynamic_colors_x[i].size() >= dyn_cols_x.size()) {
                std::copy_n(project.sprite_dynamic_colors_x[i].begin(), dyn_cols_x.size(), dyn_cols_x.begin());
            }
            data.dynasprite4cols_extra.set(i, dyn_cols_x.data(), dyn_cols_x.size(), &data.isextrasprite);
        }

        const uint8_t shape_mode = (i < project.sprite_shape_modes.size()) ? project.sprite_shape_modes[i] : 0;
        data.sprshapemode.set(i, &shape_mode, 1);
    }

    for (uint32_t i = 0; i < n_sprites; ++i) {
        const std::size_t offset = static_cast<std::size_t>(i) * MAX_SPRITE_DETECT_AREAS * 4;
        if (offset + MAX_SPRITE_DETECT_AREAS * 4 <= project.sprite_det_areas.size()) {
            data.spritedetareas.set(i,
                                    project.sprite_det_areas.data() + offset,
                                    MAX_SPRITE_DETECT_AREAS * 4);
        } else {
            std::vector<uint16_t> empty(MAX_SPRITE_DETECT_AREAS * 4, 0);
            data.spritedetareas.set(i, empty.data(), empty.size());
        }

        const std::size_t dword_offset = static_cast<std::size_t>(i) * MAX_SPRITE_DETECT_AREAS;
        if (dword_offset + MAX_SPRITE_DETECT_AREAS <= project.sprite_det_dwords.size()) {
            data.spritedetdwords.set(i,
                                     project.sprite_det_dwords.data() + dword_offset,
                                     MAX_SPRITE_DETECT_AREAS);
        }
        if (dword_offset + MAX_SPRITE_DETECT_AREAS <= project.sprite_det_dword_pos.size()) {
            data.spritedetdwordpos.set(i,
                                       project.sprite_det_dword_pos.data() + dword_offset,
                                       MAX_SPRITE_DETECT_AREAS);
        }
    }

    for (uint32_t i = 0; i < n_backgrounds; ++i) {
        const uint8_t extra = (i < project.background_extra_flags.size()) ? project.background_extra_flags[i] : 0;
        data.isextrabackground.setIndex(i, &extra, 1);

        std::vector<uint16_t> bg565;
        const cv::Mat& bg = (i < project.background_frames.size()) ? project.background_frames[i] : cv::Mat();
        CopyRgb565FromBgr(bg,
                          cv::Size(static_cast<int>(frame_width), static_cast<int>(frame_height)),
                          bg565);
        data.backgroundframes_v2.set(i, bg565.data(), bg565.size());

        if (extra) {
            std::vector<uint16_t> bg565x;
            const cv::Mat& bgx = (i < project.background_frames_x.size()) ? project.background_frames_x[i] : cv::Mat();
            CopyRgb565FromBgr(bgx,
                              cv::Size(static_cast<int>(frame_width_x), static_cast<int>(frame_height_x)),
                              bg565x);
            data.backgroundframes_v2_extra.set(i, bg565x.data(), bg565x.size(), &data.isextrabackground);
        }
    }

    for (uint32_t i = 0; i < n_frames; ++i) {
        const uint16_t bg_id = (i < project.background_ids.size()) ? project.background_ids[i] : 0xffff;
        data.backgroundIDs.set(i, &bg_id, 1);

        std::vector<uint8_t> bg_mask;
        CopyUint8FromMat(i < project.background_masks.size() ? project.background_masks[i] : cv::Mat(),
                         cv::Size(static_cast<int>(frame_width), static_cast<int>(frame_height)),
                         bg_mask,
                         0);
        data.backgroundmask.set(i, bg_mask.data(), bg_mask.size(), &data.backgroundIDs);

        const bool has_extra = (i < project.frame_extra_flags.size() && project.frame_extra_flags[i] != 0);
        if (has_extra) {
            std::vector<uint8_t> bg_mask_x;
            CopyUint8FromMat(i < project.background_masks_x.size() ? project.background_masks_x[i] : cv::Mat(),
                             cv::Size(static_cast<int>(frame_width_x), static_cast<int>(frame_height_x)),
                             bg_mask_x,
                             0);
            data.backgroundmask_extra.set(i, bg_mask_x.data(), bg_mask_x.size(), &data.backgroundIDs);
        }

        std::vector<uint8_t> dyn_dir(MAX_DYNA_SETS_PER_FRAMEN, 0);
        if (project.dynashadow_dir.size() >= (static_cast<std::size_t>(i) + 1) * MAX_DYNA_SETS_PER_FRAMEN) {
            std::memcpy(dyn_dir.data(),
                        project.dynashadow_dir.data() + static_cast<std::size_t>(i) * MAX_DYNA_SETS_PER_FRAMEN,
                        dyn_dir.size());
        }
        data.dynashadowsdir.set(i, dyn_dir.data(), dyn_dir.size());

        std::vector<uint16_t> dyn_col(MAX_DYNA_SETS_PER_FRAMEN, 0);
        if (project.dynashadow_col.size() >= (static_cast<std::size_t>(i) + 1) * MAX_DYNA_SETS_PER_FRAMEN) {
            std::memcpy(dyn_col.data(),
                        project.dynashadow_col.data() + static_cast<std::size_t>(i) * MAX_DYNA_SETS_PER_FRAMEN,
                        dyn_col.size() * sizeof(uint16_t));
        }
        data.dynashadowscol.set(i, dyn_col.data(), dyn_col.size());

        if (has_extra) {
            std::vector<uint8_t> dyn_dir_x(MAX_DYNA_SETS_PER_FRAMEN, 0);
            if (project.dynashadow_dir_x.size() >= (static_cast<std::size_t>(i) + 1) * MAX_DYNA_SETS_PER_FRAMEN) {
                std::memcpy(dyn_dir_x.data(),
                            project.dynashadow_dir_x.data() + static_cast<std::size_t>(i) * MAX_DYNA_SETS_PER_FRAMEN,
                            dyn_dir_x.size());
            }
            data.dynashadowsdir_extra.set(i, dyn_dir_x.data(), dyn_dir_x.size(), &data.isextraframe);

            std::vector<uint16_t> dyn_col_x(MAX_DYNA_SETS_PER_FRAMEN, 0);
            if (project.dynashadow_col_x.size() >= (static_cast<std::size_t>(i) + 1) * MAX_DYNA_SETS_PER_FRAMEN) {
                std::memcpy(dyn_col_x.data(),
                            project.dynashadow_col_x.data() + static_cast<std::size_t>(i) * MAX_DYNA_SETS_PER_FRAMEN,
                            dyn_col_x.size() * sizeof(uint16_t));
            }
            data.dynashadowscol_extra.set(i, dyn_col_x.data(), dyn_col_x.size(), &data.isextraframe);
        }
    }

    return true;
}

bool SaveConcentrateProject(const std::string& cromc_path,
                            const LegacyProject& project,
                            std::string* error)
{
    SerumData data;
    if (!BuildConcentrateData(project, data, error)) {
        return false;
    }
    if (data.rname[0] == '\0') {
        const std::string baseName = BaseName(cromc_path);
        const std::size_t nameCopy = std::min(baseName.size(), sizeof(data.rname) - 1);
        std::memset(data.rname, 0, sizeof(data.rname));
        std::memcpy(data.rname, baseName.data(), nameCopy);
    }
    if (!data.SaveToFile(cromc_path.c_str())) {
        if (error) {
            *error = "Failed to write .cROMc file";
        }
        return false;
    }
    return true;
}

bool SaveConcentrateProjectWithSeed(const std::string& cromc_path,
                                    const LegacyProject& project,
                                    const SerumData* seed,
                                    std::string* error)
{
    SerumData data;
    if (!BuildConcentrateData(project, data, error)) {
        return false;
    }
    if (seed) {
        MergeSerumDataFromSeed(data, *seed);
    }
    if (seed && seed->sceneGenerator && data.sceneGenerator) {
        data.sceneGenerator->setSceneData(seed->sceneGenerator->getSceneData());
        data.sceneGenerator->setDepth(data.nocolors == 16 ? 4 : 2);
    }
    if (data.rname[0] == '\0') {
        const std::string baseName = BaseName(cromc_path);
        const std::size_t nameCopy = std::min(baseName.size(), sizeof(data.rname) - 1);
        std::memset(data.rname, 0, sizeof(data.rname));
        std::memcpy(data.rname, baseName.data(), nameCopy);
    }
    if (!data.SaveToFile(cromc_path.c_str())) {
        if (error) {
            *error = "Failed to write .cROMc file";
        }
        return false;
    }
    return true;
}
