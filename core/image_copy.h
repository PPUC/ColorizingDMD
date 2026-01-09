#pragma once

#include <cstdint>
#include <opencv2/opencv.hpp>

struct ImageCopyParams {
    uint16_t* frame32;
    uint16_t* frame64;
    unsigned int width32;
    unsigned int width64;
    unsigned int sel_x;
    unsigned int sel_y;
    unsigned int sel_w;
    unsigned int sel_h;
    const uint8_t* mask64;
    const uint8_t* mask32;
    unsigned int mask_stride64;
    unsigned int mask_stride32;
    bool copy_to_32;
    bool copy_to_64;
    bool use_mask;
};

void ApplyImageToFrames(const cv::Mat& tmp32, const cv::Mat& tmp64, const ImageCopyParams& params);
