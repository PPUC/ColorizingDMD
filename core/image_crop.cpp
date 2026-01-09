#include "image_crop.h"

#include <algorithm>

bool CropAndResizeSelection(const cv::Mat& src,
                            int crop_x,
                            int crop_y,
                            int crop_w,
                            int crop_h,
                            int out_w,
                            int out_h,
                            int filter,
                            cv::Mat& out64,
                            cv::Mat& out32)
{
    if (src.empty() || crop_w <= 0 || crop_h <= 0 || out_w <= 0 || out_h <= 0) {
        return false;
    }

    int ix = std::max(0, crop_x);
    int iy = std::max(0, crop_y);
    int max_w = src.cols - ix;
    int max_h = src.rows - iy;
    int wid = std::min(crop_w, max_w);
    int hei = std::min(crop_h, max_h);
    if (wid <= 0 || hei <= 0) {
        return false;
    }

    cv::Rect croprect(ix, iy, wid, hei);
    cv::Mat croppedimg = src(croprect);

    cv::resize(croppedimg, out64, cv::Size(out_w, out_h), 0, 0, filter);
    cv::resize(croppedimg, out32, cv::Size(out_w / 2, out_h / 2), 0, 0, filter);
    return true;
}
