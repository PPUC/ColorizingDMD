#pragma once

#include <opencv2/opencv.hpp>

bool CropAndResizeSelection(const cv::Mat& src,
                            int crop_x,
                            int crop_y,
                            int crop_w,
                            int crop_h,
                            int out_w,
                            int out_h,
                            int filter,
                            cv::Mat& out64,
                            cv::Mat& out32);
