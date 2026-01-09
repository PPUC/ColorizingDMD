#pragma once

#include <opencv2/opencv.hpp>

void ApplyBrightnessContrastAndBlur(cv::Mat& mat, int brightness, int contrast, int blur);
cv::Mat PrepareImageMat(const cv::Mat& src, int brightness, int contrast, int blur);
cv::Mat LoadAndPrepareImage(const char* filename, int brightness, int contrast, int blur);
cv::Mat GetFrameAtTime(cv::VideoCapture& cap, long hour, long minute, long second, long frame, long* out_frame_rate);
cv::Mat LoadAndPrepareVideoFrame(cv::VideoCapture& cap, long hour, long minute, long second, long frame, int brightness, int contrast, int blur, long* out_frame_rate);
bool IsImageFile(const char* filename);
bool CanOpenVideoFile(const char* filename);
