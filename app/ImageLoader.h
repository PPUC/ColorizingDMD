#pragma once

#include <QString>
#include <opencv2/opencv.hpp>

bool LoadImageFile(const QString& path, cv::Mat& out_image);
