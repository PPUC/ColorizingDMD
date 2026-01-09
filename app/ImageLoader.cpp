#include "ImageLoader.h"

bool LoadImageFile(const QString& path, cv::Mat& out_image)
{
    out_image = cv::imread(path.toStdString(), cv::IMREAD_UNCHANGED);
    return !out_image.empty();
}
