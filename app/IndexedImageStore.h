#pragma once

#include <QVector>

#include <opencv2/opencv.hpp>

class IndexedImageStore
{
public:
    void add(const cv::Mat& image);
    void removeAt(int index);
    void clear();
    const cv::Mat* at(int index) const;
    cv::Mat* atMutable(int index);
    int count() const;

private:
    QVector<cv::Mat> m_images;
};
