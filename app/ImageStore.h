#pragma once

#include <QString>
#include <QVector>

#include <opencv2/opencv.hpp>

class ImageStore
{
public:
    struct Entry {
        QString path;
        cv::Mat image;
    };

    void addImage(const QString& path, const cv::Mat& image);
    void removeByPath(const QString& path);
    void clear();
    const Entry* findByPath(const QString& path) const;
    const Entry* latest() const;

private:
    QVector<Entry> m_entries;
};
