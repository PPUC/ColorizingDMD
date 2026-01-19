#pragma once

#include <QList>
#include <QSet>
#include <QVector>
#include <QHash>

#include <opencv2/opencv.hpp>
#include <functional>
#include <memory>

class IndexedImageStore
{
public:
    void add(const cv::Mat& image);
    void removeAt(int index);
    void clear();
    const cv::Mat* at(int index) const;
    cv::Mat* atMutable(int index);
    const cv::Mat* peek(int index) const;
    bool isDirty(int index) const;
    int count() const;
    void setAdapter(std::function<int()> countFn,
                    std::function<cv::Mat(int)> loader,
                    std::function<void(int, const cv::Mat&)> saver);
    void setCacheLimit(int limit);
    void setCount(int count);
    void flush();
    void flushIndex(int index);

private:
    cv::Mat loadFromAdapter(int index) const;
    void touchCacheEntry(int index) const;
    void evictIfNeeded() const;

    QVector<cv::Mat> m_images;
    int m_adapterCount = -1;
    int m_cacheLimit = 64;
    std::function<int()> m_countFn;
    std::function<cv::Mat(int)> m_loader;
    std::function<void(int, const cv::Mat&)> m_saver;
    mutable QHash<int, std::shared_ptr<cv::Mat>> m_cache;
    mutable QList<int> m_cacheOrder;
    mutable QSet<int> m_dirty;
};
