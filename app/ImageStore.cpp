#include "ImageStore.h"

void ImageStore::addImage(const QString& path, const cv::Mat& image)
{
    if (path.isEmpty() || image.empty()) {
        return;
    }
    for (int i = 0; i < m_entries.size(); ++i) {
        if (m_entries[i].path == path) {
            m_entries.removeAt(i);
            break;
        }
    }
    m_entries.prepend({path, image.clone()});
}

void ImageStore::removeByPath(const QString& path)
{
    if (path.isEmpty()) {
        return;
    }
    for (int i = 0; i < m_entries.size(); ++i) {
        if (m_entries[i].path == path) {
            m_entries.removeAt(i);
            return;
        }
    }
}

void ImageStore::clear()
{
    m_entries.clear();
}

const ImageStore::Entry* ImageStore::findByPath(const QString& path) const
{
    if (path.isEmpty()) {
        return nullptr;
    }
    for (const auto& entry : m_entries) {
        if (entry.path == path) {
            return &entry;
        }
    }
    return nullptr;
}

const ImageStore::Entry* ImageStore::latest() const
{
    if (m_entries.isEmpty()) {
        return nullptr;
    }
    return &m_entries.last();
}
