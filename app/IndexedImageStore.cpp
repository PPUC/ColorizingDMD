#include "IndexedImageStore.h"

void IndexedImageStore::add(const cv::Mat& image)
{
    m_images.push_back(image.clone());
}

void IndexedImageStore::removeAt(int index)
{
    if (index < 0 || index >= m_images.size()) {
        return;
    }
    m_images.removeAt(index);
}

void IndexedImageStore::clear()
{
    m_images.clear();
}

const cv::Mat* IndexedImageStore::at(int index) const
{
    if (index < 0 || index >= m_images.size()) {
        return nullptr;
    }
    return &m_images[index];
}

int IndexedImageStore::count() const
{
    return m_images.size();
}
