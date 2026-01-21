#include "IndexedImageStore.h"

namespace {
void RemoveFromOrder(QList<int>& order, int index)
{
    const int pos = order.indexOf(index);
    if (pos >= 0) {
        order.removeAt(pos);
    }
}
}  // namespace

void IndexedImageStore::add(const cv::Mat& image)
{
    if (m_loader) {
        const int index = count();
        if (m_saver) {
            m_saver(index, image);
        }
        setCount(index + 1);
        m_cache.insert(index, std::make_shared<cv::Mat>(image.clone()));
        touchCacheEntry(index);
        evictIfNeeded();
        return;
    }
    m_images.push_back(image.clone());
}

void IndexedImageStore::removeAt(int index)
{
    if (m_loader) {
        m_cache.remove(index);
        RemoveFromOrder(m_cacheOrder, index);
        m_dirty.remove(index);
        if (m_adapterCount > 0 && index >= 0 && index < m_adapterCount) {
            m_adapterCount -= 1;
        }
        return;
    }
    if (index < 0 || index >= m_images.size()) {
        return;
    }
    m_images.removeAt(index);
}

void IndexedImageStore::clear()
{
    m_images.clear();
    m_cache.clear();
    m_cacheOrder.clear();
    m_dirty.clear();
    m_adapterCount = -1;
    m_countFn = nullptr;
    m_loader = nullptr;
    m_saver = nullptr;
}

const cv::Mat* IndexedImageStore::at(int index) const
{
    if (index < 0 || index >= count()) {
        return nullptr;
    }
    if (m_loader) {
        auto it = m_cache.find(index);
        if (it == m_cache.end()) {
            cv::Mat image = loadFromAdapter(index);
            if (image.empty()) {
                return nullptr;
            }
            it = m_cache.insert(index, std::make_shared<cv::Mat>(std::move(image)));
            touchCacheEntry(index);
            evictIfNeeded();
        } else {
            touchCacheEntry(index);
        }
        return it.value().get();
    }
    if (index < 0 || index >= m_images.size()) {
        return nullptr;
    }
    return &m_images[index];
}

cv::Mat* IndexedImageStore::atMutable(int index)
{
    if (index < 0 || index >= count()) {
        return nullptr;
    }
    if (m_loader) {
        const cv::Mat* existing = at(index);
        if (!existing) {
            return nullptr;
        }
        m_dirty.insert(index);
        return m_cache.value(index).get();
    }
    if (index < 0 || index >= m_images.size()) {
        return nullptr;
    }
    return &m_images[index];
}

const cv::Mat* IndexedImageStore::peek(int index) const
{
    if (index < 0 || index >= count()) {
        return nullptr;
    }
    if (m_loader) {
        auto it = m_cache.find(index);
        return it == m_cache.end() ? nullptr : it.value().get();
    }
    if (index < 0 || index >= m_images.size()) {
        return nullptr;
    }
    return &m_images[index];
}

cv::Mat IndexedImageStore::loadCopy(int index) const
{
    if (index < 0 || index >= count()) {
        return cv::Mat();
    }
    if (m_loader) {
        if (const cv::Mat* cached = peek(index)) {
            return cached->clone();
        }
        return loadFromAdapter(index);
    }
    if (index < 0 || index >= m_images.size()) {
        return cv::Mat();
    }
    return m_images[index].clone();
}

bool IndexedImageStore::isDirty(int index) const
{
    if (!m_loader) {
        return false;
    }
    return m_dirty.contains(index);
}

int IndexedImageStore::count() const
{
    if (m_countFn) {
        return m_countFn();
    }
    if (m_adapterCount >= 0) {
        return m_adapterCount;
    }
    return m_images.size();
}

void IndexedImageStore::setAdapter(std::function<int()> countFn,
                                   std::function<cv::Mat(int)> loader,
                                   std::function<void(int, const cv::Mat&)> saver)
{
    m_countFn = std::move(countFn);
    m_loader = std::move(loader);
    m_saver = std::move(saver);
    m_cache.clear();
    m_cacheOrder.clear();
    m_dirty.clear();
}

void IndexedImageStore::setCacheLimit(int limit)
{
    m_cacheLimit = limit;
    evictIfNeeded();
}

void IndexedImageStore::setCount(int count)
{
    m_adapterCount = count;
}

void IndexedImageStore::flush()
{
    if (!m_saver) {
        m_dirty.clear();
        return;
    }
    for (int index : m_dirty) {
        if (m_cache.contains(index)) {
            m_saver(index, *m_cache.value(index));
        }
    }
    m_dirty.clear();
}

void IndexedImageStore::flushIndex(int index)
{
    if (!m_saver || !m_dirty.contains(index)) {
        return;
    }
    if (m_cache.contains(index)) {
        m_saver(index, *m_cache.value(index));
    }
    m_dirty.remove(index);
}

cv::Mat IndexedImageStore::loadFromAdapter(int index) const
{
    if (!m_loader) {
        return cv::Mat();
    }
    return m_loader(index);
}

void IndexedImageStore::touchCacheEntry(int index) const
{
    RemoveFromOrder(m_cacheOrder, index);
    m_cacheOrder.prepend(index);
}

void IndexedImageStore::evictIfNeeded() const
{
    if (m_cacheLimit <= 0) {
        return;
    }
    while (m_cacheOrder.size() > m_cacheLimit) {
        const int index = m_cacheOrder.takeLast();
        if (m_dirty.contains(index) && m_saver && m_cache.contains(index)) {
            m_saver(index, *m_cache.value(index));
            m_dirty.remove(index);
        }
        m_cache.remove(index);
    }
}
