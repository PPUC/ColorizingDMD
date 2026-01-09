#include "ProjectState.h"

ProjectState::ProjectState(QObject* parent)
    : QObject(parent)
    , m_frameCount(0)
    , m_spriteCount(0)
{
}

void ProjectState::newProject()
{
    setProjectPath(QString());
    m_images.clear();
    m_frames.clear();
    m_sprites.clear();
    m_frameCount = 0;
    m_spriteCount = 0;
    emit imagesChanged(m_images);
    emit framesChanged(m_frames);
    emit spritesChanged(m_sprites);
    emit countsChanged(m_frameCount, m_spriteCount);
}

void ProjectState::openProject(const QString& path)
{
    setProjectPath(path);
    addRecentFile(path);
    m_images.clear();
    m_frames.clear();
    m_sprites.clear();
    m_frameCount = 0;
    m_spriteCount = 0;
    emit imagesChanged(m_images);
    emit framesChanged(m_frames);
    emit spritesChanged(m_sprites);
    emit countsChanged(m_frameCount, m_spriteCount);
}

void ProjectState::saveProject(const QString& path)
{
    setProjectPath(path);
    addRecentFile(path);
}

void ProjectState::addRecentFile(const QString& path)
{
    if (path.isEmpty()) {
        return;
    }
    m_recentFiles.removeAll(path);
    m_recentFiles.prepend(path);
    const int maxEntries = 8;
    while (m_recentFiles.size() > maxEntries) {
        m_recentFiles.removeLast();
    }
    emit recentFilesChanged(m_recentFiles);
}

void ProjectState::addImportedImage(const QString& path)
{
    if (path.isEmpty()) {
        return;
    }
    m_images.removeAll(path);
    m_images.prepend(path);
    emit imagesChanged(m_images);
}

void ProjectState::addFrame()
{
    m_frames.append(QString("Frame %1").arg(m_frames.size()));
    m_frameCount = m_frames.size();
    emit framesChanged(m_frames);
    emit countsChanged(m_frameCount, m_spriteCount);
}

void ProjectState::addSprite()
{
    m_sprites.append(QString("Sprite %1").arg(m_sprites.size()));
    m_spriteCount = m_sprites.size();
    emit spritesChanged(m_sprites);
    emit countsChanged(m_frameCount, m_spriteCount);
}

void ProjectState::removeFrame(int index)
{
    if (index < 0 || index >= m_frames.size()) {
        return;
    }
    m_frames.removeAt(index);
    m_frameCount = m_frames.size();
    emit framesChanged(m_frames);
    emit countsChanged(m_frameCount, m_spriteCount);
}

void ProjectState::removeSprite(int index)
{
    if (index < 0 || index >= m_sprites.size()) {
        return;
    }
    m_sprites.removeAt(index);
    m_spriteCount = m_sprites.size();
    emit spritesChanged(m_sprites);
    emit countsChanged(m_frameCount, m_spriteCount);
}

void ProjectState::removeImage(int index)
{
    if (index < 0 || index >= m_images.size()) {
        return;
    }
    m_images.removeAt(index);
    emit imagesChanged(m_images);
}

void ProjectState::setFramesAndSprites(const QStringList& frames, const QStringList& sprites)
{
    m_frames = frames;
    m_sprites = sprites;
    m_frameCount = m_frames.size();
    m_spriteCount = m_sprites.size();
    emit framesChanged(m_frames);
    emit spritesChanged(m_sprites);
    emit countsChanged(m_frameCount, m_spriteCount);
}

QString ProjectState::projectPath() const
{
    return m_projectPath;
}

QStringList ProjectState::recentFiles() const
{
    return m_recentFiles;
}

QStringList ProjectState::images() const
{
    return m_images;
}

QStringList ProjectState::frames() const
{
    return m_frames;
}

QStringList ProjectState::sprites() const
{
    return m_sprites;
}

int ProjectState::frameCount() const
{
    return m_frameCount;
}

int ProjectState::spriteCount() const
{
    return m_spriteCount;
}

void ProjectState::setProjectPath(const QString& path)
{
    if (m_projectPath == path) {
        return;
    }
    m_projectPath = path;
    emit projectPathChanged(m_projectPath);
}
