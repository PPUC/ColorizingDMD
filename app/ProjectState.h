#pragma once

#include <QObject>
#include <QString>
#include <QStringList>

class ProjectState : public QObject
{
    Q_OBJECT
public:
    explicit ProjectState(QObject* parent = nullptr);

    void newProject();
    void openProject(const QString& path);
    void saveProject(const QString& path);
    void addRecentFile(const QString& path);
    void addImportedImage(const QString& path);
    void addFrame();
    void addSprite();
    void removeFrame(int index);
    void removeSprite(int index);
    void removeImage(int index);
    void setFramesAndSprites(const QStringList& frames, const QStringList& sprites);

    QString projectPath() const;
    QStringList recentFiles() const;
    QStringList images() const;
    QStringList frames() const;
    QStringList sprites() const;
    int frameCount() const;
    int spriteCount() const;

signals:
    void projectPathChanged(const QString& path);
    void recentFilesChanged(const QStringList& files);
    void imagesChanged(const QStringList& images);
    void framesChanged(const QStringList& frames);
    void spritesChanged(const QStringList& sprites);
    void countsChanged(int frames, int sprites);

private:
    void setProjectPath(const QString& path);

    QString m_projectPath;
    QStringList m_recentFiles;
    QStringList m_images;
    QStringList m_frames;
    QStringList m_sprites;
    int m_frameCount;
    int m_spriteCount;
};
