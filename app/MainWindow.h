#pragma once

#include <QMainWindow>

#include <QStringList>
#include <string>
#include <vector>

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);

private:
    void updateWindowTitle();
    void refreshRecentMenu();
    void refreshImageList();
    void refreshCounts();
    void refreshFrameSpriteLists();
    void setInspectorSelection(const QString& label);
    void updateSelectionFromLists();
    void showImageForPath(const QString& path);
    void showFrameAtIndex(int index);
    void showSpriteAtIndex(int index);
    void populateBookmarks(const std::vector<uint32_t>& frameStarts,
                           const std::vector<std::string>& names);
    void applyFrameFilter(const QString& text);
    void applySpriteFilter(const QString& text);
    void updateFrameJumpRange();

    class CanvasWidget* m_framesCanvas;
    class CanvasWidget* m_spritesCanvas;
    class CanvasWidget* m_imagesCanvas;
    class QLabel* m_toolsLabel;
    class QLabel* m_inspectorLabel;
    class QMenu* m_recentMenu;
    class QListWidget* m_framesList;
    class QListWidget* m_spritesList;
    class QListWidget* m_imagesList;
    class QLineEdit* m_frameFilter;
    class QLineEdit* m_spriteFilter;
    class QComboBox* m_bookmarksCombo;
    class QSpinBox* m_frameJump;
    class QLabel* m_projectLabel;
    class QLabel* m_countsLabel;
    class QLabel* m_selectionLabel;
    class ProjectState* m_state;
    class ImageStore* m_imageStore;
    class IndexedImageStore* m_frameStore;
    class IndexedImageStore* m_spriteStore;
};
