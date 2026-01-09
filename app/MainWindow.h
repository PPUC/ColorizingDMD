#pragma once

#include <QMainWindow>

#include <QStringList>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);

private:
    enum class DrawTool {
        Point,
        Line,
        Rect,
        RectFill,
        Circle,
        CircleFill,
        Ellipse,
        EllipseFill,
        ColorPicker,
        MagicFill
    };

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
    void updateMetadataForFrame(int index);
    void updateMetadataForSprite(int index);
    void openProjectFile(const QString& filename);
    void persistRecentFiles();
    void handleToolPress(bool isFrame, int x, int y, Qt::MouseButton button);
    void handleToolDrag(bool isFrame, int x, int y, Qt::MouseButtons buttons);
    void handleToolRelease(bool isFrame, int x, int y, Qt::MouseButton button);
    void applyToolToImage(cv::Mat& image, DrawTool tool, const QPoint& start, const QPoint& end, bool erase);
    void applyMagicFill(cv::Mat& image, int x, int y);
    void pickColorFromImage(const cv::Mat& image, int x, int y);
    cv::Scalar currentDrawColor(bool erase) const;
    void cancelCurrentDraw();

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
    class QLabel* m_frameMetaLabel;
    class QLabel* m_spriteMetaLabel;
    class ProjectState* m_state;
    class ImageStore* m_imageStore;
    class IndexedImageStore* m_frameStore;
    class IndexedImageStore* m_spriteStore;

    std::vector<uint32_t> m_frameDurations;
    std::vector<std::string> m_spriteNames;
    std::vector<uint32_t> m_sectionStarts;
    std::vector<std::string> m_sectionNames;

    bool m_drawPointEnabled = false;
    DrawTool m_drawTool = DrawTool::Point;
    cv::Scalar m_drawColor = cv::Scalar(255, 255, 255, 255);
    bool m_frameHasStart = false;
    bool m_spriteHasStart = false;
    QPoint m_frameStart;
    QPoint m_spriteStart;
    Qt::MouseButton m_frameStartButton = Qt::NoButton;
    Qt::MouseButton m_spriteStartButton = Qt::NoButton;
};
