#pragma once

#include <QMainWindow>

#include <QStringList>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

#include "legacy_project.h"

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
    enum class MaskKind {
        None,
        Comparison,
        Dynamic
    };
    enum class MaskMode {
        None,
        Comparison,
        Dynamic
    };
    struct UndoState {
        cv::Mat image;
        cv::Mat mask;
        MaskKind mask_kind = MaskKind::None;
        int mask_index = -1;
    };
    struct UndoStack {
        std::vector<UndoState> undo;
        std::vector<UndoState> redo;
    };
    enum class FrameHoverArea {
        None,
        Top,
        Bottom
    };
    enum class UndoTarget {
        Frame,
        Sprite,
        CompMask,
        DynMask
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
    bool saveProjectToPath(const QString& filename);
    bool saveLegacyProject(const QString& filename);
    LegacyProject buildLegacyProject(const QString& baseName) const;
    void refreshFramePreviews();
    void updateFramePreviewAt(int index);
    cv::Mat buildPreviewFrame(const cv::Mat& colorized,
                              const cv::Mat& reference,
                              const cv::Mat& hdFrame) const;
    void resetUndoStacks();
    void ensureUndoStacksSize();
    void pushUndoSnapshot(bool isFrame, int index);
    void pushMaskUndoSnapshot(MaskMode mode, int index);
    bool undoEdit(bool isFrame);
    bool redoEdit(bool isFrame);
    bool undoMaskEdit(MaskMode mode);
    bool redoMaskEdit(MaskMode mode);
    void updateUndoActions();
    bool isFrameContext() const;
    UndoTarget currentUndoTarget() const;
    void setHdMode(bool enabled);
    bool hasHdFrame(int index) const;
    cv::Mat* activeFrameImage(int index, bool forEdit);
    void ensureMaskDataSize();
    cv::Mat buildReferenceFrame(const cv::Mat& source) const;
    cv::Mat buildMaskPreview(const cv::Mat& frame, const cv::Mat& mask, const cv::Vec3b& color) const;
    cv::Mat buildOriginalFrame(const cv::Mat& reference) const;
    cv::Mat buildCombinedFrame(const cv::Mat& colorized,
                               const cv::Mat& reference,
                               const cv::Scalar& gapColor) const;
    cv::Mat buildCombinedMaskPreview(const cv::Mat& colorized,
                                     const cv::Mat& reference,
                                     const cv::Mat& mask,
                                     const cv::Vec3b& color,
                                     const cv::Scalar& gapColor) const;
    cv::Mat buildOriginalPreviewForIndex(int index) const;
    void updateMaskPreviewForFrame(int index);
    void setMaskMode(MaskMode mode);
    void applyToolToMask(cv::Mat& mask, DrawTool tool, const QPoint& start, const QPoint& end, bool erase);
    void applyMaskFill(cv::Mat& mask, int x, int y, bool erase);
    void refreshMaskCombos();
    void refreshDynamicMaskCombos();
    void updateMaskPreviewIcons();
    void updateDynamicMaskPreviewIcons();
    cv::Mat buildMaskIconImage(const cv::Mat& mask, const cv::Vec3b& color) const;
    void applyMaskListOrder();
    void applyDynamicMaskListOrder();
    void updateFrameCanvasImage(int index);
    int currentFrameMaskId() const;
    int currentFrameDynamicMaskId() const;
    void setCurrentFrameMaskId(int id);
    void setCurrentFrameDynamicMaskId(int id);
    cv::Mat* activeComparisonMask();
    cv::Mat* activeDynamicMask();
    void swapMaskEntries(int a, int b);
    void swapDynamicMaskEntries(int a, int b);
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
    class QListWidget* m_framePreviewList;
    class QListWidget* m_maskList;
    class QToolButton* m_maskMoveUp;
    class QToolButton* m_maskMoveDown;
    class QPushButton* m_maskClearButton;
    class QListWidget* m_dynamicMaskList;
    class QToolButton* m_dynamicMaskMoveUp;
    class QToolButton* m_dynamicMaskMoveDown;
    class QPushButton* m_dynamicMaskClearButton;
    class QComboBox* m_frameMaskAssign;
    class QComboBox* m_frameDynamicMaskAssign;
    class QCheckBox* m_shapeCompToggle;
    class QComboBox* m_hdSourceCombo;
    class QComboBox* m_hdScaleCombo;
    class QPushButton* m_hdCreateButton;
    class QPushButton* m_hdDeleteButton;
    class QTabWidget* m_canvasTabs;
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
    std::vector<UndoStack> m_frameUndoStacks;
    std::vector<UndoStack> m_frameHdUndoStacks;
    std::vector<UndoStack> m_spriteUndoStacks;
    std::vector<UndoStack> m_compMaskUndoStacks;
    std::vector<UndoStack> m_dynMaskUndoStacks;
    std::vector<cv::Mat> m_frameRefs;
    std::vector<std::vector<uint16_t>> m_frameDynamicColors;
    std::vector<cv::Mat> m_compMasks;
    std::vector<cv::Mat> m_dynamicMasks;
    std::vector<cv::Mat> m_frameExtraFrames;
    std::vector<uint8_t> m_frameExtraFlags;
    std::vector<uint8_t> m_frameCompMaskIds;
    std::vector<uint8_t> m_frameDynamicMaskIds;
    std::vector<uint8_t> m_frameShapeCompModes;

    bool m_drawPointEnabled = false;
    MaskMode m_maskMode = MaskMode::None;
    bool m_frameUndoActive = false;
    bool m_spriteUndoActive = false;
    bool m_maskReorderActive = false;
    bool m_dynamicMaskReorderActive = false;
    bool m_showOriginalFrame = true;
    bool m_frameDrawOnMask = false;
    FrameHoverArea m_frameHoverArea = FrameHoverArea::None;
    bool m_useHdFrame = false;
    class QAction* m_undoAction;
    class QAction* m_redoAction;
    DrawTool m_drawTool = DrawTool::Point;
    cv::Scalar m_drawColor = cv::Scalar(255, 255, 255, 255);
    bool m_frameHasStart = false;
    bool m_spriteHasStart = false;
    QPoint m_frameStart;
    QPoint m_spriteStart;
    Qt::MouseButton m_frameStartButton = Qt::NoButton;
    Qt::MouseButton m_spriteStartButton = Qt::NoButton;
    uint32_t m_noColors = 64;
};
