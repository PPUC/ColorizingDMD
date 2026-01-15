#pragma once

#include <QMainWindow>

#include <QStringList>
#include <QElapsedTimer>
#include <QTimer>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

#include "legacy_project.h"
#include "serum-editor.h"

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
        Dynamic,
        SpriteDetAreas
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
    struct SpriteZoneGroup {
        QRect rect;
        std::vector<int> slotIndices;
        std::vector<int> sprites;
    };
    enum class FrameHoverArea {
        None,
        Top,
        Bottom
    };
    enum class UndoTarget {
        Frame,
        Sprite,
        Background,
        CompMask,
        DynMask,
        BackgroundMask,
        Palette
    };
    enum class PreviewFilterKind {
        None,
        Mask,
        DynamicMask,
        Background,
        Sprite
    };

    void updateWindowTitle();
    void refreshRecentMenu();
    void refreshImageList();
    void refreshCounts();
    void refreshFrameSpriteLists();
    void refreshBackgroundList();
    void refreshSpriteZoneList();
    void refreshFrameSpriteSlotCombo();
    void setInspectorSelection(const QString& label);
    void updateSelectionFromLists();
    bool eventFilter(QObject* obj, QEvent* event) override;
    void showImageForPath(const QString& path);
    void showFrameAtIndex(int index);
    void showSpriteAtIndex(int index);
    void showBackgroundAtIndex(int index);
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
    void refreshFramePreviewSelection();
    void updatePreviewSelectionStyles();
    std::vector<int> selectedPreviewFrameIndices() const;
    std::vector<int> targetFrameIndices() const;
    void updatePreviewsForMaskId(int maskId);
    int previewRowForFrame(int index) const;
    PreviewFilterKind currentPreviewFilterKind() const;
    void updatePreviewFilterState();
    std::vector<int> buildPreviewFrameIndices() const;
    void refreshAllPreviews();
    cv::Mat buildPreviewFrame(int index,
                              const cv::Mat& colorized,
                              const cv::Mat& reference,
                              const cv::Mat& hdFrame) const;
    cv::Mat applyRotationPreview(const cv::Mat& colorized,
                                 int frameIndex,
                                 bool useHd) const;
    void setFrameCanvasFromComposed(int index, const cv::Mat& composed);
    void setCanvasRotationEnabled(bool enabled);
    void updateCanvasRotationFrame();
    void resetCanvasRotationState();
    void refreshRotationEditor();
    void refreshRotationList();
    void updateRotationDelay(int delayMs);
    void updateRotationDataFromList();
    cv::Mat buildReferenceForSize(int index, const cv::Size& target) const;
    cv::Mat applyDynamicColors(int index, const cv::Mat& frame, bool useHd) const;
    cv::Mat applySpritesToFrame(int index, const cv::Mat& frame, bool useHd, bool drawOutline) const;
    cv::Mat applyBackgroundComposite(int index, const cv::Mat& frame, bool useHd) const;
    cv::Mat applyBackgroundCompositeWithMask(int index,
                                             const cv::Mat& frame,
                                             const cv::Mat& mask,
                                             bool useHd) const;
    std::vector<SpriteZoneGroup> buildSpriteZonesForFrame(int frameIndex) const;
    void resetUndoStacks();
    void ensureUndoStacksSize();
    void pushPaletteUndoSnapshot();
    void pushReducedUndoSnapshot(int setIndex);
    void pushDynamicUndoSnapshot(int frameIndex, int setIndex);
    bool undoPaletteEdit();
    bool redoPaletteEdit();
    void pushUndoSnapshot(bool isFrame, int index);
    void pushMaskUndoSnapshot(MaskMode mode, int index);
    void pushBackgroundMaskUndoSnapshot(int index);
    void pushBackgroundUndoSnapshot(int index);
    void pushSpriteDynamicMaskUndoSnapshot(int index);
    void pushSpriteDetAreaUndoSnapshot(int index);
    bool undoEdit(bool isFrame);
    bool redoEdit(bool isFrame);
    bool undoBackgroundEdit();
    bool redoBackgroundEdit();
    bool undoMaskEdit(MaskMode mode);
    bool redoMaskEdit(MaskMode mode);
    bool undoBackgroundMaskEdit();
    bool redoBackgroundMaskEdit();
    void updateUndoActions();
    bool isFrameContext() const;
    UndoTarget currentUndoTarget() const;
    void setHdMode(bool enabled);
    bool hasHdFrame(int index) const;
    cv::Mat* activeFrameImage(int index, bool forEdit);
    cv::Mat* activeBackgroundImage(int index, bool forEdit);
    void ensureMaskDataSize();
    void ensureSpriteDataSize();
    void ensureBackgroundDataSize();
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
    cv::Mat renderFrameWithSerum(int index, bool useHd) const;
    cv::Mat renderFrameWithSerum(int index, bool useHd, const cv::Mat& overrideColorized) const;
    cv::Mat buildOriginalPreviewForIndex(int index) const;
    cv::Mat buildSpriteCoverageMask(int index, bool useHd) const;
    void restoreSpriteCoverage(cv::Mat& target,
                               const cv::Mat& backup,
                               const cv::Mat& spriteMask) const;
    void updateMaskPreviewForFrame(int index);
    void updateSpriteZoneOverlay(int index);
    void setMaskMode(MaskMode mode);
    void setSpriteZoneMode(bool enabled);
    void updateSpriteDetAreaOverlay(int index);
    void setSpriteDetAreaMode(bool enabled);
    void applyToolToMask(cv::Mat& mask, DrawTool tool, const QPoint& start, const QPoint& end, bool erase);
    void applyMaskFill(cv::Mat& mask, int x, int y, bool erase);
    void refreshMaskCombos();
    void refreshDynamicMaskCombos();
    void updateMaskPreviewIcons();
    void updateDynamicMaskPreviewIcons();
    cv::Mat buildMaskIconImage(const cv::Mat& mask, const cv::Vec3b& color) const;
    void applyMaskListOrder();
    void applyDynamicMaskListOrder();
    bool dynamicMapHasSet(const cv::Mat& map, int setId) const;
    cv::Mat buildDynamicMaskFromMap(const cv::Mat& map, int setId) const;
    void updateFrameCanvasImage(int index);
    void updateSpriteCanvasImage(int index);
    void updateBackgroundCanvasImage(int index);
    cv::Mat applySpriteDynamicColors(int index, const cv::Mat& sprite) const;
    QRect spriteContentRect(int index) const;
    bool hasHdBackground(int index) const;
    void updateHdControlsForContext();
    int currentFrameMaskId() const;
    int currentFrameDynamicMaskId() const;
    void setCurrentFrameMaskId(int id);
    void setCurrentFrameDynamicMaskId(int id);
    cv::Mat* activeComparisonMask();
    cv::Mat* activeDynamicMaskMap(int frameIndex);
    cv::Mat* activeSpriteDynamicMask(int spriteIndex);
    cv::Mat* activeBackgroundMask(int index);
    void swapMaskEntries(int a, int b);
    void swapDynamicMaskEntries(int a, int b);
    void persistRecentFiles();
    void handleToolPress(bool isFrame, int x, int y, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void handleToolDrag(bool isFrame, int x, int y, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers);
    void handleToolRelease(bool isFrame, int x, int y, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void handleBackgroundToolPress(int x, int y, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void handleBackgroundToolDrag(int x, int y, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers);
    void handleBackgroundToolRelease(int x, int y, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void applyToolToImage(cv::Mat& image, DrawTool tool, const QPoint& start, const QPoint& end, bool erase);
    void applyMagicFill(cv::Mat& image, int x, int y);
    void pickColorFromImage(const cv::Mat& image, int x, int y);
    cv::Scalar currentDrawColor(bool erase) const;
    void applyToolToDynamicMask(cv::Mat& map, int setId, DrawTool tool, const QPoint& start, const QPoint& end, bool erase);
    void applyDynamicMaskFill(cv::Mat& map, int setId, int x, int y, bool erase);
    void initPalette();
    void refreshPaletteList();
    int reducedSlotCount() const;
    int dynamicSlotCount() const;
    int dynamicColorsPerSet(const std::vector<uint16_t>& colors) const;
    QColor reducedSlotColor(int setIndex, int slot) const;
    QColor dynamicSlotColor(int slot) const;
    void setReducedSlotColor(int setIndex, int slot, const QColor& color);
    void setDynamicSlotColor(int frameIndex, int setIndex, int slot, const QColor& color);
    void applyDynamicColorsToFrame(int frameIndex);
    void refreshReducedPaletteUI();
    void refreshReducedPaletteButtons();
    void refreshDynamicPaletteUI();
    void refreshDynamicPaletteButtons();
    void syncDynamicSetSelection();
    void setDrawColor(const QColor& color, bool updatePaletteSelection);
    void updateCurrentColorSwatch();
    void loadPaletteFromProject(const struct LegacyProject& legacy);
    void cancelCurrentDraw();
    void startPaletteGradient();
    void cancelPaletteGradient();
    void applyPaletteGradient(int startIndex, int endIndex);
    void startPaletteSetSlot();
    void startReducedSetSlot();
    void startDynamicSetSlot();
    void startRotationSetSlot();
    void cancelPaletteSetSlot();
    void keyPressEvent(QKeyEvent* event) override;

    class CanvasWidget* m_framesCanvas;
    class CanvasWidget* m_spritesCanvas;
    class CanvasWidget* m_imagesCanvas;
    class CanvasWidget* m_backgroundsCanvas;
    class QLabel* m_toolsLabel;
    class QLabel* m_inspectorLabel;
    class QMenu* m_recentMenu;
    class QListWidget* m_framesList;
    class QListWidget* m_spritesList;
    class QListWidget* m_imagesList;
    class QListWidget* m_backgroundList;
    class QListWidget* m_spriteZoneList;
    class QLineEdit* m_frameFilter;
    class QLineEdit* m_spriteFilter;
    class QListWidget* m_framePreviewList;
    class QToolButton* m_previewFilterButton;
    class QToolButton* m_previewFilterClearButton;
    class QToolButton* m_previewSelectedButton;
    class QToolButton* m_previewHdButton;
    class QToolButton* m_previewMaskOverlayButton;
    class QToolButton* m_previewRotateButton;
    class QToolButton* m_previewRefreshButton;
    class QListWidget* m_maskList;
    class QToolButton* m_maskMoveUp;
    class QToolButton* m_maskMoveDown;
    class QPushButton* m_maskClearButton;
    class QListWidget* m_dynamicMaskList;
    class QToolButton* m_dynamicMaskMoveUp;
    class QToolButton* m_dynamicMaskMoveDown;
    class QPushButton* m_dynamicMaskClearButton;
    class QLabel* m_backgroundAssignLabel;
    class QComboBox* m_frameMaskAssign;
    class QComboBox* m_frameDynamicMaskAssign;
    class QPushButton* m_frameDynamicCopyButton;
    class QComboBox* m_frameBackgroundAssign;
    class QComboBox* m_frameSpriteSlotCombo;
    class QCheckBox* m_shapeCompToggle;
    class QComboBox* m_spriteDynamicSetCombo;
    class QComboBox* m_spriteDetAreaCombo;
    class QPushButton* m_spriteDetAreaClearButton;
    class QComboBox* m_hdSourceCombo;
    class QComboBox* m_hdScaleCombo;
    class QPushButton* m_hdCreateButton;
    class QPushButton* m_hdDeleteButton;
    class QTabWidget* m_canvasTabs;
    class QTabWidget* m_toolsTabs;
    class QWidget* m_masksTab;
    class QWidget* m_dynamicMasksTab;
    class QWidget* m_backgroundsTab;
    class QWidget* m_spritesTab;
    class QWidget* m_colorsTab;
    class QWidget* m_spriteZonesTab;
    class QComboBox* m_rotationSetCombo;
    class QSpinBox* m_rotationDelaySpin;
    class QListWidget* m_rotationList;
    class QPushButton* m_rotationAddButton;
    class QPushButton* m_rotationRemoveButton;
    class QPushButton* m_rotationUpButton;
    class QPushButton* m_rotationDownButton;
    class QPushButton* m_rotationClearButton;
    class QPushButton* m_rotationAssignButton;
    class QComboBox* m_bookmarksCombo;
    class QSpinBox* m_frameJump;
    class QLabel* m_projectLabel;
    class QLabel* m_countsLabel;
    class QLabel* m_selectionLabel;
    class QLabel* m_frameMetaLabel;
    class QLabel* m_spriteMetaLabel;
    class QLabel* m_coordLabel;
    class QToolButton* m_currentColorButton;
    class QLabel* m_colorInfoLabel;
    class QPushButton* m_colorPickButton;
    class QPushButton* m_paletteAssignButton;
    class QPushButton* m_paletteGradientButton;
    class QListWidget* m_paletteList;
    class QComboBox* m_paletteSetCombo;
    class QComboBox* m_reducedSetCombo;
    class QPushButton* m_reducedAssignButton;
    class QListWidget* m_reducedPaletteList;
    class QComboBox* m_dynamicSetCombo;
    class QPushButton* m_dynamicAssignButton;
    class QListWidget* m_dynamicPaletteList;
    class ProjectState* m_state;
    class ImageStore* m_imageStore;
    class IndexedImageStore* m_frameStore;
    class IndexedImageStore* m_spriteStore;
    class IndexedImageStore* m_backgroundStore;

    std::vector<uint32_t> m_frameDurations;
    std::vector<std::string> m_spriteNames;
    std::vector<cv::Mat> m_spriteColored;
    std::vector<cv::Mat> m_spriteColoredX;
    std::vector<cv::Mat> m_spriteOriginals;
    std::vector<cv::Mat> m_spriteMasksX;
    std::vector<cv::Mat> m_spriteDynamicMasks;
    std::vector<cv::Mat> m_spriteDynamicMasksX;
    std::vector<std::vector<uint16_t>> m_spriteDynamicColors;
    std::vector<std::vector<uint16_t>> m_spriteDynamicColorsX;
    std::vector<uint8_t> m_spriteExtraFlags;
    std::vector<uint8_t> m_spriteShapeModes;
    std::vector<uint16_t> m_spriteDetAreas;
    std::vector<uint32_t> m_spriteDetDwords;
    std::vector<uint16_t> m_spriteDetDwordPos;
    std::vector<uint8_t> m_frameSpriteAssignments;
    std::vector<uint16_t> m_frameSpriteBBoxes;
    std::vector<uint32_t> m_spriteColFromFrame;
    std::vector<uint16_t> m_spriteRects;
    std::vector<uint32_t> m_spriteRectMirror;
    std::vector<SpriteZoneGroup> m_spriteZones;
    std::vector<uint32_t> m_sectionStarts;
    std::vector<std::string> m_sectionNames;
    std::vector<UndoStack> m_frameUndoStacks;
    std::vector<UndoStack> m_frameHdUndoStacks;
    std::vector<UndoStack> m_spriteUndoStacks;
    std::vector<UndoStack> m_backgroundUndoStacks;
    std::vector<UndoStack> m_backgroundHdUndoStacks;
    std::vector<UndoStack> m_compMaskUndoStacks;
    std::vector<UndoStack> m_dynMaskUndoStacks;
    std::vector<UndoStack> m_backgroundMaskUndoStacks;
    std::vector<cv::Mat> m_frameRefs;
    std::vector<std::vector<uint16_t>> m_frameDynamicColors;
    std::vector<cv::Mat> m_compMasks;
    std::vector<cv::Mat> m_frameDynamicMaskMaps;
    std::vector<cv::Mat> m_frameDynamicMaskMapsX;
    std::vector<cv::Mat> m_backgroundFramesX;
    std::vector<uint8_t> m_backgroundExtraFlags;
    std::vector<uint16_t> m_frameBackgroundIds;
    std::vector<cv::Mat> m_frameBackgroundMasks;
    std::vector<cv::Mat> m_frameBackgroundMasksX;
    std::vector<cv::Mat> m_frameExtraFrames;
    std::vector<uint8_t> m_frameExtraFlags;
    std::vector<uint8_t> m_frameCompMaskIds;
    std::vector<uint8_t> m_frameShapeCompModes;
    std::vector<uint16_t> m_frameRotations;
    std::vector<uint16_t> m_frameRotationsX;

    bool m_drawPointEnabled = false;
    MaskMode m_maskMode = MaskMode::None;
    bool m_frameUndoActive = false;
    bool m_spriteUndoActive = false;
    bool m_spriteDynamicMaskMode = false;
    int m_spriteDynamicSetIndex = 0;
    bool m_spriteDetAreaMode = false;
    int m_spriteDetAreaIndex = 0;
    int m_selectedSpriteSlot = 0;
    int m_selectedSpriteZoneIndex = -1;
    bool m_backgroundUndoActive = false;
    bool m_maskReorderActive = false;
    bool m_dynamicMaskReorderActive = false;
    bool m_showOriginalFrame = true;
    bool m_frameDrawOnMask = false;
    bool m_frameDrawOnZone = false;
    bool m_backgroundMaskMode = false;
    bool m_spriteZoneMode = false;
    FrameHoverArea m_frameHoverArea = FrameHoverArea::None;
    bool m_useHdFrame = false;
    bool m_showBackgroundLayer = true;
    int m_lastBackgroundIndex = -1;
    bool m_useHdBackground = false;
    bool m_previewFilterEnabled = false;
    PreviewFilterKind m_previewFilterKind = PreviewFilterKind::None;
    bool m_previewSelectedOnly = false;
    bool m_previewHdOnly = false;
    bool m_previewMaskOverlayEnabled = false;
    bool m_previewRotateEnabled = false;
    bool m_canvasRotateEnabled = false;
    int m_rotationSetIndex = 0;
    int m_rotationFrameIndex = -1;
    bool m_rotationUseHd = false;
    class QTimer* m_rotationTimer;
    QElapsedTimer m_rotationClock;
    SerumEditorRotationState m_rotationState;
    std::vector<int> m_previewSelectedFrames;
    bool m_restorePreviewSelection = false;
    int m_restorePreviewCurrent = -1;
    std::vector<int> m_restorePreviewSelectionIndices;
    class QAction* m_undoAction;
    class QAction* m_redoAction;
    DrawTool m_drawTool = DrawTool::Point;
    cv::Scalar m_drawColor = cv::Scalar(255, 255, 255, 255);
    QVector<QVector<QColor>> m_fullPalettes;
    QVector<QString> m_paletteNames;
    QVector<QColor> m_paletteColors;
    int m_paletteSetIndex = 0;
    int m_currentPaletteIndex = -1;
    std::vector<uint16_t> m_reducedPaletteIndices;
    std::vector<std::string> m_reducedPaletteNames;
    int m_reducedPaletteIndex = 0;
    int m_reducedSlotIndex = -1;
    int m_dynamicSetIndex = 0;
    int m_dynamicSlotIndex = -1;
    bool m_paletteSelectionIsReference = false;
    bool m_paletteGradientActive = false;
    int m_paletteGradientStartIndex = -1;
    bool m_paletteSetSlotActive = false;
    bool m_reducedSetSlotActive = false;
    bool m_dynamicSetSlotActive = false;
    bool m_rotationSetSlotActive = false;
    struct PaletteUndoState {
        enum class Kind {
            Full,
            Reduced,
            Dynamic
        };
        Kind kind = Kind::Full;
        int palette_index = -1;
        int set_index = -1;
        int frame_index = -1;
        QVector<QColor> full_colors;
        std::vector<uint16_t> values;
    };
    std::vector<PaletteUndoState> m_paletteUndo;
    std::vector<PaletteUndoState> m_paletteRedo;
    bool m_frameHasStart = false;
    bool m_spriteHasStart = false;
    QPoint m_frameStart;
    QPoint m_spriteStart;
    Qt::MouseButton m_frameStartButton = Qt::NoButton;
    Qt::MouseButton m_spriteStartButton = Qt::NoButton;
    uint32_t m_noColors = 64;
};
