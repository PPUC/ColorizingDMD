#include "MainWindow.h"

#include <QAction>
#include <QApplication>
#include <QActionGroup>
#include <QAbstractItemView>
#include <QAbstractItemModel>
#include <QComboBox>
#include <QDockWidget>
#include <QLabel>
#include <QFile>
#include <QFileDialog>
#include <QCheckBox>
#include <QGroupBox>
#include <QPushButton>
#include <QFormLayout>
#include <QLineEdit>
#include <QListWidget>
#include <QListView>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QSettings>
#include <QSpinBox>
#include <QStatusBar>
#include <QStyle>
#include <QTabWidget>
#include <QTimer>
#include <QToolBar>
#include <QToolButton>
#include <QSize>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QWidget>
#include <QFileInfo>
#include <QSignalBlocker>
#include <QStyledItemDelegate>
#include <QPainter>
#include <QPalette>

#include <algorithm>
#include <cmath>
#include <cctype>

#include <opencv2/imgproc.hpp>

#include "CanvasWidget.h"
#include "GLCanvasWidget.h"
#include "ProjectState.h"
#include "ProjectIO.h"
#include "ImageStore.h"
#include "ImageLoader.h"
#include "IndexedImageStore.h"
#include "legacy_project.h"
#include "legacy_project_writer.h"
#include "serum_constants.h"

namespace {
constexpr int kDefaultFrameWidth = 128;
constexpr int kDefaultFrameHeight = 32;
constexpr int kDefaultSpriteWidth = 64;
constexpr int kDefaultSpriteHeight = 64;
constexpr int kPreviewIconWidth = 160;
constexpr int kPreviewIconHeight = 120;
constexpr int kPreviewItemWidth = 180;
constexpr int kPreviewItemHeight = 150;
constexpr int kMaxUndoDepth = 50;
constexpr int kFrameGapPixels = 8;

constexpr int kFrameIndexRole = Qt::UserRole + 1;
constexpr int kFrameDurationRole = Qt::UserRole + 2;

class FramePreviewDelegate : public QStyledItemDelegate
{
public:
    using QStyledItemDelegate::QStyledItemDelegate;

    void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override
    {
        painter->save();

        QStyleOptionViewItem opt = option;
        initStyleOption(&opt, index);
        QStyle* style = opt.widget ? opt.widget->style() : QApplication::style();

        style->drawPrimitive(QStyle::PE_PanelItemViewItem, &opt, painter, opt.widget);

        const QFontMetrics metrics(opt.font);
        const int padding = 6;
        const int textHeight = metrics.height() + 4;
        QRect contentRect = opt.rect.adjusted(padding, padding, -padding, -padding);
        QRect textRect(contentRect.left(), contentRect.top(), contentRect.width(), textHeight);
        QRect iconRect(contentRect.left(),
                       contentRect.top() + textHeight + 2,
                       contentRect.width(),
                       contentRect.height() - textHeight - 2);

        const QVariant frameValue = index.data(kFrameIndexRole);
        if (frameValue.isValid()) {
            const int frameIndex = frameValue.toInt();
            const int duration = index.data(kFrameDurationRole).toInt();
            const QString leftText = QString("F%1").arg(frameIndex);
            const QString rightText = duration > 0 ? QString("%1 ms").arg(duration) : "n/a";

            painter->setPen(opt.palette.text().color());
            painter->drawText(textRect, Qt::AlignLeft | Qt::AlignVCenter, leftText);
            painter->drawText(textRect, Qt::AlignRight | Qt::AlignVCenter, rightText);
        }

        const QIcon icon = qvariant_cast<QIcon>(index.data(Qt::DecorationRole));
        if (!icon.isNull()) {
            const QPixmap pixmap = icon.pixmap(opt.decorationSize);
            const QSize targetSize = pixmap.size().scaled(iconRect.size(), Qt::KeepAspectRatio);
            const QPoint topLeft(iconRect.center().x() - targetSize.width() / 2,
                                 iconRect.center().y() - targetSize.height() / 2);
            painter->drawPixmap(QRect(topLeft, targetSize), pixmap);
        }

        painter->restore();
    }
};

cv::Mat EnsureBgr(const cv::Mat& source)
{
    if (source.empty()) {
        return cv::Mat();
    }
    cv::Mat result;
    if (source.channels() == 1) {
        cv::cvtColor(source, result, cv::COLOR_GRAY2BGR);
    } else if (source.channels() == 4) {
        cv::cvtColor(source, result, cv::COLOR_BGRA2BGR);
    } else {
        result = source.clone();
    }
    return result;
}

cv::Mat MakePlaceholderImage(int width, int height, const cv::Scalar& baseColor, const cv::Scalar& gridColor)
{
    cv::Mat image(height, width, CV_8UC3, baseColor);
    const int grid = 8;
    for (int y = grid; y < height; y += grid) {
        cv::line(image, cv::Point(0, y), cv::Point(width, y), gridColor, 1);
    }
    for (int x = grid; x < width; x += grid) {
        cv::line(image, cv::Point(x, 0), cv::Point(x, height), gridColor, 1);
    }
    cv::rectangle(image, cv::Rect(0, 0, width - 1, height - 1), gridColor, 1);
    return image;
}

const cv::Mat* ResolveSourceImage(const QListWidget* imageList,
                                  const ProjectState* state,
                                  const ImageStore* store)
{
    const QListWidgetItem* currentItem = imageList ? imageList->currentItem() : nullptr;
    if (currentItem && state && state->images().contains(currentItem->text())) {
        if (const auto* entry = store->findByPath(currentItem->text())) {
            return &entry->image;
        }
    }
    if (const auto* latest = store->latest()) {
        return &latest->image;
    }
    return nullptr;
}

cv::Mat MakeFittedImage(const cv::Mat& source, int width, int height)
{
    cv::Mat bgr = EnsureBgr(source);
    if (bgr.empty() || width <= 0 || height <= 0) {
        return cv::Mat();
    }
    cv::Mat output(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    const double scale = std::min(static_cast<double>(width) / bgr.cols,
                                  static_cast<double>(height) / bgr.rows);
    const int targetW = std::max(1, static_cast<int>(bgr.cols * scale));
    const int targetH = std::max(1, static_cast<int>(bgr.rows * scale));
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(targetW, targetH), 0, 0, cv::INTER_AREA);
    const int offsetX = (width - targetW) / 2;
    const int offsetY = (height - targetH) / 2;
    resized.copyTo(output(cv::Rect(offsetX, offsetY, targetW, targetH)));
    return output;
}

bool IsLikelyJsonFile(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        return false;
    }
    while (!file.atEnd()) {
        const char ch = static_cast<char>(file.read(1)[0]);
        if (!std::isspace(static_cast<unsigned char>(ch))) {
            return ch == '{' || ch == '[';
        }
    }
    return false;
}
}

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , m_state(new ProjectState(this))
    , m_imageStore(new ImageStore())
    , m_frameStore(new IndexedImageStore())
    , m_spriteStore(new IndexedImageStore())
{
    setWindowTitle("ColorizingDMD");
    setWindowIcon(QIcon(":/app/app.ico"));
    setMinimumSize(1200, 800);

    auto* fileMenu = menuBar()->addMenu("&File");
    auto* editMenu = menuBar()->addMenu("&Edit");
    auto* viewMenu = menuBar()->addMenu("&View");
    auto* helpMenu = menuBar()->addMenu("&Help");

    auto* newAction = new QAction(QIcon(":/icons/new.png"), "&New", this);
    auto* openAction = new QAction(QIcon(":/icons/open.png"), "&Open...", this);
    auto* saveAction = new QAction(QIcon(":/icons/save.png"), "&Save", this);
    auto* saveAsAction = new QAction("Save &As...", this);
    auto* importImageAction = new QAction(QIcon(":/icons/import.png"), "Import &Image...", this);
    auto* exitAction = new QAction("E&xit", this);
    auto* addFrameAction = new QAction(QIcon(":/icons/add.png"), "Add &Frame", this);
    auto* addSpriteAction = new QAction(QIcon(":/icons/addspr.png"), "Add &Sprite", this);
    auto* removeSelectedAction = new QAction(QIcon(":/icons/remove.png"), "&Remove Selected", this);
    m_undoAction = new QAction("&Undo", this);
    m_redoAction = new QAction("&Redo", this);
    m_undoAction->setShortcut(QKeySequence::Undo);
    m_redoAction->setShortcut(QKeySequence::Redo);
    auto* enableDrawAction = new QAction("Enable &Drawing", this);
    enableDrawAction->setCheckable(true);
    auto* toolPointAction = new QAction("&Point", this);
    toolPointAction->setCheckable(true);
    auto* toolLineAction = new QAction("&Line", this);
    toolLineAction->setCheckable(true);
    auto* toolRectAction = new QAction("&Rectangle", this);
    toolRectAction->setCheckable(true);
    auto* toolRectFillAction = new QAction("Rectangle &Fill", this);
    toolRectFillAction->setCheckable(true);
    auto* toolCircleAction = new QAction("&Circle", this);
    toolCircleAction->setCheckable(true);
    auto* toolCircleFillAction = new QAction("Circle F&ill", this);
    toolCircleFillAction->setCheckable(true);
    auto* toolEllipseAction = new QAction("&Ellipse", this);
    toolEllipseAction->setCheckable(true);
    auto* toolEllipseFillAction = new QAction("Ellipse Fi&ll", this);
    toolEllipseFillAction->setCheckable(true);
    auto* toolColorPickerAction = new QAction("Color &Picker", this);
    toolColorPickerAction->setCheckable(true);
    auto* toolMagicFillAction = new QAction("&Magic Fill", this);
    toolMagicFillAction->setCheckable(true);
    auto* cancelDrawAction = new QAction("&Cancel Draw", this);
    cancelDrawAction->setShortcut(QKeySequence(Qt::Key_Escape));
    auto* fitToViewAction = new QAction("Fit to &View", this);
    removeSelectedAction->setShortcut(QKeySequence::Delete);
    removeSelectedAction->setShortcutContext(Qt::WidgetWithChildrenShortcut);

    fileMenu->addAction(newAction);
    fileMenu->addAction(openAction);
    m_recentMenu = fileMenu->addMenu("Open &Recent");
    fileMenu->addAction(saveAction);
    fileMenu->addAction(saveAsAction);
    fileMenu->addSeparator();
    fileMenu->addAction(importImageAction);
    fileMenu->addSeparator();
    fileMenu->addAction(exitAction);

    editMenu->addAction(m_undoAction);
    editMenu->addAction(m_redoAction);
    editMenu->addSeparator();
    editMenu->addAction(addFrameAction);
    editMenu->addAction(addSpriteAction);
    editMenu->addSeparator();
    editMenu->addAction(removeSelectedAction);
    auto* drawMenu = editMenu->addMenu("&Draw");
    drawMenu->addAction(enableDrawAction);
    drawMenu->addSeparator();
    drawMenu->addAction(toolPointAction);
    drawMenu->addAction(toolLineAction);
    drawMenu->addAction(toolRectAction);
    drawMenu->addAction(toolRectFillAction);
    drawMenu->addAction(toolCircleAction);
    drawMenu->addAction(toolCircleFillAction);
    drawMenu->addAction(toolEllipseAction);
    drawMenu->addAction(toolEllipseFillAction);
    drawMenu->addSeparator();
    drawMenu->addAction(toolColorPickerAction);
    drawMenu->addAction(toolMagicFillAction);
    drawMenu->addSeparator();
    drawMenu->addAction(cancelDrawAction);

    viewMenu->addSeparator();
    viewMenu->addAction(fitToViewAction);

    auto* toolbar = addToolBar("Main");
    toolbar->setIconSize(QSize(22, 22));
    toolbar->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
    toolbar->addAction(newAction);
    toolbar->addAction(openAction);
    toolbar->addAction(saveAction);
    toolbar->addSeparator();
    toolbar->addAction(importImageAction);
    toolbar->addSeparator();
    toolbar->addAction(addFrameAction);
    toolbar->addAction(addSpriteAction);
    toolbar->addSeparator();
    toolbar->addAction(enableDrawAction);

    auto* drawToolbar = addToolBar("Draw");
    drawToolbar->setIconSize(QSize(18, 18));
    drawToolbar->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
    drawToolbar->addAction(toolPointAction);
    drawToolbar->addAction(toolLineAction);
    drawToolbar->addAction(toolRectAction);
    drawToolbar->addAction(toolRectFillAction);
    drawToolbar->addAction(toolCircleAction);
    drawToolbar->addAction(toolCircleFillAction);
    drawToolbar->addAction(toolEllipseAction);
    drawToolbar->addAction(toolEllipseFillAction);
    drawToolbar->addAction(toolColorPickerAction);
    drawToolbar->addAction(toolMagicFillAction);

    addAction(cancelDrawAction);

    auto* status = statusBar();
    status->showMessage("Qt port: UI scaffolding in progress");

    connect(exitAction, &QAction::triggered, qApp, &QApplication::quit);
    connect(m_undoAction, &QAction::triggered, this, [this]() {
        const bool isFrame = isFrameContext();
        if (undoEdit(isFrame)) {
            statusBar()->showMessage("Undo", 1500);
        }
    });
    connect(m_redoAction, &QAction::triggered, this, [this]() {
        const bool isFrame = isFrameContext();
        if (redoEdit(isFrame)) {
            statusBar()->showMessage("Redo", 1500);
        }
    });
    connect(newAction, &QAction::triggered, this, [this]() {
        m_state->newProject();
        m_imageStore->clear();
        m_frameStore->clear();
        m_spriteStore->clear();
        m_frameDurations.clear();
        m_spriteNames.clear();
        m_sectionStarts.clear();
        m_sectionNames.clear();
        m_frameRefs.clear();
        m_frameDynamicColors.clear();
        m_compMasks.clear();
        m_dynamicMasks.clear();
        m_frameCompMaskIds.clear();
        m_frameDynamicMaskIds.clear();
        m_frameShapeCompModes.clear();
        m_noColors = 64;
        resetUndoStacks();
        updateMetadataForFrame(-1);
        updateMetadataForSprite(-1);
        populateBookmarks({}, {});
        statusBar()->showMessage("New project (stub)", 3000);
    });
    connect(openAction, &QAction::triggered, this, [this]() {
        const QString filename = QFileDialog::getOpenFileName(this, "Open Project", QString(), "Serum Projects (*.crom *.cROM *.crp *.cRP);;All Files (*.*)");
        if (!filename.isEmpty()) {
            openProjectFile(filename);
        }
    });
    connect(saveAction, &QAction::triggered, this, [this]() {
        QString filename = m_state->projectPath();
        if (filename.isEmpty()) {
            filename = QFileDialog::getSaveFileName(this, "Save Project", QString(), "Serum Projects (*.crom *.cROM *.crp *.cRP)");
            if (filename.isEmpty()) {
                return;
            }
        }
        if (saveProjectToPath(filename)) {
            statusBar()->showMessage(QString("Save: %1").arg(m_state->projectPath()), 5000);
        } else {
            statusBar()->showMessage("Save failed", 5000);
        }
    });
    connect(saveAsAction, &QAction::triggered, this, [this]() {
        const QString filename = QFileDialog::getSaveFileName(this, "Save Project As", QString(), "Serum Projects (*.crom *.cROM *.crp *.cRP)");
        if (filename.isEmpty()) {
            return;
        }
        if (saveProjectToPath(filename)) {
            statusBar()->showMessage(QString("Save As: %1").arg(m_state->projectPath()), 5000);
        } else {
            statusBar()->showMessage("Save failed", 5000);
        }
    });
    connect(importImageAction, &QAction::triggered, this, [this]() {
        const QString filename = QFileDialog::getOpenFileName(this, "Import Image", QString(), "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*.*)");
        if (!filename.isEmpty()) {
            cv::Mat image;
            if (!LoadImageFile(filename, image)) {
                statusBar()->showMessage(QString("Import failed: %1").arg(filename), 5000);
                return;
            }
            m_imageStore->addImage(filename, image);
            m_state->addImportedImage(filename);
            m_imagesCanvas->setTitle(QString("Imported: %1").arg(filename));
            m_imagesCanvas->setStatusText(QString("Imported %1").arg(filename));
            m_imagesCanvas->setImage(image);
            statusBar()->showMessage(QString("Import image: %1").arg(filename), 5000);
        }
    });
    connect(addFrameAction, &QAction::triggered, this, [this]() {
        cv::Mat frameImage;
        const cv::Mat* sourceImage = ResolveSourceImage(m_imagesList, m_state, m_imageStore);
        if (sourceImage && !sourceImage->empty()) {
            frameImage = MakeFittedImage(*sourceImage, kDefaultFrameWidth, kDefaultFrameHeight);
        }
        if (frameImage.empty()) {
            frameImage = MakePlaceholderImage(kDefaultFrameWidth, kDefaultFrameHeight,
                                              cv::Scalar(18, 18, 18),
                                              cv::Scalar(55, 55, 55));
        }
        m_frameStore->add(frameImage);
        m_state->addFrame();
        ensureUndoStacksSize();
        if (m_framesList->count() > 0) {
            m_framesList->setCurrentRow(m_framesList->count() - 1);
        }
    });
    connect(addSpriteAction, &QAction::triggered, this, [this]() {
        cv::Mat spriteImage;
        const cv::Mat* sourceImage = ResolveSourceImage(m_imagesList, m_state, m_imageStore);
        if (sourceImage && !sourceImage->empty()) {
            spriteImage = MakeFittedImage(*sourceImage, kDefaultSpriteWidth, kDefaultSpriteHeight);
        }
        if (spriteImage.empty()) {
            spriteImage = MakePlaceholderImage(kDefaultSpriteWidth, kDefaultSpriteHeight,
                                               cv::Scalar(24, 24, 24),
                                               cv::Scalar(70, 70, 70));
        }
        m_spriteStore->add(spriteImage);
        m_state->addSprite();
        ensureUndoStacksSize();
        if (m_spritesList->count() > 0) {
            m_spritesList->setCurrentRow(m_spritesList->count() - 1);
        }
    });
    auto* drawToolGroup = new QActionGroup(this);
    drawToolGroup->setExclusive(true);
    drawToolGroup->addAction(toolPointAction);
    drawToolGroup->addAction(toolLineAction);
    drawToolGroup->addAction(toolRectAction);
    drawToolGroup->addAction(toolRectFillAction);
    drawToolGroup->addAction(toolCircleAction);
    drawToolGroup->addAction(toolCircleFillAction);
    drawToolGroup->addAction(toolEllipseAction);
    drawToolGroup->addAction(toolEllipseFillAction);
    drawToolGroup->addAction(toolColorPickerAction);
    drawToolGroup->addAction(toolMagicFillAction);
    toolPointAction->setChecked(true);

    connect(enableDrawAction, &QAction::toggled, this, [this](bool checked) {
        m_drawPointEnabled = checked;
        m_framesCanvas->canvas()->setPanningEnabled(!checked);
        m_spritesCanvas->canvas()->setPanningEnabled(!checked);
        if (!checked) {
            m_framesCanvas->canvas()->clearPreviewImage();
            m_spritesCanvas->canvas()->clearPreviewImage();
            m_frameHasStart = false;
            m_spriteHasStart = false;
            m_frameUndoActive = false;
            m_spriteUndoActive = false;
        }
        statusBar()->showMessage(checked ? "Drawing: enabled" : "Drawing: disabled", 2000);
    });
    connect(toolPointAction, &QAction::triggered, this, [this]() { m_drawTool = DrawTool::Point; });
    connect(toolLineAction, &QAction::triggered, this, [this]() { m_drawTool = DrawTool::Line; });
    connect(toolRectAction, &QAction::triggered, this, [this]() { m_drawTool = DrawTool::Rect; });
    connect(toolRectFillAction, &QAction::triggered, this, [this]() { m_drawTool = DrawTool::RectFill; });
    connect(toolCircleAction, &QAction::triggered, this, [this]() { m_drawTool = DrawTool::Circle; });
    connect(toolCircleFillAction, &QAction::triggered, this, [this]() { m_drawTool = DrawTool::CircleFill; });
    connect(toolEllipseAction, &QAction::triggered, this, [this]() { m_drawTool = DrawTool::Ellipse; });
    connect(toolEllipseFillAction, &QAction::triggered, this, [this]() { m_drawTool = DrawTool::EllipseFill; });
    connect(toolColorPickerAction, &QAction::triggered, this, [this]() { m_drawTool = DrawTool::ColorPicker; });
    connect(toolMagicFillAction, &QAction::triggered, this, [this]() { m_drawTool = DrawTool::MagicFill; });
    connect(cancelDrawAction, &QAction::triggered, this, [this]() { cancelCurrentDraw(); });
    connect(fitToViewAction, &QAction::triggered, this, [this]() {
        m_framesCanvas->canvas()->requestFitOnResize(true);
        m_spritesCanvas->canvas()->requestFitOnResize(true);
        statusBar()->showMessage("Fit to view", 1500);
    });
    connect(removeSelectedAction, &QAction::triggered, this, [this]() {
        if (m_framesList->hasFocus()) {
            const int row = m_framesList->currentRow();
            m_frameStore->removeAt(row);
            m_state->removeFrame(row);
            if (row >= 0 && row < static_cast<int>(m_frameRefs.size())) {
                m_frameRefs.erase(m_frameRefs.begin() + row);
            }
            if (row >= 0 && row < static_cast<int>(m_frameDynamicColors.size())) {
                m_frameDynamicColors.erase(m_frameDynamicColors.begin() + row);
            }
            if (row >= 0 && row < static_cast<int>(m_frameCompMaskIds.size())) {
                m_frameCompMaskIds.erase(m_frameCompMaskIds.begin() + row);
            }
            if (row >= 0 && row < static_cast<int>(m_frameDynamicMaskIds.size())) {
                m_frameDynamicMaskIds.erase(m_frameDynamicMaskIds.begin() + row);
            }
            ensureUndoStacksSize();
        } else if (m_spritesList->hasFocus()) {
            const int row = m_spritesList->currentRow();
            m_spriteStore->removeAt(row);
            m_state->removeSprite(row);
            ensureUndoStacksSize();
        } else if (m_imagesList->hasFocus()) {
            const int row = m_imagesList->currentRow();
            if (row >= 0 && row < m_state->images().size()) {
                const QString path = m_state->images().at(row);
                m_state->removeImage(row);
                m_imageStore->removeByPath(path);
            }
        }
        updateSelectionFromLists();
    });

    auto* tabs = new QTabWidget(this);
    m_canvasTabs = tabs;
    m_framesCanvas = new CanvasWidget("Frames canvas (placeholder)", tabs);
    m_spritesCanvas = new CanvasWidget("Sprites canvas (placeholder)", tabs);
    m_imagesCanvas = new CanvasWidget("Images canvas (placeholder)", tabs);
    tabs->addTab(m_framesCanvas, "Frames");
    tabs->addTab(m_spritesCanvas, "Sprites");
    tabs->addTab(m_imagesCanvas, "Images");
    setCentralWidget(tabs);
    connect(tabs, &QTabWidget::currentChanged, this, [this](int) {
        updateUndoActions();
    });

    auto* toolDock = new QDockWidget("Tools", this);
    toolDock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    auto* toolsTabs = new QTabWidget(toolDock);
    auto* framesTab = new QWidget(toolsTabs);
    auto* spritesTab = new QWidget(toolsTabs);
    auto* imagesTab = new QWidget(toolsTabs);
    auto* masksTab = new QWidget(toolsTabs);
    auto* dynamicMasksTab = new QWidget(toolsTabs);

    m_frameFilter = new QLineEdit(framesTab);
    m_frameFilter->setPlaceholderText("Filter frames...");
    m_framesList = new QListWidget(framesTab);
    auto* framesLayout = new QVBoxLayout(framesTab);
    framesLayout->addWidget(m_frameFilter);
    framesLayout->addWidget(m_framesList, 1);
    framesTab->setLayout(framesLayout);

    m_spriteFilter = new QLineEdit(spritesTab);
    m_spriteFilter->setPlaceholderText("Filter sprites...");
    m_spritesList = new QListWidget(spritesTab);
    auto* spritesLayout = new QVBoxLayout(spritesTab);
    spritesLayout->addWidget(m_spriteFilter);
    spritesLayout->addWidget(m_spritesList, 1);
    spritesTab->setLayout(spritesLayout);

    m_imagesList = new QListWidget(imagesTab);
    auto* imagesLayout = new QVBoxLayout(imagesTab);
    imagesLayout->addWidget(m_imagesList, 1);
    imagesTab->setLayout(imagesLayout);

    auto* masksLayout = new QVBoxLayout(masksTab);
    m_maskList = new QListWidget(masksTab);
    m_maskList->setSelectionMode(QAbstractItemView::SingleSelection);
    m_maskList->setViewMode(QListView::IconMode);
    m_maskList->setFlow(QListView::TopToBottom);
    m_maskList->setWrapping(false);
    m_maskList->setMovement(QListView::Snap);
    m_maskList->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_maskList->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    m_maskList->setResizeMode(QListView::Adjust);
    m_maskList->setDragDropMode(QAbstractItemView::DragOnly);
    m_maskList->setDragDropOverwriteMode(false);
    m_maskList->setDragEnabled(true);
    m_maskList->setAcceptDrops(false);
    m_maskList->setDropIndicatorShown(true);
    m_maskList->setIconSize(QSize(kPreviewIconWidth, kPreviewIconHeight));
    m_maskList->setGridSize(QSize(kPreviewItemWidth, kPreviewItemHeight));
    m_maskList->setSpacing(6);
    m_maskMoveUp = new QToolButton(masksTab);
    m_maskMoveUp->setText("Up");
    m_maskMoveDown = new QToolButton(masksTab);
    m_maskMoveDown->setText("Down");
    m_maskClearButton = new QPushButton("Clear mask", masksTab);
    auto* compButtons = new QHBoxLayout();
    compButtons->addWidget(m_maskMoveUp);
    compButtons->addWidget(m_maskMoveDown);
    compButtons->addStretch(1);
    compButtons->addWidget(m_maskClearButton);
    masksLayout->addWidget(m_maskList, 1);
    masksLayout->addLayout(compButtons);
    masksTab->setLayout(masksLayout);

    auto* dynLayout = new QVBoxLayout(dynamicMasksTab);
    m_dynamicMaskList = new QListWidget(dynamicMasksTab);
    m_dynamicMaskList->setSelectionMode(QAbstractItemView::SingleSelection);
    m_dynamicMaskList->setViewMode(QListView::IconMode);
    m_dynamicMaskList->setFlow(QListView::TopToBottom);
    m_dynamicMaskList->setWrapping(false);
    m_dynamicMaskList->setMovement(QListView::Snap);
    m_dynamicMaskList->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_dynamicMaskList->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    m_dynamicMaskList->setResizeMode(QListView::Adjust);
    m_dynamicMaskList->setDragDropMode(QAbstractItemView::DragOnly);
    m_dynamicMaskList->setDragDropOverwriteMode(false);
    m_dynamicMaskList->setDragEnabled(true);
    m_dynamicMaskList->setAcceptDrops(false);
    m_dynamicMaskList->setDropIndicatorShown(true);
    m_dynamicMaskList->setIconSize(QSize(kPreviewIconWidth, kPreviewIconHeight));
    m_dynamicMaskList->setGridSize(QSize(kPreviewItemWidth, kPreviewItemHeight));
    m_dynamicMaskList->setSpacing(6);
    m_dynamicMaskMoveUp = new QToolButton(dynamicMasksTab);
    m_dynamicMaskMoveUp->setText("Up");
    m_dynamicMaskMoveDown = new QToolButton(dynamicMasksTab);
    m_dynamicMaskMoveDown->setText("Down");
    m_dynamicMaskClearButton = new QPushButton("Clear mask", dynamicMasksTab);
    auto* dynButtons = new QHBoxLayout();
    dynButtons->addWidget(m_dynamicMaskMoveUp);
    dynButtons->addWidget(m_dynamicMaskMoveDown);
    dynButtons->addStretch(1);
    dynButtons->addWidget(m_dynamicMaskClearButton);
    dynLayout->addWidget(m_dynamicMaskList, 1);
    dynLayout->addLayout(dynButtons);
    dynamicMasksTab->setLayout(dynLayout);

    toolsTabs->addTab(framesTab, "Frames");
    toolsTabs->addTab(spritesTab, "Sprites");
    toolsTabs->addTab(imagesTab, "Images");
    toolsTabs->addTab(masksTab, "Masks");
    toolsTabs->addTab(dynamicMasksTab, "Dynamic Masks");
    toolDock->setWidget(toolsTabs);
    addDockWidget(Qt::LeftDockWidgetArea, toolDock);

    auto* inspectorDock = new QDockWidget("Inspector", this);
    inspectorDock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    auto* inspectorWidget = new QWidget(inspectorDock);
    auto* inspectorLayout = new QFormLayout(inspectorWidget);
    m_projectLabel = new QLabel("None", inspectorWidget);
    m_bookmarksCombo = new QComboBox(inspectorWidget);
    m_bookmarksCombo->setEnabled(false);
    m_frameJump = new QSpinBox(inspectorWidget);
    m_frameJump->setEnabled(false);
    m_frameJump->setPrefix("Frame ");
    m_countsLabel = new QLabel("Frames: 0, Sprites: 0", inspectorWidget);
    m_selectionLabel = new QLabel("None", inspectorWidget);
    m_frameMetaLabel = new QLabel("-", inspectorWidget);
    m_spriteMetaLabel = new QLabel("-", inspectorWidget);
    m_frameMaskAssign = new QComboBox(inspectorWidget);
    m_frameDynamicMaskAssign = new QComboBox(inspectorWidget);
    m_compMaskEditToggle = new QCheckBox("Edit mask", inspectorWidget);
    m_compMaskPreviewToggle = new QCheckBox("Preview mask", inspectorWidget);
    m_dynMaskEditToggle = new QCheckBox("Edit dynamic", inspectorWidget);
    m_dynMaskPreviewToggle = new QCheckBox("Preview dynamic", inspectorWidget);
    m_shapeCompToggle = new QCheckBox("Shape comparison", inspectorWidget);
    inspectorLayout->addRow("Project", m_projectLabel);
    inspectorLayout->addRow("Bookmarks", m_bookmarksCombo);
    inspectorLayout->addRow("Go to frame", m_frameJump);
    inspectorLayout->addRow("Counts", m_countsLabel);
    inspectorLayout->addRow("Selection", m_selectionLabel);
    inspectorLayout->addRow("Frame info", m_frameMetaLabel);
    inspectorLayout->addRow("Sprite info", m_spriteMetaLabel);
    inspectorLayout->addRow("Mask", m_frameMaskAssign);
    inspectorLayout->addRow("Dynamic mask", m_frameDynamicMaskAssign);
    inspectorLayout->addRow("Mask edit", m_compMaskEditToggle);
    inspectorLayout->addRow("Mask preview", m_compMaskPreviewToggle);
    inspectorLayout->addRow("Dynamic edit", m_dynMaskEditToggle);
    inspectorLayout->addRow("Dynamic preview", m_dynMaskPreviewToggle);
    inspectorLayout->addRow("Shape compare", m_shapeCompToggle);
    inspectorWidget->setLayout(inspectorLayout);
    inspectorDock->setWidget(inspectorWidget);
    addDockWidget(Qt::RightDockWidgetArea, inspectorDock);

    auto* previewDock = new QDockWidget("Frame Preview", this);
    previewDock->setAllowedAreas(Qt::BottomDockWidgetArea);
    auto* previewWidget = new QWidget(previewDock);
    auto* previewLayout = new QVBoxLayout(previewWidget);
    previewLayout->setContentsMargins(6, 6, 6, 6);
    previewLayout->setSpacing(6);

    auto* previewBar = new QHBoxLayout();
    auto* previewLabel = new QLabel("Preview", previewWidget);
    previewBar->addWidget(previewLabel);
    previewBar->addStretch(1);
    previewLayout->addLayout(previewBar);

    m_framePreviewList = new QListWidget(previewWidget);
    m_framePreviewList->setViewMode(QListView::IconMode);
    m_framePreviewList->setFlow(QListView::LeftToRight);
    m_framePreviewList->setWrapping(false);
    m_framePreviewList->setMovement(QListView::Static);
    m_framePreviewList->setSelectionMode(QAbstractItemView::SingleSelection);
    m_framePreviewList->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    m_framePreviewList->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_framePreviewList->setResizeMode(QListView::Adjust);
    m_framePreviewList->setIconSize(QSize(kPreviewIconWidth, kPreviewIconHeight));
    m_framePreviewList->setGridSize(QSize(kPreviewItemWidth, kPreviewItemHeight));
    m_framePreviewList->setSpacing(6);
    m_framePreviewList->setItemDelegate(new FramePreviewDelegate(m_framePreviewList));
    previewLayout->addWidget(m_framePreviewList);

    previewWidget->setLayout(previewLayout);
    previewDock->setWidget(previewWidget);
    addDockWidget(Qt::BottomDockWidgetArea, previewDock);

    viewMenu->addAction(toolDock->toggleViewAction());
    viewMenu->addAction(inspectorDock->toggleViewAction());
    viewMenu->addAction(previewDock->toggleViewAction());

    auto* aboutAction = new QAction("&About", this);
    helpMenu->addAction(aboutAction);
    connect(aboutAction, &QAction::triggered, this, [this]() {
        QMessageBox::information(this, "About ColorizingDMD", "ColorizingDMD Qt port (UI scaffolding).");
    });

    connect(m_state, &ProjectState::projectPathChanged, this, [this](const QString& path) {
        updateWindowTitle();
        m_projectLabel->setText(path.isEmpty() ? "None" : path);
    });
    connect(m_state, &ProjectState::recentFilesChanged, this, [this]() {
        refreshRecentMenu();
        persistRecentFiles();
    });
    connect(m_state, &ProjectState::imagesChanged, this, [this]() {
        refreshImageList();
    });
    connect(m_state, &ProjectState::framesChanged, this, [this]() {
        refreshFrameSpriteLists();
    });
    connect(m_state, &ProjectState::spritesChanged, this, [this]() {
        refreshFrameSpriteLists();
    });
    connect(m_state, &ProjectState::countsChanged, this, [this]() {
        refreshCounts();
        refreshFrameSpriteLists();
        updateFrameJumpRange();
    });
    connect(m_compMaskEditToggle, &QCheckBox::toggled, this, [this](bool enabled) {
        m_compMaskEditEnabled = enabled;
        if (enabled) {
            if (!m_compMaskPreviewToggle->isChecked()) {
                QSignalBlocker blocker(m_compMaskPreviewToggle);
                m_compMaskPreviewToggle->setChecked(true);
                m_compMaskPreviewEnabled = true;
            }
            if (m_dynMaskEditToggle->isChecked()) {
                QSignalBlocker blocker(m_dynMaskEditToggle);
                m_dynMaskEditToggle->setChecked(false);
                m_dynMaskEditEnabled = false;
            }
            if (m_dynMaskPreviewToggle->isChecked()) {
                QSignalBlocker blocker(m_dynMaskPreviewToggle);
                m_dynMaskPreviewToggle->setChecked(false);
                m_dynMaskPreviewEnabled = false;
            }
        }
        updateMaskPreviewForFrame(m_framesList->currentRow());
    });
    connect(m_compMaskPreviewToggle, &QCheckBox::toggled, this, [this](bool enabled) {
        m_compMaskPreviewEnabled = enabled;
        if (enabled && m_dynMaskPreviewToggle->isChecked()) {
            QSignalBlocker blocker(m_dynMaskPreviewToggle);
            m_dynMaskPreviewToggle->setChecked(false);
            m_dynMaskPreviewEnabled = false;
        }
        updateMaskPreviewForFrame(m_framesList->currentRow());
    });
    connect(m_dynMaskEditToggle, &QCheckBox::toggled, this, [this](bool enabled) {
        m_dynMaskEditEnabled = enabled;
        if (enabled) {
            if (!m_dynMaskPreviewToggle->isChecked()) {
                QSignalBlocker blocker(m_dynMaskPreviewToggle);
                m_dynMaskPreviewToggle->setChecked(true);
                m_dynMaskPreviewEnabled = true;
            }
            if (m_compMaskEditToggle->isChecked()) {
                QSignalBlocker blocker(m_compMaskEditToggle);
                m_compMaskEditToggle->setChecked(false);
                m_compMaskEditEnabled = false;
            }
            if (m_compMaskPreviewToggle->isChecked()) {
                QSignalBlocker blocker(m_compMaskPreviewToggle);
                m_compMaskPreviewToggle->setChecked(false);
                m_compMaskPreviewEnabled = false;
            }
        }
        updateMaskPreviewForFrame(m_framesList->currentRow());
    });
    connect(m_dynMaskPreviewToggle, &QCheckBox::toggled, this, [this](bool enabled) {
        m_dynMaskPreviewEnabled = enabled;
        if (enabled && m_compMaskPreviewToggle->isChecked()) {
            QSignalBlocker blocker(m_compMaskPreviewToggle);
            m_compMaskPreviewToggle->setChecked(false);
            m_compMaskPreviewEnabled = false;
        }
        updateMaskPreviewForFrame(m_framesList->currentRow());
    });
    connect(m_shapeCompToggle, &QCheckBox::toggled, this, [this](bool enabled) {
        const int row = m_framesList ? m_framesList->currentRow() : -1;
        if (row < 0 || row >= static_cast<int>(m_frameShapeCompModes.size())) {
            return;
        }
        m_frameShapeCompModes[static_cast<std::size_t>(row)] = enabled ? 1 : 0;
    });
    connect(m_frameMaskAssign, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
        setCurrentFrameMaskId(m_frameMaskAssign->currentData().toInt());
        updateMaskPreviewForFrame(m_framesList->currentRow());
    });
    connect(m_frameDynamicMaskAssign, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
        setCurrentFrameDynamicMaskId(m_frameDynamicMaskAssign->currentData().toInt());
        updateMaskPreviewForFrame(m_framesList->currentRow());
    });
    connect(m_maskList, &QListWidget::currentRowChanged, this, [this](int) {
        updateMaskPreviewForFrame(m_framesList->currentRow());
    });
    connect(m_dynamicMaskList, &QListWidget::currentRowChanged, this, [this](int) {
        updateMaskPreviewForFrame(m_framesList->currentRow());
    });
    connect(m_maskMoveUp, &QToolButton::clicked, this, [this]() {
        const int index = m_maskList->currentRow();
        if (index > 0) {
            QListWidgetItem* item = m_maskList->takeItem(index);
            m_maskList->insertItem(index - 1, item);
            m_maskList->setCurrentRow(index - 1);
            applyMaskListOrder();
        }
    });
    connect(m_maskMoveDown, &QToolButton::clicked, this, [this]() {
        const int index = m_maskList->currentRow();
        if (index >= 0 && index + 1 < m_maskList->count()) {
            QListWidgetItem* item = m_maskList->takeItem(index);
            m_maskList->insertItem(index + 1, item);
            m_maskList->setCurrentRow(index + 1);
            applyMaskListOrder();
        }
    });
    connect(m_dynamicMaskMoveUp, &QToolButton::clicked, this, [this]() {
        const int index = m_dynamicMaskList->currentRow();
        if (index > 0) {
            QListWidgetItem* item = m_dynamicMaskList->takeItem(index);
            m_dynamicMaskList->insertItem(index - 1, item);
            m_dynamicMaskList->setCurrentRow(index - 1);
            applyDynamicMaskListOrder();
        }
    });
    connect(m_dynamicMaskMoveDown, &QToolButton::clicked, this, [this]() {
        const int index = m_dynamicMaskList->currentRow();
        if (index >= 0 && index + 1 < m_dynamicMaskList->count()) {
            QListWidgetItem* item = m_dynamicMaskList->takeItem(index);
            m_dynamicMaskList->insertItem(index + 1, item);
            m_dynamicMaskList->setCurrentRow(index + 1);
            applyDynamicMaskListOrder();
        }
    });
    connect(m_maskClearButton, &QPushButton::clicked, this, [this]() {
        const int maskId = m_maskList->currentRow();
        if (cv::Mat* mask = activeComparisonMask()) {
            const int frameIndex = m_framesList ? m_framesList->currentRow() : -1;
            if (frameIndex >= 0 && frameIndex < static_cast<int>(m_frameUndoStacks.size())) {
                UndoState state;
                if (const cv::Mat* image = m_frameStore->at(frameIndex)) {
                    state.image = image->clone();
                }
                state.mask = mask->clone();
                state.mask_kind = MaskKind::Comparison;
                state.mask_index = maskId;
                m_frameUndoStacks[static_cast<std::size_t>(frameIndex)].undo.push_back(std::move(state));
                m_frameUndoStacks[static_cast<std::size_t>(frameIndex)].redo.clear();
                updateUndoActions();
            }
            mask->setTo(cv::Scalar(0));
            updateMaskPreviewIcons();
            updateMaskPreviewForFrame(m_framesList->currentRow());
        }
    });
    connect(m_dynamicMaskClearButton, &QPushButton::clicked, this, [this]() {
        const int maskId = m_dynamicMaskList->currentRow();
        if (cv::Mat* mask = activeDynamicMask()) {
            const int frameIndex = m_framesList ? m_framesList->currentRow() : -1;
            if (frameIndex >= 0 && frameIndex < static_cast<int>(m_frameUndoStacks.size())) {
                UndoState state;
                if (const cv::Mat* image = m_frameStore->at(frameIndex)) {
                    state.image = image->clone();
                }
                state.mask = mask->clone();
                state.mask_kind = MaskKind::Dynamic;
                state.mask_index = maskId;
                m_frameUndoStacks[static_cast<std::size_t>(frameIndex)].undo.push_back(std::move(state));
                m_frameUndoStacks[static_cast<std::size_t>(frameIndex)].redo.clear();
                updateUndoActions();
            }
            mask->setTo(cv::Scalar(0));
            updateDynamicMaskPreviewIcons();
            updateMaskPreviewForFrame(m_framesList->currentRow());
        }
    });

    connect(m_bookmarksCombo, &QComboBox::currentIndexChanged, this, [this](int index) {
        if (index < 0) {
            return;
        }
        const QVariant value = m_bookmarksCombo->currentData();
        if (!value.isValid()) {
            return;
        }
        const int frameIndex = value.toInt();
        if (frameIndex >= 0 && frameIndex < m_framesList->count()) {
            m_framesList->setCurrentRow(frameIndex);
        }
    });

    connect(m_framePreviewList, &QListWidget::currentRowChanged, this, [this](int row) {
        if (row < 0 || row >= m_framesList->count()) {
            return;
        }
        QSignalBlocker blocker(m_framesList);
        m_framesList->setCurrentRow(row);
        showFrameAtIndex(row);
        updateMetadataForFrame(row);
        updateUndoActions();
    });

    // Drag-reorder disabled; up/down buttons apply ordering changes.

    connect(m_framesList, &QListWidget::currentTextChanged, this, [this](const QString& text) {
        if (!text.isEmpty()) {
            setInspectorSelection(QString("Frame: %1").arg(text));
            m_framesCanvas->setTitle(QString("Frame canvas - %1").arg(text));
            m_framesCanvas->setStatusText(QString("Selected %1").arg(text));
            showFrameAtIndex(m_framesList->currentRow());
            {
                QSignalBlocker blocker(m_framePreviewList);
                m_framePreviewList->setCurrentRow(m_framesList->currentRow());
            }
            updateUndoActions();
            if (m_framesList->currentRow() >= 0) {
                QSignalBlocker blocker(m_frameJump);
                m_frameJump->setValue(m_framesList->currentRow());
            }
            updateMetadataForFrame(m_framesList->currentRow());
        } else {
            updateSelectionFromLists();
        }
    });
    connect(m_spritesList, &QListWidget::currentTextChanged, this, [this](const QString& text) {
        if (!text.isEmpty()) {
            setInspectorSelection(QString("Sprite: %1").arg(text));
            m_spritesCanvas->setTitle(QString("Sprite canvas - %1").arg(text));
            m_spritesCanvas->setStatusText(QString("Selected %1").arg(text));
            showSpriteAtIndex(m_spritesList->currentRow());
            updateUndoActions();
            updateMetadataForSprite(m_spritesList->currentRow());
        } else {
            updateSelectionFromLists();
        }
    });
    connect(m_imagesList, &QListWidget::currentTextChanged, this, [this](const QString& text) {
        if (!text.isEmpty()) {
            setInspectorSelection(QString("Image: %1").arg(text));
            m_imagesCanvas->setTitle(QString("Image canvas - %1").arg(text));
            m_imagesCanvas->setStatusText(QString("Selected %1").arg(text));
            if (m_state->images().contains(text)) {
                showImageForPath(text);
            }
        } else {
            updateSelectionFromLists();
        }
    });

    connect(m_framesCanvas->canvas(), &GLCanvasWidget::imageClicked, this, [this](int x, int y, Qt::MouseButton button) {
        handleToolPress(true, x, y, button);
    });
    connect(m_framesCanvas->canvas(), &GLCanvasWidget::imageDragged, this, [this](int x, int y, Qt::MouseButtons buttons) {
        handleToolDrag(true, x, y, buttons);
    });
    connect(m_framesCanvas->canvas(), &GLCanvasWidget::imageReleased, this, [this](int x, int y, Qt::MouseButton button) {
        handleToolRelease(true, x, y, button);
    });
    connect(m_framesCanvas->canvas(), &GLCanvasWidget::maskDropped, this, [this](const QString& kind, int index) {
        if (m_framesList->currentRow() < 0) {
            return;
        }
        ensureMaskDataSize();
        if (kind == "mask") {
            if (index < 0 || index >= MAX_MASKS) {
                return;
            }
            m_frameMaskAssign->setCurrentIndex(index + 1);
            if (m_maskList) {
                m_maskList->setCurrentRow(index);
            }
            statusBar()->showMessage(QString("Assigned mask %1").arg(index), 2000);
        } else if (kind == "dynamic") {
            if (index < 0 || index >= MAX_DYNA_SETS_PER_FRAMEN) {
                return;
            }
            m_frameDynamicMaskAssign->setCurrentIndex(index + 1);
            if (m_dynamicMaskList) {
                m_dynamicMaskList->setCurrentRow(index);
            }
            statusBar()->showMessage(QString("Assigned dynamic mask %1").arg(index), 2000);
        }
        updateMaskPreviewForFrame(m_framesList->currentRow());
    });
    connect(m_spritesCanvas->canvas(), &GLCanvasWidget::imageClicked, this, [this](int x, int y, Qt::MouseButton button) {
        handleToolPress(false, x, y, button);
    });
    connect(m_spritesCanvas->canvas(), &GLCanvasWidget::imageDragged, this, [this](int x, int y, Qt::MouseButtons buttons) {
        handleToolDrag(false, x, y, buttons);
    });
    connect(m_spritesCanvas->canvas(), &GLCanvasWidget::imageReleased, this, [this](int x, int y, Qt::MouseButton button) {
        handleToolRelease(false, x, y, button);
    });

    connect(m_framesCanvas, &CanvasWidget::fitRequested, this, [this]() {
        m_framesCanvas->canvas()->requestFitOnResize(true);
    });
    connect(m_spritesCanvas, &CanvasWidget::fitRequested, this, [this]() {
        m_spritesCanvas->canvas()->requestFitOnResize(true);
    });
    connect(m_framesCanvas, &CanvasWidget::gridToggled, this, [this](bool enabled) {
        m_framesCanvas->canvas()->setGridEnabled(enabled);
    });
    connect(m_spritesCanvas, &CanvasWidget::gridToggled, this, [this](bool enabled) {
        m_spritesCanvas->canvas()->setGridEnabled(enabled);
    });
    connect(m_framesCanvas, &CanvasWidget::originalToggled, this, [this](bool enabled) {
        m_showOriginalFrame = enabled;
        updateFrameCanvasImage(m_framesList->currentRow());
        updateMaskPreviewForFrame(m_framesList->currentRow());
    });

    connect(m_frameFilter, &QLineEdit::textChanged, this, [this](const QString& text) {
        applyFrameFilter(text);
    });
    connect(m_spriteFilter, &QLineEdit::textChanged, this, [this](const QString& text) {
        applySpriteFilter(text);
    });
    connect(m_frameJump, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (value >= 0 && value < m_framesList->count()) {
            m_framesList->setCurrentRow(value);
        }
    });

    refreshRecentMenu();
    refreshImageList();
    refreshCounts();
    refreshFrameSpriteLists();
    updateUndoActions();

    QSettings settings("PPUC", "ColorizingDMD");
    const QStringList recent = settings.value("recentFiles").toStringList();
    if (!recent.isEmpty()) {
        m_state->setRecentFiles(recent);
    }
}

void MainWindow::updateWindowTitle()
{
    if (m_state->projectPath().isEmpty()) {
        setWindowTitle("ColorizingDMD");
        return;
    }
    setWindowTitle(QString("ColorizingDMD - %1").arg(m_state->projectPath()));
}

void MainWindow::persistRecentFiles()
{
    QSettings settings("PPUC", "ColorizingDMD");
    settings.setValue("recentFiles", m_state->recentFiles());
    settings.sync();
}

void MainWindow::openProjectFile(const QString& filename)
{
    if (filename.isEmpty()) {
        return;
    }
    const QFileInfo info(filename);
    const QString suffix = info.suffix().toLower();
    if (suffix == "crom" || suffix == "crp") {
        if (suffix == "crom" && IsLikelyJsonFile(filename)) {
            QString error;
            m_imageStore->clear();
            m_frameStore->clear();
            m_spriteStore->clear();
            m_frameDurations.clear();
            m_spriteNames.clear();
            m_sectionStarts.clear();
            m_sectionNames.clear();
            m_frameRefs.clear();
            m_frameDynamicColors.clear();
            m_compMasks.clear();
            m_dynamicMasks.clear();
            m_frameCompMaskIds.clear();
            m_frameDynamicMaskIds.clear();
            m_frameShapeCompModes.clear();
            m_noColors = 64;
            updateMetadataForFrame(-1);
            updateMetadataForSprite(-1);
            populateBookmarks({}, {});
            if (LoadProjectJson(*m_state, filename, &error)) {
                resetUndoStacks();
                statusBar()->showMessage(QString("Open: %1").arg(filename), 5000);
                persistRecentFiles();
            } else {
                statusBar()->showMessage(QString("Open failed: %1").arg(error), 5000);
            }
            return;
        }
        QString cromPath = filename;
        QString rpPath;
        if (suffix == "crp") {
            const QString base = info.completeBaseName();
            const QString dir = info.absolutePath();
            const QString candidate = dir + "/" + base + ".crom";
            if (QFileInfo::exists(candidate)) {
                cromPath = candidate;
            } else {
                const QString candidateUpper = dir + "/" + base + ".cROM";
                if (QFileInfo::exists(candidateUpper)) {
                    cromPath = candidateUpper;
                }
            }
            rpPath = filename;
        } else {
            const QString base = info.completeBaseName();
            const QString dir = info.absolutePath();
            const QString candidate = dir + "/" + base + ".cRP";
            if (QFileInfo::exists(candidate)) {
                rpPath = candidate;
            }
        }
        LegacyProject legacy;
        std::string error;
        if (!LoadLegacyProject(cromPath.toStdString(), rpPath.toStdString(), legacy, &error)) {
            statusBar()->showMessage(QString("Open failed: %1").arg(QString::fromStdString(error)), 5000);
            return;
        }

        m_imageStore->clear();
        m_frameStore->clear();
        m_spriteStore->clear();
        m_frameDurations.clear();
        m_spriteNames.clear();
        m_sectionStarts.clear();
        m_sectionNames.clear();
        m_frameRefs.clear();
        m_frameDynamicColors.clear();
        m_compMasks.clear();
        m_dynamicMasks.clear();
        m_frameCompMaskIds.clear();
        m_frameDynamicMaskIds.clear();
        m_frameShapeCompModes.clear();
        m_noColors = legacy.no_colors > 0 ? legacy.no_colors : 64;
        updateMetadataForFrame(-1);
        updateMetadataForSprite(-1);
        for (const auto& frame : legacy.frames) {
            m_frameStore->add(frame);
        }
        for (const auto& sprite : legacy.sprites) {
            m_spriteStore->add(sprite);
        }
        m_frameDurations = legacy.frame_durations;
        m_spriteNames = legacy.sprite_labels;
        m_sectionStarts = legacy.section_firsts;
        m_sectionNames = legacy.section_names;
        m_frameRefs = legacy.frame_refs;
        m_frameDynamicColors = legacy.frame_dynamic_colors;
        m_compMasks = legacy.comp_masks;
        m_dynamicMasks = legacy.dynamic_masks;
        m_frameCompMaskIds = legacy.frame_comp_mask_ids;
        m_frameDynamicMaskIds = legacy.frame_dynamic_mask_ids;
        m_frameShapeCompModes = legacy.frame_shape_comp_modes;

        QStringList frames;
        for (int i = 0; i < static_cast<int>(legacy.frames.size()); ++i) {
            frames.append(QString("Frame %1").arg(i));
        }
        QStringList sprites;
        if (!legacy.sprite_labels.empty()) {
            for (const auto& label : legacy.sprite_labels) {
                sprites.append(QString::fromStdString(label));
            }
        } else {
            for (int i = 0; i < static_cast<int>(legacy.sprites.size()); ++i) {
                sprites.append(QString("Sprite %1").arg(i));
            }
        }

        {
            QSignalBlocker blocker(m_state);
            m_state->newProject();
            m_state->openProject(cromPath);
            m_state->setFramesAndSprites(frames, sprites);
        }
        resetUndoStacks();
        updateWindowTitle();
        m_projectLabel->setText(cromPath);
        refreshRecentMenu();
        refreshImageList();
        refreshCounts();
        refreshFrameSpriteLists();
        if (!legacy.frames.empty()) {
            m_framesList->setCurrentRow(0);
            QTimer::singleShot(0, this, [this]() {
                m_framesCanvas->canvas()->requestFitOnResize(true);
            });
        }
        populateBookmarks(legacy.section_firsts, legacy.section_names);
        statusBar()->showMessage(QString("Open legacy: %1").arg(cromPath), 5000);
        m_state->setRecentFiles(m_state->recentFiles());
        persistRecentFiles();
        return;
    }

    QString error;
    m_imageStore->clear();
    m_frameStore->clear();
    m_spriteStore->clear();
    m_frameDurations.clear();
    m_spriteNames.clear();
    m_sectionStarts.clear();
    m_sectionNames.clear();
    m_frameRefs.clear();
    m_frameDynamicColors.clear();
    m_compMasks.clear();
    m_dynamicMasks.clear();
    m_frameCompMaskIds.clear();
    m_frameDynamicMaskIds.clear();
    m_frameShapeCompModes.clear();
    m_noColors = 64;
    updateMetadataForFrame(-1);
    updateMetadataForSprite(-1);
    populateBookmarks({}, {});
    if (LoadProjectJson(*m_state, filename, &error)) {
        statusBar()->showMessage(QString("Open: %1").arg(filename), 5000);
        persistRecentFiles();
    } else {
        statusBar()->showMessage(QString("Open failed: %1").arg(error), 5000);
    }
}

bool MainWindow::saveProjectToPath(const QString& filename)
{
    if (filename.isEmpty()) {
        return false;
    }
    QFileInfo info(filename);
    QString suffix = info.suffix().toLower();
    QString target = filename;
    if (suffix.isEmpty()) {
        target = filename + ".crom";
        suffix = "crom";
    }
    if (suffix == "crom" || suffix == "crp") {
        return saveLegacyProject(target);
    }
    if (SaveProjectJson(*m_state, target)) {
        m_state->saveProject(target);
        return true;
    }
    return false;
}

bool MainWindow::saveLegacyProject(const QString& filename)
{
    if (m_frameStore->count() == 0) {
        statusBar()->showMessage("Save failed: no frames", 5000);
        return false;
    }
    ensureMaskDataSize();
    QFileInfo info(filename);
    const QString suffix = info.suffix().toLower();
    const QString dir = info.absolutePath();
    const QString base = info.completeBaseName();
    QString cromPath = filename;
    QString rpPath = dir + "/" + base + ".cRP";
    if (suffix == "crp") {
        rpPath = filename;
        cromPath = dir + "/" + base + ".crom";
    }

    LegacyProject project = buildLegacyProject(base);
    std::string error;
    if (!SaveLegacyProject(cromPath.toStdString(), rpPath.toStdString(), project, &error)) {
        statusBar()->showMessage(QString("Save failed: %1").arg(QString::fromStdString(error)), 5000);
        return false;
    }

    m_state->saveProject(cromPath);
    return true;
}

LegacyProject MainWindow::buildLegacyProject(const QString& baseName) const
{
    LegacyProject project;
    project.name = baseName.toStdString();

    const int frameCount = m_frameStore->count();
    project.frames.reserve(static_cast<std::size_t>(frameCount));
    for (int i = 0; i < frameCount; ++i) {
        if (const cv::Mat* frame = m_frameStore->at(i)) {
            project.frames.push_back(frame->clone());
        }
    }

    const int spriteCount = m_spriteStore->count();
    project.sprites.reserve(static_cast<std::size_t>(spriteCount));
    for (int i = 0; i < spriteCount; ++i) {
        if (const cv::Mat* sprite = m_spriteStore->at(i)) {
            project.sprites.push_back(sprite->clone());
        }
    }

    project.frame_durations.assign(static_cast<std::size_t>(frameCount), 30);
    for (int i = 0; i < frameCount && i < static_cast<int>(m_frameDurations.size()); ++i) {
        project.frame_durations[i] = m_frameDurations[i];
    }

    project.section_firsts = m_sectionStarts;
    project.section_names = m_sectionNames;
    project.frame_refs = m_frameRefs;
    project.no_colors = m_noColors > 0 ? m_noColors : 64;
    project.comp_masks = m_compMasks;
    project.frame_comp_mask_ids = m_frameCompMaskIds;
    project.frame_shape_comp_modes = m_frameShapeCompModes;
    project.dynamic_masks = m_dynamicMasks;
    project.frame_dynamic_mask_ids = m_frameDynamicMaskIds;
    project.frame_dynamic_colors = m_frameDynamicColors;

    const QStringList spriteLabels = m_state->sprites();
    project.sprite_labels.reserve(static_cast<std::size_t>(spriteCount));
    for (int i = 0; i < spriteCount; ++i) {
        if (i < spriteLabels.size()) {
            project.sprite_labels.push_back(spriteLabels.at(i).toStdString());
        } else {
            project.sprite_labels.push_back(QString("Sprite %1").arg(i).toStdString());
        }
    }

    return project;
}

void MainWindow::refreshFramePreviews()
{
    QSignalBlocker blocker(m_framePreviewList);
    m_framePreviewList->clear();

    const int count = m_frameStore->count();
    if (count <= 0 || (count == 1 && m_framesList->item(0)->text().startsWith("No frames"))) {
        auto* item = new QListWidgetItem("No frames");
        item->setFlags(item->flags() & ~Qt::ItemIsEnabled);
        m_framePreviewList->addItem(item);
        return;
    }

    for (int i = 0; i < count; ++i) {
        const cv::Mat* image = m_frameStore->at(i);
        if (!image || image->empty()) {
            continue;
        }

        cv::Mat reference = buildOriginalPreviewForIndex(i);
        cv::Mat previewMat = buildPreviewFrame(*image, reference);
        cv::Mat rgb;
        cv::cvtColor(previewMat, rgb, cv::COLOR_BGR2RGB);
        QImage previewImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
        QPixmap pixmap = QPixmap::fromImage(previewImage.copy());
        pixmap = pixmap.scaled(kPreviewIconWidth, kPreviewIconHeight, Qt::KeepAspectRatio, Qt::FastTransformation);

        auto* item = new QListWidgetItem();
        item->setIcon(QIcon(pixmap));
        item->setText(QString());
        item->setSizeHint(QSize(kPreviewItemWidth, kPreviewItemHeight));
        item->setData(kFrameIndexRole, i);
        int duration = 30;
        if (i < static_cast<int>(m_frameDurations.size()) && m_frameDurations[i] > 0) {
            duration = static_cast<int>(m_frameDurations[i]);
        }
        item->setData(kFrameDurationRole, duration);
        item->setToolTip(QString("Frame %1 (%2 ms)").arg(i).arg(duration));
        m_framePreviewList->addItem(item);
    }

    if (m_framesList->currentRow() >= 0 && m_framesList->currentRow() < m_framePreviewList->count()) {
        m_framePreviewList->setCurrentRow(m_framesList->currentRow());
    }
}

void MainWindow::updateFramePreviewAt(int index)
{
    if (index < 0 || index >= m_framePreviewList->count()) {
        return;
    }
    const cv::Mat* image = m_frameStore->at(index);
    if (!image || image->empty()) {
        return;
    }
    QListWidgetItem* item = m_framePreviewList->item(index);
    if (!item) {
        return;
    }
    cv::Mat reference = buildOriginalPreviewForIndex(index);
    const QColor gap = m_framesCanvas->palette().color(QPalette::Window);
    const cv::Scalar gapColor(gap.blue(), gap.green(), gap.red());
    cv::Mat previewMat = buildPreviewFrame(*image, reference);
    cv::Mat rgb;
    cv::cvtColor(previewMat, rgb, cv::COLOR_BGR2RGB);
    QImage previewImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
    QPixmap pixmap = QPixmap::fromImage(previewImage.copy());
    pixmap = pixmap.scaled(kPreviewIconWidth, kPreviewIconHeight, Qt::KeepAspectRatio, Qt::FastTransformation);
    item->setIcon(QIcon(pixmap));
}

cv::Mat MainWindow::buildPreviewFrame(const cv::Mat& colorized, const cv::Mat& reference) const
{
    cv::Mat color = EnsureBgr(colorized);
    cv::Mat original = buildOriginalFrame(reference);
    if (color.empty() && original.empty()) {
        return cv::Mat();
    }
    if (color.empty()) {
        return original;
    }
    if (original.empty()) {
        return color;
    }
    const QColor gap = m_framePreviewList
        ? m_framePreviewList->palette().color(QPalette::Base)
        : QApplication::palette().color(QPalette::Base);
    return buildCombinedFrame(color, original, cv::Scalar(gap.blue(), gap.green(), gap.red()));
}

cv::Mat MainWindow::buildOriginalFrame(const cv::Mat& reference) const
{
    if (reference.empty()) {
        return cv::Mat();
    }
    cv::Mat ref;
    if (reference.type() == CV_8UC1) {
        ref = reference;
    } else {
        cv::Mat gray;
        cv::cvtColor(EnsureBgr(reference), gray, cv::COLOR_BGR2GRAY);
        ref = gray;
    }
    const int levels = m_noColors > 0 ? static_cast<int>(m_noColors) : 64;
    cv::Mat output(ref.rows, ref.cols, CV_8UC3);
    for (int y = 0; y < ref.rows; ++y) {
        const uint8_t* srcRow = ref.ptr<uint8_t>(y);
        cv::Vec3b* dstRow = output.ptr<cv::Vec3b>(y);
        for (int x = 0; x < ref.cols; ++x) {
            const int idx = static_cast<int>(std::min<int>(srcRow[x], levels - 1));
            const double t = levels > 1 ? static_cast<double>(idx) / (levels - 1) : 0.0;
            const uint8_t r = static_cast<uint8_t>(255.0 * t);
            const uint8_t g = static_cast<uint8_t>(140.0 * t);
            dstRow[x] = cv::Vec3b(0, g, r);
        }
    }
    return output;
}

cv::Mat MainWindow::buildCombinedFrame(const cv::Mat& colorized,
                                       const cv::Mat& reference,
                                       const cv::Scalar& gapColor) const
{
    if (colorized.empty() || reference.empty()) {
        return cv::Mat();
    }
    cv::Mat top = EnsureBgr(colorized);
    cv::Mat bottom = EnsureBgr(reference);
    if (top.size() != bottom.size()) {
        cv::resize(bottom, bottom, top.size(), 0.0, 0.0, cv::INTER_NEAREST);
    }
    cv::Mat combined(top.rows + bottom.rows + kFrameGapPixels, top.cols, CV_8UC3, gapColor);
    top.copyTo(combined(cv::Rect(0, 0, top.cols, top.rows)));
    bottom.copyTo(combined(cv::Rect(0, top.rows + kFrameGapPixels, bottom.cols, bottom.rows)));
    return combined;
}

cv::Mat MainWindow::buildCombinedMaskPreview(const cv::Mat& colorized,
                                             const cv::Mat& reference,
                                             const cv::Mat& mask,
                                             const cv::Vec3b& color,
                                             const cv::Scalar& gapColor) const
{
    cv::Mat original = buildOriginalFrame(reference);
    if (original.empty()) {
        return buildMaskPreview(colorized, mask, color);
    }
    cv::Mat previewOriginal = buildMaskPreview(original, mask, color);
    return buildCombinedFrame(colorized, previewOriginal, gapColor);
}

cv::Mat MainWindow::buildOriginalPreviewForIndex(int index) const
{
    if (index < 0) {
        return cv::Mat();
    }
    if (index >= 0 && index < static_cast<int>(m_frameRefs.size())) {
        const cv::Mat& ref = m_frameRefs[static_cast<std::size_t>(index)];
        if (!ref.empty()) {
            return ref;
        }
    }
    if (const cv::Mat* frame = m_frameStore->at(index)) {
        return buildReferenceFrame(*frame);
    }
    return cv::Mat();
}

void MainWindow::updateFrameCanvasImage(int index)
{
    if (!m_framesCanvas) {
        return;
    }
    if (index < 0) {
        m_framesCanvas->setImage(cv::Mat());
        return;
    }
    const cv::Mat* image = m_frameStore->at(index);
    if (!image || image->empty()) {
        m_framesCanvas->setImage(cv::Mat());
        return;
    }
    cv::Mat reference = buildOriginalPreviewForIndex(index);
    cv::Mat original = buildOriginalFrame(reference);
    if (original.empty() || !m_showOriginalFrame) {
        m_framesCanvas->setImage(*image);
        m_framesCanvas->canvas()->setGridSegments(0, 0, 0);
        return;
    }
    const QColor gap = m_framesCanvas->palette().color(QPalette::Window);
    cv::Mat combined = buildCombinedFrame(*image, original, cv::Scalar(gap.blue(), gap.green(), gap.red()));
    m_framesCanvas->setImage(combined);
    m_framesCanvas->canvas()->setGridSegments(image->rows, kFrameGapPixels, original.rows);
}

namespace {
cv::Vec3b MaskColorForIndex(int index)
{
    const double hue = std::fmod(index * (360.0 / std::max(1, MAX_DYNA_SETS_PER_FRAMEN)), 360.0);
    const double c = 1.0;
    const double x = c * (1.0 - std::fabs(std::fmod(hue / 60.0, 2.0) - 1.0));
    double r = 0.0;
    double g = 0.0;
    double b = 0.0;
    if (hue < 60.0) {
        r = c;
        g = x;
    } else if (hue < 120.0) {
        r = x;
        g = c;
    } else if (hue < 180.0) {
        g = c;
        b = x;
    } else if (hue < 240.0) {
        g = x;
        b = c;
    } else if (hue < 300.0) {
        r = x;
        b = c;
    } else {
        r = c;
        b = x;
    }
    const uint8_t rb = static_cast<uint8_t>(r * 255.0);
    const uint8_t gb = static_cast<uint8_t>(g * 255.0);
    const uint8_t bb = static_cast<uint8_t>(b * 255.0);
    return cv::Vec3b(bb, gb, rb);
}

uint16_t BgrToRgb565(const cv::Vec3b& color)
{
    const uint8_t b = color[0];
    const uint8_t g = color[1];
    const uint8_t r = color[2];
    const uint16_t r5 = static_cast<uint16_t>(r >> 3);
    const uint16_t g6 = static_cast<uint16_t>(g >> 2);
    const uint16_t b5 = static_cast<uint16_t>(b >> 3);
    return static_cast<uint16_t>((r5 << 11) | (g6 << 5) | b5);
}

cv::Vec3b Rgb565ToBgr(uint16_t value)
{
    const uint8_t r5 = static_cast<uint8_t>((value >> 11) & 0x1f);
    const uint8_t g6 = static_cast<uint8_t>((value >> 5) & 0x3f);
    const uint8_t b5 = static_cast<uint8_t>(value & 0x1f);
    const uint8_t r8 = static_cast<uint8_t>((r5 << 3) | (r5 >> 2));
    const uint8_t g8 = static_cast<uint8_t>((g6 << 2) | (g6 >> 4));
    const uint8_t b8 = static_cast<uint8_t>((b5 << 3) | (b5 >> 2));
    return cv::Vec3b(b8, g8, r8);
}
}

void MainWindow::ensureMaskDataSize()
{
    const int frameCount = m_frameStore ? m_frameStore->count() : 0;
    if (frameCount < 0) {
        return;
    }

    const cv::Mat* firstFrame = (frameCount > 0) ? m_frameStore->at(0) : nullptr;
    const int width = firstFrame ? firstFrame->cols : 0;
    const int height = firstFrame ? firstFrame->rows : 0;

    if (m_compMasks.size() != MAX_MASKS) {
        m_compMasks.resize(MAX_MASKS);
    }
    if (m_dynamicMasks.size() != MAX_DYNA_SETS_PER_FRAMEN) {
        m_dynamicMasks.resize(MAX_DYNA_SETS_PER_FRAMEN);
    }
    for (int i = 0; i < MAX_MASKS; ++i) {
        cv::Mat& mask = m_compMasks[static_cast<std::size_t>(i)];
        if (width > 0 && height > 0 && (mask.empty() || mask.cols != width || mask.rows != height)) {
            mask = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
        }
    }
    for (int i = 0; i < MAX_DYNA_SETS_PER_FRAMEN; ++i) {
        cv::Mat& mask = m_dynamicMasks[static_cast<std::size_t>(i)];
        if (width > 0 && height > 0 && (mask.empty() || mask.cols != width || mask.rows != height)) {
            mask = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
        }
    }

    m_frameRefs.resize(static_cast<std::size_t>(frameCount));
    m_frameDynamicColors.resize(static_cast<std::size_t>(frameCount));
    m_frameCompMaskIds.resize(static_cast<std::size_t>(frameCount), 255);
    m_frameDynamicMaskIds.resize(static_cast<std::size_t>(frameCount), 255);
    m_frameShapeCompModes.resize(static_cast<std::size_t>(frameCount), 0);

    for (int i = 0; i < frameCount; ++i) {
        if (const cv::Mat* frame = m_frameStore->at(i)) {
            cv::Mat& ref = m_frameRefs[static_cast<std::size_t>(i)];
            if (ref.empty() || ref.rows != frame->rows || ref.cols != frame->cols) {
                ref = buildReferenceFrame(*frame);
            }
        }
        std::vector<uint16_t>& colors = m_frameDynamicColors[static_cast<std::size_t>(i)];
        if (colors.empty()) {
            colors.resize(MAX_DYNA_SETS_PER_FRAMEN * 64, 0);
            for (int set = 0; set < MAX_DYNA_SETS_PER_FRAMEN; ++set) {
                const cv::Vec3b base = MaskColorForIndex(set);
                for (int c = 0; c < 64; ++c) {
                    const double t = static_cast<double>(c) / 63.0;
                    cv::Vec3b value(static_cast<uint8_t>(base[0] * t),
                                    static_cast<uint8_t>(base[1] * t),
                                    static_cast<uint8_t>(base[2] * t));
                    colors[set * 64 + c] = BgrToRgb565(value);
                }
            }
        }
    }

    const bool hasFrames = frameCount > 0 && m_framesList && !(frameCount == 1 && m_framesList->item(0)->text().startsWith("No frames"));
    m_frameMaskAssign->setEnabled(hasFrames);
    m_frameDynamicMaskAssign->setEnabled(hasFrames);
    m_compMaskEditToggle->setEnabled(hasFrames);
    m_compMaskPreviewToggle->setEnabled(hasFrames);
    m_dynMaskEditToggle->setEnabled(hasFrames);
    m_dynMaskPreviewToggle->setEnabled(hasFrames);
    m_shapeCompToggle->setEnabled(hasFrames);
    m_maskList->setEnabled(hasFrames);
    m_maskMoveUp->setEnabled(hasFrames);
    m_maskMoveDown->setEnabled(hasFrames);
    m_maskClearButton->setEnabled(hasFrames);
    m_dynamicMaskList->setEnabled(hasFrames);
    m_dynamicMaskMoveUp->setEnabled(hasFrames);
    m_dynamicMaskMoveDown->setEnabled(hasFrames);
    m_dynamicMaskClearButton->setEnabled(hasFrames);

    refreshMaskCombos();
    refreshDynamicMaskCombos();
}

cv::Mat MainWindow::buildReferenceFrame(const cv::Mat& source) const
{
    cv::Mat bgr = EnsureBgr(source);
    if (bgr.empty()) {
        return cv::Mat();
    }
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat ref(gray.rows, gray.cols, CV_8UC1);
    for (int y = 0; y < gray.rows; ++y) {
        const uint8_t* src = gray.ptr<uint8_t>(y);
        uint8_t* dst = ref.ptr<uint8_t>(y);
        for (int x = 0; x < gray.cols; ++x) {
            const int value = static_cast<int>((src[x] * 63 + 127) / 255);
            dst[x] = static_cast<uint8_t>(std::clamp(value, 0, 63));
        }
    }
    return ref;
}

cv::Mat MainWindow::buildMaskPreview(const cv::Mat& frame, const cv::Mat& mask, const cv::Vec3b& color) const
{
    cv::Mat preview = EnsureBgr(frame);
    if (preview.empty() || mask.empty()) {
        return preview;
    }
    const double alpha = 0.5;
    for (int y = 0; y < preview.rows; ++y) {
        cv::Vec3b* row = preview.ptr<cv::Vec3b>(y);
        const uint8_t* mrow = mask.ptr<uint8_t>(y);
        for (int x = 0; x < preview.cols; ++x) {
            if (!mrow[x]) {
                continue;
            }
            row[x][0] = static_cast<uint8_t>(row[x][0] * (1.0 - alpha) + color[0] * alpha);
            row[x][1] = static_cast<uint8_t>(row[x][1] * (1.0 - alpha) + color[1] * alpha);
            row[x][2] = static_cast<uint8_t>(row[x][2] * (1.0 - alpha) + color[2] * alpha);
        }
    }
    return preview;
}

void MainWindow::updateMaskPreviewForFrame(int index)
{
    if (index < 0 || index >= m_frameStore->count()) {
        m_framesCanvas->canvas()->clearPreviewImage();
        return;
    }
    const cv::Mat* frame = m_frameStore->at(index);
    if (!frame || frame->empty()) {
        return;
    }
    ensureMaskDataSize();
    cv::Mat reference = buildOriginalPreviewForIndex(index);
    const QColor gap = m_framesCanvas->palette().color(QPalette::Window);
    const cv::Scalar gapColor(gap.blue(), gap.green(), gap.red());
    if (!m_showOriginalFrame && (m_dynMaskPreviewEnabled || m_compMaskPreviewEnabled)) {
        m_framesCanvas->canvas()->clearPreviewImage();
        return;
    }
    if (m_dynMaskPreviewEnabled) {
        const int maskId = currentFrameDynamicMaskId();
        if (maskId >= 0 && maskId < static_cast<int>(m_dynamicMasks.size())) {
            const cv::Mat& mask = m_dynamicMasks[static_cast<std::size_t>(maskId)];
            const cv::Mat preview = buildCombinedMaskPreview(*frame, reference, mask, cv::Vec3b(0, 200, 255), gapColor);
            m_framesCanvas->canvas()->setPreviewImage(preview);
            return;
        }
    }
    if (m_compMaskPreviewEnabled) {
        const int maskId = currentFrameMaskId();
        if (maskId >= 0 && maskId < static_cast<int>(m_compMasks.size())) {
            const cv::Mat& mask = m_compMasks[static_cast<std::size_t>(maskId)];
            const cv::Mat preview = buildCombinedMaskPreview(*frame, reference, mask, cv::Vec3b(200, 0, 200), gapColor);
            m_framesCanvas->canvas()->setPreviewImage(preview);
            return;
        }
    }
    m_framesCanvas->canvas()->clearPreviewImage();
}

void MainWindow::refreshMaskCombos()
{
    if (!m_maskList || !m_frameMaskAssign) {
        return;
    }
    const int currentMask = m_maskList->currentRow();
    const int currentAssign = currentFrameMaskId();

    QSignalBlocker maskBlocker(m_maskList);
    m_maskList->clear();
    for (int i = 0; i < MAX_MASKS; ++i) {
        auto* item = new QListWidgetItem(QString("Mask %1").arg(i));
        item->setData(Qt::UserRole, i);
        item->setData(Qt::UserRole + 1, QStringLiteral("mask"));
        item->setSizeHint(QSize(kPreviewItemWidth, kPreviewItemHeight));
        m_maskList->addItem(item);
    }

    QSignalBlocker assignBlocker(m_frameMaskAssign);
    m_frameMaskAssign->clear();
    m_frameMaskAssign->addItem("None", -1);
    for (int i = 0; i < MAX_MASKS; ++i) {
        m_frameMaskAssign->addItem(QString("Mask %1").arg(i), i);
    }

    if (currentMask >= 0 && currentMask < m_maskList->count()) {
        m_maskList->setCurrentRow(currentMask);
    } else if (m_maskList->count() > 0) {
        m_maskList->setCurrentRow(0);
    }
    if (currentAssign >= 0) {
        m_frameMaskAssign->setCurrentIndex(currentAssign + 1);
    } else {
        m_frameMaskAssign->setCurrentIndex(0);
    }
    updateMaskPreviewIcons();
}

void MainWindow::refreshDynamicMaskCombos()
{
    if (!m_dynamicMaskList || !m_frameDynamicMaskAssign) {
        return;
    }
    const int currentMask = m_dynamicMaskList->currentRow();
    const int currentAssign = currentFrameDynamicMaskId();

    QSignalBlocker maskBlocker(m_dynamicMaskList);
    m_dynamicMaskList->clear();
    for (int i = 0; i < MAX_DYNA_SETS_PER_FRAMEN; ++i) {
        auto* item = new QListWidgetItem(QString("Dynamic %1").arg(i));
        item->setData(Qt::UserRole, i);
        item->setData(Qt::UserRole + 1, QStringLiteral("dynamic"));
        item->setSizeHint(QSize(kPreviewItemWidth, kPreviewItemHeight));
        m_dynamicMaskList->addItem(item);
    }

    QSignalBlocker assignBlocker(m_frameDynamicMaskAssign);
    m_frameDynamicMaskAssign->clear();
    m_frameDynamicMaskAssign->addItem("None", -1);
    for (int i = 0; i < MAX_DYNA_SETS_PER_FRAMEN; ++i) {
        m_frameDynamicMaskAssign->addItem(QString("Dynamic %1").arg(i), i);
    }

    if (currentMask >= 0 && currentMask < m_dynamicMaskList->count()) {
        m_dynamicMaskList->setCurrentRow(currentMask);
    } else if (m_dynamicMaskList->count() > 0) {
        m_dynamicMaskList->setCurrentRow(0);
    }
    if (currentAssign >= 0) {
        m_frameDynamicMaskAssign->setCurrentIndex(currentAssign + 1);
    } else {
        m_frameDynamicMaskAssign->setCurrentIndex(0);
    }
    updateDynamicMaskPreviewIcons();
}

cv::Mat MainWindow::buildMaskIconImage(const cv::Mat& mask, const cv::Vec3b& color) const
{
    if (mask.empty()) {
        return cv::Mat();
    }
    cv::Mat output(mask.rows, mask.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < mask.rows; ++y) {
        const uint8_t* src = mask.ptr<uint8_t>(y);
        cv::Vec3b* dst = output.ptr<cv::Vec3b>(y);
        for (int x = 0; x < mask.cols; ++x) {
            if (src[x]) {
                dst[x] = color;
            }
        }
    }
    return output;
}

void MainWindow::updateMaskPreviewIcons()
{
    if (!m_maskList) {
        return;
    }
    const cv::Vec3b color(200, 0, 200);
    for (int i = 0; i < m_maskList->count() && i < static_cast<int>(m_compMasks.size()); ++i) {
        const cv::Mat& mask = m_compMasks[static_cast<std::size_t>(i)];
        cv::Mat iconMat = buildMaskIconImage(mask, color);
        if (iconMat.empty()) {
            if (QListWidgetItem* item = m_maskList->item(i)) {
                item->setIcon(QIcon());
            }
            continue;
        }
        cv::Mat rgb;
        cv::cvtColor(iconMat, rgb, cv::COLOR_BGR2RGB);
        QImage iconImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
        QPixmap pixmap = QPixmap::fromImage(iconImage.copy());
        pixmap = pixmap.scaled(kPreviewIconWidth, kPreviewIconHeight, Qt::KeepAspectRatio, Qt::FastTransformation);
        if (QListWidgetItem* item = m_maskList->item(i)) {
            item->setIcon(QIcon(pixmap));
        }
    }
}

void MainWindow::updateDynamicMaskPreviewIcons()
{
    if (!m_dynamicMaskList) {
        return;
    }
    const cv::Vec3b color(0, 200, 255);
    for (int i = 0; i < m_dynamicMaskList->count() && i < static_cast<int>(m_dynamicMasks.size()); ++i) {
        const cv::Mat& mask = m_dynamicMasks[static_cast<std::size_t>(i)];
        cv::Mat iconMat = buildMaskIconImage(mask, color);
        if (iconMat.empty()) {
            if (QListWidgetItem* item = m_dynamicMaskList->item(i)) {
                item->setIcon(QIcon());
            }
            continue;
        }
        cv::Mat rgb;
        cv::cvtColor(iconMat, rgb, cv::COLOR_BGR2RGB);
        QImage iconImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
        QPixmap pixmap = QPixmap::fromImage(iconImage.copy());
        pixmap = pixmap.scaled(kPreviewIconWidth, kPreviewIconHeight, Qt::KeepAspectRatio, Qt::FastTransformation);
        if (QListWidgetItem* item = m_dynamicMaskList->item(i)) {
            item->setIcon(QIcon(pixmap));
        }
    }
}

void MainWindow::applyMaskListOrder()
{
    if (!m_maskList || m_maskReorderActive) {
        return;
    }
    m_maskReorderActive = true;
    const int count = m_maskList->count();
    std::vector<int> mapping(static_cast<std::size_t>(count), -1);
    std::vector<cv::Mat> reordered(m_compMasks.size());
    for (int i = 0; i < count && i < static_cast<int>(m_compMasks.size()); ++i) {
        QListWidgetItem* item = m_maskList->item(i);
        const int oldIndex = item ? item->data(Qt::UserRole).toInt() : i;
        if (oldIndex >= 0 && oldIndex < static_cast<int>(m_compMasks.size())) {
            reordered[static_cast<std::size_t>(i)] = m_compMasks[static_cast<std::size_t>(oldIndex)];
            mapping[static_cast<std::size_t>(oldIndex)] = i;
        }
    }
    if (!reordered.empty()) {
        m_compMasks = std::move(reordered);
    }
    for (auto& id : m_frameCompMaskIds) {
        if (id != 255 && id < mapping.size() && mapping[id] >= 0) {
            id = static_cast<uint8_t>(mapping[id]);
        }
    }
    for (int i = 0; i < count; ++i) {
        QListWidgetItem* item = m_maskList->item(i);
        if (item) {
            item->setText(QString("Mask %1").arg(i));
            item->setData(Qt::UserRole, i);
            item->setData(Qt::UserRole + 1, QStringLiteral("mask"));
            item->setSizeHint(QSize(kPreviewItemWidth, kPreviewItemHeight));
        }
    }
    updateMaskPreviewIcons();
    if (m_framesList && m_framesList->currentRow() >= 0) {
        const int row = m_framesList->currentRow();
        QSignalBlocker blocker(m_frameMaskAssign);
        const uint8_t value = m_frameCompMaskIds[static_cast<std::size_t>(row)];
        m_frameMaskAssign->setCurrentIndex(value == 255 ? 0 : value + 1);
    }
    updateMaskPreviewForFrame(m_framesList->currentRow());
    m_maskReorderActive = false;
}

void MainWindow::applyDynamicMaskListOrder()
{
    if (!m_dynamicMaskList || m_dynamicMaskReorderActive) {
        return;
    }
    m_dynamicMaskReorderActive = true;
    const int count = m_dynamicMaskList->count();
    std::vector<int> mapping(static_cast<std::size_t>(count), -1);
    std::vector<cv::Mat> reordered(m_dynamicMasks.size());
    for (int i = 0; i < count && i < static_cast<int>(m_dynamicMasks.size()); ++i) {
        QListWidgetItem* item = m_dynamicMaskList->item(i);
        const int oldIndex = item ? item->data(Qt::UserRole).toInt() : i;
        if (oldIndex >= 0 && oldIndex < static_cast<int>(m_dynamicMasks.size())) {
            reordered[static_cast<std::size_t>(i)] = m_dynamicMasks[static_cast<std::size_t>(oldIndex)];
            mapping[static_cast<std::size_t>(oldIndex)] = i;
        }
    }
    if (!reordered.empty()) {
        m_dynamicMasks = std::move(reordered);
    }
    for (auto& id : m_frameDynamicMaskIds) {
        if (id != 255 && id < mapping.size() && mapping[id] >= 0) {
            id = static_cast<uint8_t>(mapping[id]);
        }
    }
    for (int i = 0; i < count; ++i) {
        QListWidgetItem* item = m_dynamicMaskList->item(i);
        if (item) {
            item->setText(QString("Dynamic %1").arg(i));
            item->setData(Qt::UserRole, i);
            item->setData(Qt::UserRole + 1, QStringLiteral("dynamic"));
            item->setSizeHint(QSize(kPreviewItemWidth, kPreviewItemHeight));
        }
    }
    updateDynamicMaskPreviewIcons();
    if (m_framesList && m_framesList->currentRow() >= 0) {
        const int row = m_framesList->currentRow();
        QSignalBlocker blocker(m_frameDynamicMaskAssign);
        const uint8_t value = m_frameDynamicMaskIds[static_cast<std::size_t>(row)];
        m_frameDynamicMaskAssign->setCurrentIndex(value == 255 ? 0 : value + 1);
    }
    updateMaskPreviewForFrame(m_framesList->currentRow());
    m_dynamicMaskReorderActive = false;
}

int MainWindow::currentFrameMaskId() const
{
    const int row = m_framesList ? m_framesList->currentRow() : -1;
    if (row < 0 || row >= static_cast<int>(m_frameCompMaskIds.size())) {
        return -1;
    }
    const uint8_t value = m_frameCompMaskIds[static_cast<std::size_t>(row)];
    return value == 255 ? -1 : static_cast<int>(value);
}

int MainWindow::currentFrameDynamicMaskId() const
{
    const int row = m_framesList ? m_framesList->currentRow() : -1;
    if (row < 0 || row >= static_cast<int>(m_frameDynamicMaskIds.size())) {
        return -1;
    }
    const uint8_t value = m_frameDynamicMaskIds[static_cast<std::size_t>(row)];
    return value == 255 ? -1 : static_cast<int>(value);
}

void MainWindow::setCurrentFrameMaskId(int id)
{
    const int row = m_framesList ? m_framesList->currentRow() : -1;
    if (row < 0 || row >= static_cast<int>(m_frameCompMaskIds.size())) {
        return;
    }
    const uint8_t value = (id < 0) ? 255 : static_cast<uint8_t>(id);
    m_frameCompMaskIds[static_cast<std::size_t>(row)] = value;
}

void MainWindow::setCurrentFrameDynamicMaskId(int id)
{
    const int row = m_framesList ? m_framesList->currentRow() : -1;
    if (row < 0 || row >= static_cast<int>(m_frameDynamicMaskIds.size())) {
        return;
    }
    const uint8_t value = (id < 0) ? 255 : static_cast<uint8_t>(id);
    m_frameDynamicMaskIds[static_cast<std::size_t>(row)] = value;
}

cv::Mat* MainWindow::activeComparisonMask()
{
    const int id = m_maskList ? m_maskList->currentRow() : -1;
    if (id < 0 || id >= static_cast<int>(m_compMasks.size())) {
        return nullptr;
    }
    return &m_compMasks[static_cast<std::size_t>(id)];
}

cv::Mat* MainWindow::activeDynamicMask()
{
    const int id = m_dynamicMaskList ? m_dynamicMaskList->currentRow() : -1;
    if (id < 0 || id >= static_cast<int>(m_dynamicMasks.size())) {
        return nullptr;
    }
    return &m_dynamicMasks[static_cast<std::size_t>(id)];
}

void MainWindow::swapMaskEntries(int a, int b)
{
    if (a < 0 || b < 0 || a >= static_cast<int>(m_compMasks.size()) || b >= static_cast<int>(m_compMasks.size())) {
        return;
    }
    std::swap(m_compMasks[static_cast<std::size_t>(a)], m_compMasks[static_cast<std::size_t>(b)]);
    for (auto& id : m_frameCompMaskIds) {
        if (id == a) {
            id = static_cast<uint8_t>(b);
        } else if (id == b) {
            id = static_cast<uint8_t>(a);
        }
    }
    updateMaskPreviewIcons();
    if (m_framesList && m_framesList->currentRow() >= 0) {
        const int row = m_framesList->currentRow();
        QSignalBlocker blocker(m_frameMaskAssign);
        const uint8_t value = m_frameCompMaskIds[static_cast<std::size_t>(row)];
        m_frameMaskAssign->setCurrentIndex(value == 255 ? 0 : value + 1);
    }
    updateMaskPreviewForFrame(m_framesList->currentRow());
}

void MainWindow::swapDynamicMaskEntries(int a, int b)
{
    if (a < 0 || b < 0 || a >= static_cast<int>(m_dynamicMasks.size()) || b >= static_cast<int>(m_dynamicMasks.size())) {
        return;
    }
    std::swap(m_dynamicMasks[static_cast<std::size_t>(a)], m_dynamicMasks[static_cast<std::size_t>(b)]);
    for (auto& id : m_frameDynamicMaskIds) {
        if (id == a) {
            id = static_cast<uint8_t>(b);
        } else if (id == b) {
            id = static_cast<uint8_t>(a);
        }
    }
    updateDynamicMaskPreviewIcons();
    if (m_framesList && m_framesList->currentRow() >= 0) {
        const int row = m_framesList->currentRow();
        QSignalBlocker blocker(m_frameDynamicMaskAssign);
        const uint8_t value = m_frameDynamicMaskIds[static_cast<std::size_t>(row)];
        m_frameDynamicMaskAssign->setCurrentIndex(value == 255 ? 0 : value + 1);
    }
    updateMaskPreviewForFrame(m_framesList->currentRow());
}

void MainWindow::applyToolToMask(cv::Mat& mask, DrawTool tool, const QPoint& start, const QPoint& end, bool erase)
{
    const uint8_t value = erase ? 0 : 1;
    if (tool == DrawTool::Point) {
        if (start.x() >= 0 && start.x() < mask.cols && start.y() >= 0 && start.y() < mask.rows) {
            mask.at<uint8_t>(start.y(), start.x()) = value;
        }
        return;
    }
    if (tool == DrawTool::Line) {
        cv::line(mask, cv::Point(start.x(), start.y()), cv::Point(end.x(), end.y()), cv::Scalar(value), 1);
        return;
    }
    if (tool == DrawTool::Rect || tool == DrawTool::RectFill) {
        const cv::Point tl(std::min(start.x(), end.x()), std::min(start.y(), end.y()));
        const cv::Point br(std::max(start.x(), end.x()), std::max(start.y(), end.y()));
        const int thickness = (tool == DrawTool::RectFill) ? -1 : 1;
        cv::rectangle(mask, cv::Rect(tl, br), cv::Scalar(value), thickness);
        return;
    }
    if (tool == DrawTool::Circle || tool == DrawTool::CircleFill) {
        const int dx = end.x() - start.x();
        const int dy = end.y() - start.y();
        const int radius = static_cast<int>(std::sqrt(dx * dx + dy * dy));
        const int thickness = (tool == DrawTool::CircleFill) ? -1 : 1;
        cv::circle(mask, cv::Point(start.x(), start.y()), radius, cv::Scalar(value), thickness);
        return;
    }
    if (tool == DrawTool::Ellipse || tool == DrawTool::EllipseFill) {
        const cv::Point center((start.x() + end.x()) / 2, (start.y() + end.y()) / 2);
        const cv::Size axes(std::abs(end.x() - start.x()) / 2, std::abs(end.y() - start.y()) / 2);
        const int thickness = (tool == DrawTool::EllipseFill) ? -1 : 1;
        cv::ellipse(mask, center, axes, 0.0, 0.0, 360.0, cv::Scalar(value), thickness);
        return;
    }
}

void MainWindow::applyMaskFill(cv::Mat& mask, int x, int y, bool erase)
{
    if (mask.empty()) {
        return;
    }
    if (x < 0 || y < 0 || x >= mask.cols || y >= mask.rows) {
        return;
    }
    const uint8_t value = erase ? 0 : 1;
    const uint8_t target = mask.at<uint8_t>(y, x);
    if (target == value) {
        return;
    }
    cv::Mat floodMask(mask.rows + 2, mask.cols + 2, CV_8UC1, cv::Scalar(0));
    cv::floodFill(mask, floodMask, cv::Point(x, y), cv::Scalar(value), nullptr, cv::Scalar(0), cv::Scalar(0), 4);
}

void MainWindow::resetUndoStacks()
{
    m_frameUndoStacks.clear();
    m_spriteUndoStacks.clear();
    m_frameUndoActive = false;
    m_spriteUndoActive = false;
    updateUndoActions();
}

void MainWindow::ensureUndoStacksSize()
{
    const int frameCount = m_frameStore ? m_frameStore->count() : 0;
    const int spriteCount = m_spriteStore ? m_spriteStore->count() : 0;
    if (frameCount >= 0) {
        m_frameUndoStacks.resize(static_cast<std::size_t>(frameCount));
    }
    if (spriteCount >= 0) {
        m_spriteUndoStacks.resize(static_cast<std::size_t>(spriteCount));
    }
    updateUndoActions();
}

void MainWindow::pushUndoSnapshot(bool isFrame, int index)
{
    if (index < 0) {
        return;
    }
    std::vector<UndoStack>& stacks = isFrame ? m_frameUndoStacks : m_spriteUndoStacks;
    if (index >= static_cast<int>(stacks.size())) {
        return;
    }
    cv::Mat* image = isFrame ? m_frameStore->atMutable(index) : m_spriteStore->atMutable(index);
    if (!image || image->empty()) {
        return;
    }
    UndoStack& stack = stacks[static_cast<std::size_t>(index)];
    UndoState state;
    state.image = image->clone();
    if (isFrame && m_compMaskEditEnabled) {
        const int maskId = m_maskList->currentRow();
        if (maskId >= 0 && maskId < static_cast<int>(m_compMasks.size())) {
            state.mask = m_compMasks[static_cast<std::size_t>(maskId)].clone();
            state.mask_kind = MaskKind::Comparison;
            state.mask_index = maskId;
        }
    } else if (isFrame && m_dynMaskEditEnabled) {
        const int maskId = m_dynamicMaskList->currentRow();
        if (maskId >= 0 && maskId < static_cast<int>(m_dynamicMasks.size())) {
            state.mask = m_dynamicMasks[static_cast<std::size_t>(maskId)].clone();
            state.mask_kind = MaskKind::Dynamic;
            state.mask_index = maskId;
        }
    }
    stack.undo.push_back(std::move(state));
    if (stack.undo.size() > kMaxUndoDepth) {
        stack.undo.erase(stack.undo.begin());
    }
    stack.redo.clear();
    updateUndoActions();
}

bool MainWindow::undoEdit(bool isFrame)
{
    const int index = isFrame ? m_framesList->currentRow() : m_spritesList->currentRow();
    std::vector<UndoStack>& stacks = isFrame ? m_frameUndoStacks : m_spriteUndoStacks;
    if (index < 0 || index >= static_cast<int>(stacks.size())) {
        return false;
    }
    UndoStack& stack = stacks[static_cast<std::size_t>(index)];
    if (stack.undo.empty()) {
        return false;
    }
    cv::Mat* image = isFrame ? m_frameStore->atMutable(index) : m_spriteStore->atMutable(index);
    if (!image || image->empty()) {
        return false;
    }
    UndoState current;
    current.image = image->clone();
    if (isFrame && m_compMaskEditEnabled) {
        const int maskId = m_maskList->currentRow();
        if (maskId >= 0 && maskId < static_cast<int>(m_compMasks.size())) {
            current.mask = m_compMasks[static_cast<std::size_t>(maskId)].clone();
            current.mask_kind = MaskKind::Comparison;
            current.mask_index = maskId;
        }
    } else if (isFrame && m_dynMaskEditEnabled) {
        const int maskId = m_dynamicMaskList->currentRow();
        if (maskId >= 0 && maskId < static_cast<int>(m_dynamicMasks.size())) {
            current.mask = m_dynamicMasks[static_cast<std::size_t>(maskId)].clone();
            current.mask_kind = MaskKind::Dynamic;
            current.mask_index = maskId;
        }
    }
    stack.redo.push_back(std::move(current));
    UndoState previous = stack.undo.back();
    stack.undo.pop_back();
    *image = previous.image.clone();
    if (isFrame && !previous.mask.empty()) {
        if (previous.mask_kind == MaskKind::Comparison &&
            previous.mask_index >= 0 && previous.mask_index < static_cast<int>(m_compMasks.size())) {
            m_compMasks[static_cast<std::size_t>(previous.mask_index)] = previous.mask.clone();
            updateMaskPreviewIcons();
        } else if (previous.mask_kind == MaskKind::Dynamic &&
                   previous.mask_index >= 0 && previous.mask_index < static_cast<int>(m_dynamicMasks.size())) {
            m_dynamicMasks[static_cast<std::size_t>(previous.mask_index)] = previous.mask.clone();
            updateDynamicMaskPreviewIcons();
        }
    }
    if (isFrame) {
        updateFrameCanvasImage(index);
        updateFramePreviewAt(index);
        updateMaskPreviewForFrame(index);
    } else {
        m_spritesCanvas->setImage(*image);
    }
    updateUndoActions();
    return true;
}

bool MainWindow::redoEdit(bool isFrame)
{
    const int index = isFrame ? m_framesList->currentRow() : m_spritesList->currentRow();
    std::vector<UndoStack>& stacks = isFrame ? m_frameUndoStacks : m_spriteUndoStacks;
    if (index < 0 || index >= static_cast<int>(stacks.size())) {
        return false;
    }
    UndoStack& stack = stacks[static_cast<std::size_t>(index)];
    if (stack.redo.empty()) {
        return false;
    }
    cv::Mat* image = isFrame ? m_frameStore->atMutable(index) : m_spriteStore->atMutable(index);
    if (!image || image->empty()) {
        return false;
    }
    UndoState current;
    current.image = image->clone();
    if (isFrame && m_compMaskEditEnabled) {
        const int maskId = m_maskList->currentRow();
        if (maskId >= 0 && maskId < static_cast<int>(m_compMasks.size())) {
            current.mask = m_compMasks[static_cast<std::size_t>(maskId)].clone();
            current.mask_kind = MaskKind::Comparison;
            current.mask_index = maskId;
        }
    } else if (isFrame && m_dynMaskEditEnabled) {
        const int maskId = m_dynamicMaskList->currentRow();
        if (maskId >= 0 && maskId < static_cast<int>(m_dynamicMasks.size())) {
            current.mask = m_dynamicMasks[static_cast<std::size_t>(maskId)].clone();
            current.mask_kind = MaskKind::Dynamic;
            current.mask_index = maskId;
        }
    }
    stack.undo.push_back(std::move(current));
    if (stack.undo.size() > kMaxUndoDepth) {
        stack.undo.erase(stack.undo.begin());
    }
    UndoState next = stack.redo.back();
    stack.redo.pop_back();
    *image = next.image.clone();
    if (isFrame && !next.mask.empty()) {
        if (next.mask_kind == MaskKind::Comparison &&
            next.mask_index >= 0 && next.mask_index < static_cast<int>(m_compMasks.size())) {
            m_compMasks[static_cast<std::size_t>(next.mask_index)] = next.mask.clone();
            updateMaskPreviewIcons();
        } else if (next.mask_kind == MaskKind::Dynamic &&
                   next.mask_index >= 0 && next.mask_index < static_cast<int>(m_dynamicMasks.size())) {
            m_dynamicMasks[static_cast<std::size_t>(next.mask_index)] = next.mask.clone();
            updateDynamicMaskPreviewIcons();
        }
    }
    if (isFrame) {
        updateFrameCanvasImage(index);
        updateFramePreviewAt(index);
        updateMaskPreviewForFrame(index);
    } else {
        m_spritesCanvas->setImage(*image);
    }
    updateUndoActions();
    return true;
}

void MainWindow::updateUndoActions()
{
    const bool isFrame = isFrameContext();
    const int index = isFrame ? m_framesList->currentRow() : m_spritesList->currentRow();
    const std::vector<UndoStack>& stacks = isFrame ? m_frameUndoStacks : m_spriteUndoStacks;
    bool canUndo = false;
    bool canRedo = false;
    if (index >= 0 && index < static_cast<int>(stacks.size())) {
        const UndoStack& stack = stacks[static_cast<std::size_t>(index)];
        canUndo = !stack.undo.empty();
        canRedo = !stack.redo.empty();
    }
    if (m_undoAction) {
        m_undoAction->setEnabled(canUndo);
    }
    if (m_redoAction) {
        m_redoAction->setEnabled(canRedo);
    }
}

bool MainWindow::isFrameContext() const
{
    if (m_canvasTabs && m_canvasTabs->currentWidget() == m_framesCanvas) {
        return true;
    }
    if (m_canvasTabs && m_canvasTabs->currentWidget() == m_spritesCanvas) {
        return false;
    }
    return m_framesList && m_framesList->currentRow() >= 0;
}

void MainWindow::refreshRecentMenu()
{
    m_recentMenu->clear();
    for (const auto& entry : m_state->recentFiles()) {
        auto* action = new QAction(entry, this);
        connect(action, &QAction::triggered, this, [this, entry]() {
            openProjectFile(entry);
        });
        m_recentMenu->addAction(action);
    }
    if (m_state->recentFiles().isEmpty()) {
        m_recentMenu->addAction("(Empty)")->setEnabled(false);
    }
}

void MainWindow::refreshImageList()
{
    m_imagesList->clear();
    for (const auto& entry : m_state->images()) {
        m_imagesList->addItem(entry);
    }
    if (m_state->images().isEmpty()) {
        m_imagesList->addItem("No images imported");
        m_imagesCanvas->setTitle("Images canvas (placeholder)");
        m_imagesCanvas->setStatusText("No image selected");
        m_imagesCanvas->setImage(cv::Mat());
        return;
    }
    if (m_imagesList->currentRow() < 0 && m_imagesList->count() > 0) {
        m_imagesList->setCurrentRow(0);
        return;
    }
    const QString currentPath = m_imagesList->currentItem() ? m_imagesList->currentItem()->text() : QString();
    if (!currentPath.isEmpty() && m_state->images().contains(currentPath)) {
        showImageForPath(currentPath);
    } else if (const auto* last = m_imageStore->latest()) {
        if (!last->image.empty()) {
            m_imagesCanvas->setImage(last->image);
        }
    }
}

void MainWindow::refreshCounts()
{
    m_countsLabel->setText(QString("Frames: %1, Sprites: %2").arg(m_state->frameCount()).arg(m_state->spriteCount()));
}

void MainWindow::refreshFrameSpriteLists()
{
    m_framesList->clear();
    m_spritesList->clear();

    if (m_state->frames().isEmpty()) {
        m_framesList->addItem("No frames loaded");
        m_framesCanvas->setTitle("Frames canvas (placeholder)");
        m_framesCanvas->setStatusText("No frames loaded");
        updateFrameCanvasImage(-1);
        m_frameStore->clear();
        updateMetadataForFrame(-1);
    } else {
        while (m_frameStore->count() < m_state->frames().size()) {
            m_frameStore->add(MakePlaceholderImage(kDefaultFrameWidth, kDefaultFrameHeight,
                                                   cv::Scalar(18, 18, 18),
                                                   cv::Scalar(55, 55, 55)));
        }
        while (m_frameStore->count() > m_state->frames().size()) {
            m_frameStore->removeAt(m_frameStore->count() - 1);
        }
        m_framesList->addItems(m_state->frames());
        if (m_framesList->currentRow() < 0 && m_framesList->count() > 0) {
            m_framesList->setCurrentRow(0);
        } else if (m_framesList->currentRow() >= 0) {
            showFrameAtIndex(m_framesList->currentRow());
        }
    }

    if (m_state->sprites().isEmpty()) {
        m_spritesList->addItem("No sprites loaded");
        m_spritesCanvas->setTitle("Sprites canvas (placeholder)");
        m_spritesCanvas->setStatusText("No sprites loaded");
        m_spritesCanvas->setImage(cv::Mat());
        m_spriteStore->clear();
        updateMetadataForSprite(-1);
    } else {
        while (m_spriteStore->count() < m_state->sprites().size()) {
            m_spriteStore->add(MakePlaceholderImage(kDefaultSpriteWidth, kDefaultSpriteHeight,
                                                    cv::Scalar(24, 24, 24),
                                                    cv::Scalar(70, 70, 70)));
        }
        while (m_spriteStore->count() > m_state->sprites().size()) {
            m_spriteStore->removeAt(m_spriteStore->count() - 1);
        }
        m_spritesList->addItems(m_state->sprites());
        if (m_spritesList->currentRow() < 0 && m_spritesList->count() > 0) {
            m_spritesList->setCurrentRow(0);
        } else if (m_spritesList->currentRow() >= 0) {
            showSpriteAtIndex(m_spritesList->currentRow());
        }
    }

    updateFrameJumpRange();
    refreshFramePreviews();
    ensureUndoStacksSize();
    ensureMaskDataSize();
}

void MainWindow::setInspectorSelection(const QString& label)
{
    m_selectionLabel->setText(label);
}

void MainWindow::updateSelectionFromLists()
{
    if (m_framesList->currentRow() >= 0) {
        setInspectorSelection(QString("Frame: %1").arg(m_framesList->currentItem()->text()));
        showFrameAtIndex(m_framesList->currentRow());
        updateMetadataForFrame(m_framesList->currentRow());
        return;
    }
    if (m_spritesList->currentRow() >= 0) {
        setInspectorSelection(QString("Sprite: %1").arg(m_spritesList->currentItem()->text()));
        showSpriteAtIndex(m_spritesList->currentRow());
        updateMetadataForSprite(m_spritesList->currentRow());
        return;
    }
    if (m_imagesList->currentRow() >= 0) {
        setInspectorSelection(QString("Image: %1").arg(m_imagesList->currentItem()->text()));
        showImageForPath(m_imagesList->currentItem()->text());
        return;
    }
    setInspectorSelection("None");
    updateMetadataForFrame(-1);
    updateMetadataForSprite(-1);
    updateUndoActions();
    if (!m_compMaskPreviewEnabled && !m_dynMaskPreviewEnabled) {
        m_framesCanvas->canvas()->clearPreviewImage();
    }
}

void MainWindow::showImageForPath(const QString& path)
{
    if (path.isEmpty()) {
        return;
    }
    const auto* entry = m_imageStore->findByPath(path);
    if (!entry) {
        cv::Mat image;
        if (LoadImageFile(path, image)) {
            m_imageStore->addImage(path, image);
            entry = m_imageStore->findByPath(path);
        }
    }
    if (entry && !entry->image.empty()) {
        m_imagesCanvas->setImage(entry->image);
    } else {
        m_imagesCanvas->setImage(cv::Mat());
    }
}

void MainWindow::showFrameAtIndex(int index)
{
    const cv::Mat* image = m_frameStore->at(index);
    if (image && !image->empty()) {
        m_framesCanvas->canvas()->clearPreviewImage();
        updateFrameCanvasImage(index);
        if (index == 0 && m_drawPointEnabled == false) {
            QTimer::singleShot(0, this, [this]() {
                m_framesCanvas->canvas()->requestFitOnResize(true);
            });
        }
        if (m_frameMaskAssign && index >= 0 && index < static_cast<int>(m_frameCompMaskIds.size())) {
            QSignalBlocker blockAssign(m_frameMaskAssign);
            const int maskId = m_frameCompMaskIds[static_cast<std::size_t>(index)];
            m_frameMaskAssign->setCurrentIndex(maskId == 255 ? 0 : maskId + 1);
        }
        if (m_frameDynamicMaskAssign && index >= 0 && index < static_cast<int>(m_frameDynamicMaskIds.size())) {
            QSignalBlocker blockAssign(m_frameDynamicMaskAssign);
            const int maskId = m_frameDynamicMaskIds[static_cast<std::size_t>(index)];
            m_frameDynamicMaskAssign->setCurrentIndex(maskId == 255 ? 0 : maskId + 1);
        }
        if (m_shapeCompToggle && index >= 0 && index < static_cast<int>(m_frameShapeCompModes.size())) {
            QSignalBlocker blockShape(m_shapeCompToggle);
            const uint8_t value = m_frameShapeCompModes[static_cast<std::size_t>(index)];
            m_shapeCompToggle->setChecked(value != 0);
        }
        updateMaskPreviewForFrame(index);
    } else {
        updateFrameCanvasImage(-1);
    }
}

void MainWindow::showSpriteAtIndex(int index)
{
    const cv::Mat* image = m_spriteStore->at(index);
    if (image && !image->empty()) {
        m_spritesCanvas->canvas()->clearPreviewImage();
        m_spritesCanvas->setImage(*image);
    } else {
        m_spritesCanvas->setImage(cv::Mat());
    }
}

void MainWindow::populateBookmarks(const std::vector<uint32_t>& frameStarts,
                                   const std::vector<std::string>& names)
{
    m_bookmarksCombo->clear();
    if (frameStarts.empty() || names.empty()) {
        m_bookmarksCombo->setEnabled(false);
        m_bookmarksCombo->addItem("No bookmarks");
        return;
    }

    const std::size_t count = std::min(frameStarts.size(), names.size());
    int added = 0;
    for (std::size_t i = 0; i < count; ++i) {
        if (names[i].empty()) {
            continue;
        }
        const QString label = QString::fromStdString(names[i]) +
            QString(" (Frame %1)").arg(frameStarts[i]);
        m_bookmarksCombo->addItem(label, static_cast<int>(frameStarts[i]));
        ++added;
    }
    if (added == 0) {
        m_bookmarksCombo->setEnabled(false);
        m_bookmarksCombo->addItem("No bookmarks");
    } else {
        m_bookmarksCombo->setEnabled(true);
        m_bookmarksCombo->setCurrentIndex(0);
    }
}

void MainWindow::handleToolPress(bool isFrame, int x, int y, Qt::MouseButton button)
{
    if (!m_drawPointEnabled) {
        return;
    }
    if (isFrame && (m_compMaskEditEnabled || m_dynMaskEditEnabled)) {
        const int index = m_framesList->currentRow();
        if (index < 0) {
            return;
        }
        if (!m_showOriginalFrame) {
            statusBar()->showMessage("Show the original frame to edit masks.", 2000);
            return;
        }
        const cv::Mat* frameImage = m_frameStore->at(index);
        const int frameHeight = frameImage ? frameImage->rows : 0;
        if (frameHeight > 0) {
            if (y < frameHeight) {
                statusBar()->showMessage("Mask editing uses the original frame below.", 2000);
                return;
            }
            y -= frameHeight;
        }
        ensureMaskDataSize();
        cv::Mat* mask = nullptr;
        if (m_compMaskEditEnabled) {
            const int assigned = currentFrameMaskId();
            const int selected = m_maskList->currentRow();
            if (assigned < 0 || assigned != selected) {
                statusBar()->showMessage("Assign the selected mask to this frame before editing.", 2000);
                return;
            }
            mask = activeComparisonMask();
        } else if (m_dynMaskEditEnabled) {
            const int assigned = currentFrameDynamicMaskId();
            const int selected = m_dynamicMaskList->currentRow();
            if (assigned < 0 || assigned != selected) {
                statusBar()->showMessage("Assign the selected dynamic mask to this frame before editing.", 2000);
                return;
            }
            mask = activeDynamicMask();
        }
        if (!mask || mask->empty()) {
            return;
        }
        if (m_drawTool == DrawTool::MagicFill || m_drawTool == DrawTool::Point) {
            if (x < 0 || y < 0 || x >= mask->cols || y >= mask->rows) {
                return;
            }
            if (!m_frameUndoActive) {
                pushUndoSnapshot(true, index);
                m_frameUndoActive = true;
            }
            if (m_drawTool == DrawTool::MagicFill) {
                applyMaskFill(*mask, x, y, button == Qt::RightButton);
            } else {
                applyToolToMask(*mask, DrawTool::Point, QPoint(x, y), QPoint(x, y), button == Qt::RightButton);
            }
            if (m_compMaskEditEnabled) {
                updateMaskPreviewIcons();
            } else {
                updateDynamicMaskPreviewIcons();
            }
            updateMaskPreviewForFrame(index);
            return;
        }
        if (m_drawTool == DrawTool::ColorPicker) {
            return;
        }
        if (!m_frameUndoActive) {
            pushUndoSnapshot(true, index);
            m_frameUndoActive = true;
        }
        m_frameStart = QPoint(x, y);
        m_frameHasStart = true;
        m_frameStartButton = button;
        return;
    }
    cv::Mat* image = isFrame ? m_frameStore->atMutable(m_framesList->currentRow())
                             : m_spriteStore->atMutable(m_spritesList->currentRow());
    if (!image || image->empty()) {
        return;
    }
    if (isFrame) {
        const int frameHeight = image->rows;
        if (frameHeight > 0 && y >= frameHeight) {
            statusBar()->showMessage("Color edits apply to the top frame only.", 2000);
            return;
        }
    }
    const int currentIndex = isFrame ? m_framesList->currentRow() : m_spritesList->currentRow();
    bool& undoActive = isFrame ? m_frameUndoActive : m_spriteUndoActive;
    if (m_drawTool != DrawTool::ColorPicker) {
        if (!undoActive) {
            pushUndoSnapshot(isFrame, currentIndex);
            undoActive = true;
        }
    }
    if (m_drawTool == DrawTool::Point) {
        applyToolToImage(*image, DrawTool::Point, QPoint(x, y), QPoint(x, y), button == Qt::RightButton);
    } else if (m_drawTool == DrawTool::ColorPicker) {
        pickColorFromImage(*image, x, y);
    } else if (m_drawTool == DrawTool::MagicFill) {
        applyMagicFill(*image, x, y);
    } else {
        if (isFrame) {
            m_frameStart = QPoint(x, y);
            m_frameHasStart = true;
            m_frameStartButton = button;
        } else {
            m_spriteStart = QPoint(x, y);
            m_spriteHasStart = true;
            m_spriteStartButton = button;
        }
    }
    if (isFrame) {
        updateFrameCanvasImage(m_framesList->currentRow());
        updateFramePreviewAt(m_framesList->currentRow());
        updateMaskPreviewForFrame(m_framesList->currentRow());
    } else {
        m_spritesCanvas->setImage(*image);
    }
}

void MainWindow::handleToolDrag(bool isFrame, int x, int y, Qt::MouseButtons buttons)
{
    if (!m_drawPointEnabled) {
        return;
    }
    if (isFrame && (m_compMaskEditEnabled || m_dynMaskEditEnabled)) {
        const int index = m_framesList->currentRow();
        if (index < 0) {
            return;
        }
        if (!m_showOriginalFrame) {
            return;
        }
        const cv::Mat* frameImage = m_frameStore->at(index);
        const int frameHeight = frameImage ? frameImage->rows : 0;
        if (frameHeight > 0) {
            if (y < frameHeight) {
                return;
            }
            y -= frameHeight;
        }
        cv::Mat* mask = nullptr;
        cv::Vec3b color(200, 0, 200);
        if (m_compMaskEditEnabled) {
            const int assigned = currentFrameMaskId();
            const int selected = m_maskList->currentRow();
            if (assigned < 0 || assigned != selected) {
                return;
            }
            mask = activeComparisonMask();
            color = cv::Vec3b(200, 0, 200);
        } else if (m_dynMaskEditEnabled) {
            const int assigned = currentFrameDynamicMaskId();
            const int selected = m_dynamicMaskList->currentRow();
            if (assigned < 0 || assigned != selected) {
                return;
            }
            mask = activeDynamicMask();
            color = cv::Vec3b(0, 200, 255);
        }
        if (!mask || mask->empty()) {
            return;
        }
        if (m_drawTool == DrawTool::Point) {
            const bool erase = buttons.testFlag(Qt::RightButton);
            applyToolToMask(*mask, DrawTool::Point, QPoint(x, y), QPoint(x, y), erase);
            if (m_compMaskEditEnabled) {
                updateMaskPreviewIcons();
            } else {
                updateDynamicMaskPreviewIcons();
            }
            updateMaskPreviewForFrame(index);
            return;
        }
        if (!m_frameHasStart) {
            return;
        }
        cv::Mat previewMask = mask->clone();
        const bool erase = buttons.testFlag(Qt::RightButton);
        applyToolToMask(previewMask, m_drawTool, m_frameStart, QPoint(x, y), erase);
        if (const cv::Mat* frame = m_frameStore->at(index)) {
            cv::Mat reference = buildOriginalPreviewForIndex(index);
            const QColor gap = m_framesCanvas->palette().color(QPalette::Window);
            const cv::Scalar gapColor(gap.blue(), gap.green(), gap.red());
            cv::Mat preview = buildCombinedMaskPreview(*frame, reference, previewMask, color, gapColor);
            m_framesCanvas->canvas()->setPreviewImage(preview);
        }
        return;
    }
    cv::Mat* image = isFrame ? m_frameStore->atMutable(m_framesList->currentRow())
                             : m_spriteStore->atMutable(m_spritesList->currentRow());
    if (!image || image->empty()) {
        return;
    }
    if (isFrame) {
        const int frameHeight = image->rows;
        if (frameHeight > 0 && y >= frameHeight) {
            return;
        }
    }
    if (m_drawTool == DrawTool::Point) {
        const bool erase = buttons.testFlag(Qt::RightButton);
        applyToolToImage(*image, DrawTool::Point, QPoint(x, y), QPoint(x, y), erase);
        if (isFrame) {
            updateFrameCanvasImage(m_framesList->currentRow());
            updateFramePreviewAt(m_framesList->currentRow());
            updateMaskPreviewForFrame(m_framesList->currentRow());
        } else {
            m_spritesCanvas->setImage(*image);
        }
        return;
    }
    const bool hasStart = isFrame ? m_frameHasStart : m_spriteHasStart;
    if (!hasStart) {
        return;
    }
    const QPoint start = isFrame ? m_frameStart : m_spriteStart;
    const bool erase = buttons.testFlag(Qt::RightButton);
    cv::Mat preview = image->clone();
    applyToolToImage(preview, m_drawTool, start, QPoint(x, y), erase);
    if (isFrame) {
        cv::Mat reference = buildOriginalPreviewForIndex(m_framesList->currentRow());
        cv::Mat original = m_showOriginalFrame ? buildOriginalFrame(reference) : cv::Mat();
        if (original.empty()) {
            m_framesCanvas->canvas()->setPreviewImage(preview);
        } else {
            const QColor gap = m_framesCanvas->palette().color(QPalette::Window);
            cv::Mat combined = buildCombinedFrame(preview,
                                                  original,
                                                  cv::Scalar(gap.blue(), gap.green(), gap.red()));
            m_framesCanvas->canvas()->setPreviewImage(combined);
        }
    } else {
        m_spritesCanvas->canvas()->setPreviewImage(preview);
    }
}

void MainWindow::handleToolRelease(bool isFrame, int x, int y, Qt::MouseButton button)
{
    if (!m_drawPointEnabled) {
        return;
    }
    if (isFrame && (m_compMaskEditEnabled || m_dynMaskEditEnabled)) {
        if (m_drawTool == DrawTool::Point ||
            m_drawTool == DrawTool::ColorPicker ||
            m_drawTool == DrawTool::MagicFill) {
            m_frameUndoActive = false;
            return;
        }
        if (!m_frameHasStart) {
            return;
        }
        const int index = m_framesList->currentRow();
        if (index < 0) {
            m_frameHasStart = false;
            return;
        }
        if (!m_showOriginalFrame) {
            m_frameHasStart = false;
            return;
        }
        const cv::Mat* frameImage = m_frameStore->at(index);
        const int frameHeight = frameImage ? frameImage->rows : 0;
        if (frameHeight > 0) {
            if (y < frameHeight) {
                m_frameHasStart = false;
                return;
            }
            y -= frameHeight;
        }
        cv::Mat* mask = nullptr;
        if (m_compMaskEditEnabled) {
            const int assigned = currentFrameMaskId();
            const int selected = m_maskList->currentRow();
            if (assigned < 0 || assigned != selected) {
                m_frameHasStart = false;
                return;
            }
            mask = activeComparisonMask();
        } else if (m_dynMaskEditEnabled) {
            const int assigned = currentFrameDynamicMaskId();
            const int selected = m_dynamicMaskList->currentRow();
            if (assigned < 0 || assigned != selected) {
                m_frameHasStart = false;
                return;
            }
            mask = activeDynamicMask();
        }
        if (!mask || mask->empty()) {
            m_frameHasStart = false;
            return;
        }
        const bool erase = (m_frameStartButton == Qt::RightButton || button == Qt::RightButton);
        applyToolToMask(*mask, m_drawTool, m_frameStart, QPoint(x, y), erase);
        m_frameHasStart = false;
        m_framesCanvas->canvas()->clearPreviewImage();
        if (m_compMaskEditEnabled) {
            updateMaskPreviewIcons();
        } else {
            updateDynamicMaskPreviewIcons();
        }
        updateMaskPreviewForFrame(index);
        m_frameUndoActive = false;
        return;
    }
    if (m_drawTool == DrawTool::Point ||
        m_drawTool == DrawTool::ColorPicker ||
        m_drawTool == DrawTool::MagicFill) {
        if (isFrame) {
            m_frameUndoActive = false;
        } else {
            m_spriteUndoActive = false;
        }
        return;
    }
    bool hasStart = isFrame ? m_frameHasStart : m_spriteHasStart;
    if (!hasStart) {
        return;
    }
    QPoint start = isFrame ? m_frameStart : m_spriteStart;
    Qt::MouseButton startButton = isFrame ? m_frameStartButton : m_spriteStartButton;
    if (isFrame) {
        m_frameHasStart = false;
    } else {
        m_spriteHasStart = false;
    }
    cv::Mat* image = isFrame ? m_frameStore->atMutable(m_framesList->currentRow())
                             : m_spriteStore->atMutable(m_spritesList->currentRow());
    if (!image || image->empty()) {
        return;
    }
    if (isFrame) {
        const int frameHeight = image->rows;
        if (frameHeight > 0 && y >= frameHeight) {
            if (isFrame) {
                m_frameHasStart = false;
            }
            return;
        }
    }
    const bool erase = (startButton == Qt::RightButton || button == Qt::RightButton);
    applyToolToImage(*image, m_drawTool, start, QPoint(x, y), erase);
    if (isFrame) {
        m_framesCanvas->canvas()->clearPreviewImage();
        updateFrameCanvasImage(m_framesList->currentRow());
        updateFramePreviewAt(m_framesList->currentRow());
        updateMaskPreviewForFrame(m_framesList->currentRow());
        m_frameUndoActive = false;
    } else {
        m_spritesCanvas->canvas()->clearPreviewImage();
        m_spritesCanvas->setImage(*image);
        m_spriteUndoActive = false;
    }
}

cv::Scalar MainWindow::currentDrawColor(bool erase) const
{
    if (erase) {
        return cv::Scalar(0, 0, 0, 255);
    }
    return m_drawColor;
}

void MainWindow::applyToolToImage(cv::Mat& image,
                                  DrawTool tool,
                                  const QPoint& start,
                                  const QPoint& end,
                                  bool erase)
{
    const cv::Scalar color = currentDrawColor(erase);
    const cv::Point p1(start.x(), start.y());
    const cv::Point p2(end.x(), end.y());
    if (tool == DrawTool::Point) {
        if (image.type() == CV_8UC3) {
            image.at<cv::Vec3b>(p1.y, p1.x) = cv::Vec3b(static_cast<uint8_t>(color[0]),
                                                        static_cast<uint8_t>(color[1]),
                                                        static_cast<uint8_t>(color[2]));
        } else if (image.type() == CV_8UC4) {
            image.at<cv::Vec4b>(p1.y, p1.x) = cv::Vec4b(static_cast<uint8_t>(color[0]),
                                                        static_cast<uint8_t>(color[1]),
                                                        static_cast<uint8_t>(color[2]),
                                                        static_cast<uint8_t>(color[3]));
        } else if (image.type() == CV_8UC1) {
            image.at<uint8_t>(p1.y, p1.x) = static_cast<uint8_t>(erase ? 0 : 255);
        }
        return;
    }

    int thickness = 1;
    if (tool == DrawTool::RectFill || tool == DrawTool::CircleFill || tool == DrawTool::EllipseFill) {
        thickness = cv::FILLED;
    }

    const int dx = p2.x - p1.x;
    const int dy = p2.y - p1.y;
    const int radius = static_cast<int>(std::sqrt(static_cast<double>(dx * dx + dy * dy)));
    const cv::Point center((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
    const cv::Size axes(std::abs(dx) / 2, std::abs(dy) / 2);

    switch (tool) {
        case DrawTool::Line:
            cv::line(image, p1, p2, color, 1);
            break;
        case DrawTool::Rect:
        case DrawTool::RectFill:
            cv::rectangle(image, p1, p2, color, thickness);
            break;
        case DrawTool::Circle:
        case DrawTool::CircleFill:
            cv::circle(image, p1, radius, color, thickness);
            break;
        case DrawTool::Ellipse:
        case DrawTool::EllipseFill:
            cv::ellipse(image, center, axes, 0.0, 0.0, 360.0, color, thickness);
            break;
        default:
            break;
    }
}

void MainWindow::applyMagicFill(cv::Mat& image, int x, int y)
{
    if (image.empty()) {
        return;
    }
    cv::Scalar newColor = currentDrawColor(false);
    cv::Rect bounds;
    cv::Mat mask(image.rows + 2, image.cols + 2, CV_8UC1, cv::Scalar(0));
    cv::floodFill(image,
                  mask,
                  cv::Point(x, y),
                  newColor,
                  &bounds,
                  cv::Scalar(0, 0, 0, 0),
                  cv::Scalar(0, 0, 0, 0),
                  4);
}

void MainWindow::pickColorFromImage(const cv::Mat& image, int x, int y)
{
    if (image.empty()) {
        return;
    }
    if (image.type() == CV_8UC3) {
        const cv::Vec3b color = image.at<cv::Vec3b>(y, x);
        m_drawColor = cv::Scalar(color[0], color[1], color[2], 255);
    } else if (image.type() == CV_8UC4) {
        const cv::Vec4b color = image.at<cv::Vec4b>(y, x);
        m_drawColor = cv::Scalar(color[0], color[1], color[2], color[3]);
    } else if (image.type() == CV_8UC1) {
        const uint8_t value = image.at<uint8_t>(y, x);
        m_drawColor = cv::Scalar(value, value, value, 255);
    }
    statusBar()->showMessage(QString("Picked color: %1, %2, %3")
                                 .arg(m_drawColor[2])
                                 .arg(m_drawColor[1])
                                 .arg(m_drawColor[0]),
                             2000);
}

void MainWindow::cancelCurrentDraw()
{
    m_frameHasStart = false;
    m_spriteHasStart = false;
    m_frameUndoActive = false;
    m_spriteUndoActive = false;
    m_framesCanvas->canvas()->clearPreviewImage();
    m_spritesCanvas->canvas()->clearPreviewImage();
    updateMaskPreviewForFrame(m_framesList->currentRow());
    statusBar()->showMessage("Draw canceled", 1500);
}

void MainWindow::updateMetadataForFrame(int index)
{
    if (index < 0 || index >= static_cast<int>(m_frameDurations.size())) {
        m_frameMetaLabel->setText("-");
        return;
    }
    QString section;
    if (!m_sectionStarts.empty() && !m_sectionNames.empty()) {
        const std::size_t count = std::min(m_sectionStarts.size(), m_sectionNames.size());
        for (std::size_t i = 0; i < count; ++i) {
            const uint32_t start = m_sectionStarts[i];
            const uint32_t next = (i + 1 < count) ? m_sectionStarts[i + 1] : static_cast<uint32_t>(m_frameDurations.size());
            if (index >= static_cast<int>(start) && index < static_cast<int>(next)) {
                if (!m_sectionNames[i].empty()) {
                    section = QString::fromStdString(m_sectionNames[i]);
                }
                break;
            }
        }
    }
    const QString duration = m_frameDurations[index] > 0 ? QString("%1 ms").arg(m_frameDurations[index]) : "n/a";
    if (!section.isEmpty()) {
        m_frameMetaLabel->setText(QString("%1, %2").arg(duration, section));
    } else {
        m_frameMetaLabel->setText(duration);
    }
}

void MainWindow::updateMetadataForSprite(int index)
{
    if (index < 0 || index >= static_cast<int>(m_spriteNames.size())) {
        m_spriteMetaLabel->setText("-");
        return;
    }
    const std::string& name = m_spriteNames[index];
    if (name.empty()) {
        m_spriteMetaLabel->setText("-");
    } else {
        m_spriteMetaLabel->setText(QString::fromStdString(name));
    }
}

void MainWindow::applyFrameFilter(const QString& text)
{
    const QString needle = text.trimmed();
    for (int i = 0; i < m_framesList->count(); ++i) {
        auto* item = m_framesList->item(i);
        const bool isPlaceholder = item->text().startsWith("No frames");
        if (needle.isEmpty() || isPlaceholder) {
            item->setHidden(false);
        } else {
            item->setHidden(!item->text().contains(needle, Qt::CaseInsensitive));
        }
    }
}

void MainWindow::applySpriteFilter(const QString& text)
{
    const QString needle = text.trimmed();
    for (int i = 0; i < m_spritesList->count(); ++i) {
        auto* item = m_spritesList->item(i);
        const bool isPlaceholder = item->text().startsWith("No sprites");
        if (needle.isEmpty() || isPlaceholder) {
            item->setHidden(false);
        } else {
            item->setHidden(!item->text().contains(needle, Qt::CaseInsensitive));
        }
    }
}

void MainWindow::updateFrameJumpRange()
{
    const int count = m_framesList->count();
    if (count <= 0 || (count == 1 && m_framesList->item(0)->text().startsWith("No frames"))) {
        m_frameJump->setEnabled(false);
        m_frameJump->setRange(0, 0);
        m_frameJump->setValue(0);
        return;
    }
    m_frameJump->setEnabled(true);
    m_frameJump->setRange(0, std::max(0, count - 1));
}
