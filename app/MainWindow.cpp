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
#include <QSizePolicy>
#include <QStyle>
#include <QScrollArea>
#include <QColorDialog>
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
constexpr int kPreviewIconWidthHd = kPreviewIconWidth * 2;
constexpr int kPreviewIconHeightHd = kPreviewIconHeight * 2;
constexpr int kPreviewItemWidthHd = kPreviewIconWidthHd + 12;
constexpr int kPreviewItemHeightHd = kPreviewIconHeightHd + 28;
constexpr int kPaletteSwatchSize = 26;
constexpr int kPaletteItemSize = 32;
constexpr int kReducedPaletteCount = 64;
constexpr int kMaxUndoDepth = 50;
constexpr int kFrameGapPixels = 8;

constexpr int kFrameIndexRole = Qt::UserRole + 1;
constexpr int kFrameDurationRole = Qt::UserRole + 2;
constexpr int kPreviewIconSizeRole = Qt::UserRole + 3;
constexpr int kPaletteDisabledRole = Qt::UserRole + 4;

cv::Mat EnsureBgr(const cv::Mat& source);
uint16_t BgrToRgb565(const cv::Vec3b& color);
cv::Vec3b Rgb565ToBgr(uint16_t value);

struct FrameLayout {
    int topWidth = 0;
    int topHeight = 0;
    int bottomWidth = 0;
    int bottomHeight = 0;
    int combinedWidth = 0;
    int topX = 0;
    int bottomX = 0;
};

FrameLayout BuildFrameLayout(int topWidth, int topHeight, int bottomWidth, int bottomHeight)
{
    FrameLayout layout;
    layout.topWidth = topWidth;
    layout.topHeight = topHeight;
    layout.bottomWidth = bottomWidth;
    layout.bottomHeight = bottomHeight;
    layout.combinedWidth = std::max(layout.topWidth, layout.bottomWidth);
    if (layout.combinedWidth > 0) {
        layout.topX = (layout.combinedWidth - layout.topWidth) / 2;
        layout.bottomX = (layout.combinedWidth - layout.bottomWidth) / 2;
    }
    return layout;
}

FrameLayout BuildFrameLayout(const cv::Mat& top, const cv::Mat& bottom)
{
    return BuildFrameLayout(top.cols, top.rows, bottom.cols, bottom.rows);
}

cv::Mat BuildStackedFrames(const std::vector<cv::Mat>& frames, const cv::Scalar& gapColor)
{
    std::vector<cv::Mat> valid;
    valid.reserve(frames.size());
    for (const auto& frame : frames) {
        if (!frame.empty()) {
            valid.push_back(frame);
        }
    }
    if (valid.empty()) {
        return cv::Mat();
    }
    if (valid.size() == 1) {
        return valid.front();
    }
    int width = 0;
    int height = 0;
    for (const auto& frame : valid) {
        width = std::max(width, frame.cols);
        height += frame.rows;
    }
    height += static_cast<int>(valid.size() - 1) * kFrameGapPixels;
    cv::Mat combined(height, width, CV_8UC3, gapColor);
    int y = 0;
    for (const auto& frame : valid) {
        const int x = (width - frame.cols) / 2;
        frame.copyTo(combined(cv::Rect(x, y, frame.cols, frame.rows)));
        y += frame.rows + kFrameGapPixels;
    }
    return combined;
}

cv::Mat BuildDisplayOriginal(const cv::Mat& original, const cv::Size& topSize)
{
    if (original.empty()) {
        return cv::Mat();
    }
    if (topSize.width <= 0 || topSize.height <= 0 || original.size() == topSize) {
        return original;
    }
    cv::Mat scaled;
    cv::resize(original, scaled, topSize, 0.0, 0.0, cv::INTER_NEAREST);
    return scaled;
}

cv::Mat AdjustBrightnessToSource(const cv::Mat& source, const cv::Mat& resized)
{
    if (source.empty() || resized.empty()) {
        return resized;
    }
    const cv::Scalar srcMean = cv::mean(source);
    const cv::Scalar dstMean = cv::mean(resized);
    std::array<double, 3> scale{1.0, 1.0, 1.0};
    for (int c = 0; c < 3; ++c) {
        if (dstMean[c] > 1e-3) {
            scale[c] = srcMean[c] / dstMean[c];
        }
    }
    cv::Mat floatMat;
    resized.convertTo(floatMat, CV_32FC3);
    std::vector<cv::Mat> channels;
    cv::split(floatMat, channels);
    for (int c = 0; c < 3 && c < static_cast<int>(channels.size()); ++c) {
        channels[c] *= static_cast<float>(scale[c]);
    }
    cv::merge(channels, floatMat);
    cv::Mat adjusted;
    floatMat.convertTo(adjusted, CV_8UC3);
    return adjusted;
}

cv::Mat Scale2xBgr(const cv::Mat& source)
{
    cv::Mat src = EnsureBgr(source);
    if (src.empty()) {
        return cv::Mat();
    }
    cv::Mat out(src.rows * 2, src.cols * 2, CV_8UC3);
    auto sample = [&](int x, int y) -> cv::Vec3b {
        const int sx = std::clamp(x, 0, src.cols - 1);
        const int sy = std::clamp(y, 0, src.rows - 1);
        return src.at<cv::Vec3b>(sy, sx);
    };
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            const cv::Vec3b A = sample(x - 1, y - 1);
            const cv::Vec3b B = sample(x, y - 1);
            const cv::Vec3b C = sample(x + 1, y - 1);
            const cv::Vec3b D = sample(x - 1, y);
            const cv::Vec3b E = sample(x, y);
            const cv::Vec3b F = sample(x + 1, y);
            const cv::Vec3b G = sample(x - 1, y + 1);
            const cv::Vec3b H = sample(x, y + 1);
            const cv::Vec3b I = sample(x + 1, y + 1);

            cv::Vec3b E0 = E;
            cv::Vec3b E1 = E;
            cv::Vec3b E2 = E;
            cv::Vec3b E3 = E;
            if (B != H && D != F) {
                E0 = (D == B) ? D : E;
                E1 = (B == F) ? F : E;
                E2 = (D == H) ? D : E;
                E3 = (H == F) ? F : E;
            }

            out.at<cv::Vec3b>(y * 2, x * 2) = E0;
            out.at<cv::Vec3b>(y * 2, x * 2 + 1) = E1;
            out.at<cv::Vec3b>(y * 2 + 1, x * 2) = E2;
            out.at<cv::Vec3b>(y * 2 + 1, x * 2 + 1) = E3;
        }
    }
    return out;
}

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
        painter->setClipRect(opt.rect);

        const QFontMetrics metrics(opt.font);
        const int padding = 6;
        const int textHeight = metrics.height() + 4;
        QRect contentRect = opt.rect.adjusted(padding, padding, -padding, -padding);
        const QSize iconSize = index.data(kPreviewIconSizeRole).toSize().isValid()
            ? index.data(kPreviewIconSizeRole).toSize()
            : opt.decorationSize.isValid() ? opt.decorationSize : QSize(kPreviewIconWidth, kPreviewIconHeight);
        const int iconX = contentRect.left() + (contentRect.width() - iconSize.width()) / 2;
        QRect iconRect(iconX,
                       contentRect.top() + textHeight + 2,
                       iconSize.width(),
                       contentRect.height() - textHeight - 2);
        QRect previewRect = iconRect;

        const QIcon icon = qvariant_cast<QIcon>(index.data(Qt::DecorationRole));
        if (!icon.isNull()) {
            const QPixmap pixmap = icon.pixmap(iconSize);
            const QSize targetSize = pixmap.size().scaled(iconRect.size(), Qt::KeepAspectRatio);
            const QPoint topLeft(iconRect.center().x() - targetSize.width() / 2,
                                 iconRect.center().y() - targetSize.height() / 2);
            previewRect = QRect(topLeft, targetSize);
            painter->drawPixmap(previewRect, pixmap);
        }

        const QVariant frameValue = index.data(kFrameIndexRole);
        if (frameValue.isValid()) {
            const int frameIndex = frameValue.toInt();
            const int duration = index.data(kFrameDurationRole).toInt();
            const QString leftText = QString("F%1").arg(frameIndex);
            const QString rightText = duration > 0 ? QString("%1 ms").arg(duration) : "n/a";
            painter->setPen(opt.palette.text().color());
            const QRect textRect(previewRect.left(), contentRect.top(), previewRect.width(), textHeight);
            painter->drawText(textRect, Qt::AlignLeft | Qt::AlignVCenter, leftText);
            painter->drawText(textRect, Qt::AlignRight | Qt::AlignVCenter, rightText);
        }

        painter->restore();
    }

    QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const override
    {
        const QFontMetrics metrics(option.font);
        const int padding = 6;
        const int textHeight = metrics.height() + 4;
        const QSize iconSize = index.data(kPreviewIconSizeRole).toSize().isValid()
            ? index.data(kPreviewIconSizeRole).toSize()
            : option.decorationSize.isValid() ? option.decorationSize : QSize(kPreviewIconWidth, kPreviewIconHeight);
        const int height = iconSize.height() + textHeight + padding * 2;
        const int width = std::max(iconSize.width() + padding * 2, kPreviewItemWidth);
        return QSize(width, height);
    }
};

class ToolPreviewDelegate : public QStyledItemDelegate
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
        const QSize iconSize = index.data(kPreviewIconSizeRole).toSize().isValid()
            ? index.data(kPreviewIconSizeRole).toSize()
            : opt.decorationSize.isValid() ? opt.decorationSize : QSize(kPreviewIconWidth, kPreviewIconHeight);
        const int iconX = contentRect.left() + (contentRect.width() - iconSize.width()) / 2;
        QRect iconRect(iconX,
                       contentRect.top(),
                       iconSize.width(),
                       std::max(0, contentRect.height() - textHeight - 2));
        QRect textRect(iconRect.left(),
                       contentRect.bottom() - textHeight + 1,
                       iconRect.width(),
                       textHeight);

        const QIcon icon = qvariant_cast<QIcon>(index.data(Qt::DecorationRole));
        if (!icon.isNull()) {
            const QPixmap pixmap = icon.pixmap(iconSize);
            const QSize targetSize = pixmap.size().scaled(iconRect.size(), Qt::KeepAspectRatio);
            const QPoint topLeft(iconRect.center().x() - targetSize.width() / 2,
                                 iconRect.center().y() - targetSize.height() / 2);
            painter->drawPixmap(QRect(topLeft, targetSize), pixmap);
        }

        const QString text = index.data(Qt::DisplayRole).toString();
        painter->setPen(opt.palette.text().color());
        painter->drawText(textRect, Qt::AlignHCenter | Qt::AlignVCenter, text);

        painter->restore();
    }

    QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const override
    {
        const QFontMetrics metrics(option.font);
        const int padding = 6;
        const int textHeight = metrics.height() + 4;
        const QSize iconSize = index.data(kPreviewIconSizeRole).toSize().isValid()
            ? index.data(kPreviewIconSizeRole).toSize()
            : option.decorationSize.isValid() ? option.decorationSize : QSize(kPreviewIconWidth, kPreviewIconHeight);
        const int height = iconSize.height() + textHeight + padding * 2;
        const int width = std::max(iconSize.width() + padding * 2, kPreviewItemWidth);
        return QSize(width, height);
    }
};

class PaletteSwatchDelegate : public QStyledItemDelegate
{
public:
    using QStyledItemDelegate::QStyledItemDelegate;

    void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override
    {
        painter->save();

        QStyleOptionViewItem opt = option;
        initStyleOption(&opt, index);
        painter->fillRect(opt.rect, opt.palette.base());
        painter->setClipRect(opt.rect);

        const QIcon icon = qvariant_cast<QIcon>(index.data(Qt::DecorationRole));
        const int swatchSize = kPaletteItemSize;
        const QRect swatchRect(opt.rect.center().x() - swatchSize / 2,
                               opt.rect.center().y() - swatchSize / 2,
                               swatchSize,
                               swatchSize);
        if (!icon.isNull()) {
            const QSize iconSize = opt.decorationSize.isValid() ? opt.decorationSize
                                                                : QSize(kPaletteSwatchSize, kPaletteSwatchSize);
            const QPixmap pixmap = icon.pixmap(iconSize, QIcon::Normal, QIcon::Off);
            const QSize targetSize = pixmap.size().scaled(swatchRect.size(), Qt::KeepAspectRatio);
            const QPoint topLeft(swatchRect.center().x() - targetSize.width() / 2,
                                 swatchRect.center().y() - targetSize.height() / 2);
            painter->drawPixmap(QRect(topLeft, targetSize), pixmap);
        }

        QPen basePen(QColor(0, 0, 0));
        basePen.setWidth(1);
        painter->setPen(basePen);
        painter->drawRect(swatchRect.adjusted(0, 0, -1, -1));

        if (index.data(kPaletteDisabledRole).toBool()) {
            painter->fillRect(swatchRect, QColor(0, 0, 0, 120));
            QPen crossPen(QColor(220, 0, 0, 200));
            crossPen.setWidth(2);
            painter->setPen(crossPen);
            painter->drawLine(swatchRect.topLeft() + QPoint(2, 2),
                              swatchRect.bottomRight() - QPoint(2, 2));
            painter->drawLine(swatchRect.topRight() + QPoint(-2, 2),
                              swatchRect.bottomLeft() + QPoint(2, -2));
        }

        if (option.state & QStyle::State_Selected) {
            const int thickness = std::max(2, kPaletteSwatchSize / 3);
            QColor selectionColor(255, 140, 0);
            if (opt.widget) {
                const QVariant value = opt.widget->property("selectionColor");
                if (value.isValid() && value.canConvert<QColor>()) {
                    selectionColor = value.value<QColor>();
                }
            }
            QPen pen(selectionColor);
            pen.setWidth(thickness);
            pen.setJoinStyle(Qt::MiterJoin);
            painter->setPen(pen);
            const int inset = thickness / 2;
            const QRect outline = swatchRect.adjusted(-inset, -inset, inset, inset);
            const QRect clipped = outline.intersected(opt.rect.adjusted(0, 0, -1, -1));
            painter->drawRect(clipped);
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

bool MaskHasContent(const cv::Mat& mask)
{
    return !mask.empty() && cv::countNonZero(mask) > 0;
}

bool IsAllBlackFrame(const cv::Mat& frame)
{
    if (frame.empty()) {
        return true;
    }
    cv::Mat bgr = EnsureBgr(frame);
    if (bgr.empty()) {
        return true;
    }
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    return cv::countNonZero(gray) == 0;
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

cv::Mat BuildBackgroundPreview(const cv::Mat& sdFrame, const cv::Mat& hdFrame, const cv::Scalar& gapColor)
{
    cv::Mat sd = EnsureBgr(sdFrame);
    cv::Mat hd = EnsureBgr(hdFrame);
    if (sd.empty() && hd.empty()) {
        return cv::Mat();
    }
    return BuildStackedFrames({sd, hd}, gapColor);
}

QSize PreviewItemSizeForIcon(const QSize& iconSize, const QFont& font)
{
    const QFontMetrics metrics(font);
    const int padding = 6;
    const int textHeight = metrics.height() + 4;
    const int height = iconSize.height() + textHeight + padding * 2;
    const int width = std::max(iconSize.width() + padding * 2, kPreviewItemWidth);
    return QSize(width, height);
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
    , m_backgroundStore(new IndexedImageStore())
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
        const UndoTarget target = currentUndoTarget();
        bool handled = false;
        switch (target) {
            case UndoTarget::Frame:
                handled = undoEdit(true);
                break;
            case UndoTarget::Sprite:
                handled = undoEdit(false);
                break;
            case UndoTarget::Background:
                handled = undoBackgroundEdit();
                break;
            case UndoTarget::CompMask:
                handled = undoMaskEdit(MaskMode::Comparison);
                break;
            case UndoTarget::DynMask:
                handled = undoMaskEdit(MaskMode::Dynamic);
                break;
            case UndoTarget::BackgroundMask:
                handled = undoBackgroundMaskEdit();
                break;
        }
        if (handled) {
            statusBar()->showMessage("Undo", 1500);
        }
    });
    connect(m_redoAction, &QAction::triggered, this, [this]() {
        const UndoTarget target = currentUndoTarget();
        bool handled = false;
        switch (target) {
            case UndoTarget::Frame:
                handled = redoEdit(true);
                break;
            case UndoTarget::Sprite:
                handled = redoEdit(false);
                break;
            case UndoTarget::Background:
                handled = redoBackgroundEdit();
                break;
            case UndoTarget::CompMask:
                handled = redoMaskEdit(MaskMode::Comparison);
                break;
            case UndoTarget::DynMask:
                handled = redoMaskEdit(MaskMode::Dynamic);
                break;
            case UndoTarget::BackgroundMask:
                handled = redoBackgroundMaskEdit();
                break;
        }
        if (handled) {
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
        m_backgroundsCanvas->canvas()->setPanningEnabled(!checked);
        m_framesCanvas->canvas()->setHoverPixelEnabled(checked);
        m_spritesCanvas->canvas()->setHoverPixelEnabled(checked);
        m_backgroundsCanvas->canvas()->setHoverPixelEnabled(checked);
        if (!checked) {
            m_framesCanvas->canvas()->clearPreviewImage();
            m_spritesCanvas->canvas()->clearPreviewImage();
            m_backgroundsCanvas->canvas()->clearPreviewImage();
            m_frameHasStart = false;
            m_spriteHasStart = false;
            m_frameUndoActive = false;
            m_spriteUndoActive = false;
            m_backgroundUndoActive = false;
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
        m_backgroundsCanvas->canvas()->requestFitOnResize(true);
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
    m_backgroundsCanvas = new CanvasWidget("Backgrounds canvas (placeholder)", tabs);
    m_framesCanvas->setMaskButtonsVisible(true);
    m_spritesCanvas->setMaskButtonsVisible(false);
    m_imagesCanvas->setMaskButtonsVisible(false);
    m_backgroundsCanvas->setMaskButtonsVisible(false);
    m_framesCanvas->setBackgroundMaskVisible(true);
    m_spritesCanvas->setBackgroundMaskVisible(false);
    m_imagesCanvas->setBackgroundMaskVisible(false);
    m_backgroundsCanvas->setBackgroundMaskVisible(false);
    m_framesCanvas->setBackgroundVisible(true);
    m_spritesCanvas->setBackgroundVisible(false);
    m_imagesCanvas->setBackgroundVisible(false);
    m_backgroundsCanvas->setBackgroundVisible(false);
    m_backgroundsCanvas->setHdButtonEnabled(true);
    tabs->addTab(m_framesCanvas, "Frames");
    tabs->addTab(m_spritesCanvas, "Sprites");
    tabs->addTab(m_imagesCanvas, "Images");
    tabs->addTab(m_backgroundsCanvas, "Backgrounds");
    setCentralWidget(tabs);
    connect(tabs, &QTabWidget::currentChanged, this, [this](int) {
        updateUndoActions();
        updateHdControlsForContext();
    });

    auto* toolDock = new QDockWidget("Tools", this);
    toolDock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    auto* toolsTabs = new QTabWidget(toolDock);
    m_toolsTabs = toolsTabs;
    auto* framesTab = new QWidget(toolsTabs);
    auto* spritesTab = new QWidget(toolsTabs);
    auto* imagesTab = new QWidget(toolsTabs);
    auto* masksTab = new QWidget(toolsTabs);
    auto* dynamicMasksTab = new QWidget(toolsTabs);
    auto* backgroundsTab = new QWidget(toolsTabs);
    m_masksTab = masksTab;
    m_dynamicMasksTab = dynamicMasksTab;
    m_backgroundsTab = backgroundsTab;
    m_spritesTab = spritesTab;

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

    m_backgroundList = new QListWidget(backgroundsTab);
    m_backgroundList->setSelectionMode(QAbstractItemView::SingleSelection);
    m_backgroundList->setViewMode(QListView::IconMode);
    m_backgroundList->setFlow(QListView::TopToBottom);
    m_backgroundList->setWrapping(false);
    m_backgroundList->setMovement(QListView::Snap);
    m_backgroundList->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_backgroundList->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    m_backgroundList->setResizeMode(QListView::Adjust);
    m_backgroundList->setUniformItemSizes(false);
    m_backgroundList->setDragDropMode(QAbstractItemView::DragOnly);
    m_backgroundList->setDragDropOverwriteMode(false);
    m_backgroundList->setDragEnabled(true);
    m_backgroundList->setAcceptDrops(false);
    m_backgroundList->setDropIndicatorShown(false);
    m_backgroundList->setIconSize(QSize(kPreviewIconWidthHd, kPreviewIconHeightHd));
    m_backgroundList->setGridSize(QSize());
    m_backgroundList->setSpacing(4);
    m_backgroundList->setItemDelegate(new ToolPreviewDelegate(m_backgroundList));
    auto* backgroundsLayout = new QVBoxLayout(backgroundsTab);
    backgroundsLayout->addWidget(m_backgroundList, 1);
    backgroundsTab->setLayout(backgroundsLayout);

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
    m_maskList->setDropIndicatorShown(false);
    m_maskList->setIconSize(QSize(kPreviewIconWidth, kPreviewIconHeight));
    m_maskList->setGridSize(QSize(kPreviewItemWidth, kPreviewItemHeight));
    m_maskList->setSpacing(6);
    m_maskList->setItemDelegate(new ToolPreviewDelegate(m_maskList));
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
    m_dynamicMaskList->setDropIndicatorShown(false);
    m_dynamicMaskList->setIconSize(QSize(kPreviewIconWidth, kPreviewIconHeight));
    m_dynamicMaskList->setGridSize(QSize(kPreviewItemWidth, kPreviewItemHeight));
    m_dynamicMaskList->setSpacing(6);
    m_dynamicMaskList->setItemDelegate(new ToolPreviewDelegate(m_dynamicMaskList));
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

    auto* colorsTab = new QWidget(toolsTabs);
    m_currentColorButton = new QToolButton(colorsTab);
    m_currentColorButton->setAutoRaise(true);
    m_currentColorButton->setFixedSize(32, 32);
    m_currentColorButton->setToolTip("Current color");
    m_colorInfoLabel = new QLabel("RGB565: 0xFFFF\nRGB: 255,255,255", colorsTab);
    m_colorInfoLabel->setFixedWidth(160);
    m_paletteSetCombo = new QComboBox(colorsTab);
    m_colorPickButton = new QPushButton("Pick Color...", colorsTab);
    m_paletteAssignButton = new QPushButton("Set Slot", colorsTab);
    m_paletteList = new QListWidget(colorsTab);
    m_paletteList->setViewMode(QListView::IconMode);
    m_paletteList->setFlow(QListView::LeftToRight);
    m_paletteList->setWrapping(true);
    m_paletteList->setMovement(QListView::Static);
    m_paletteList->setSelectionMode(QAbstractItemView::SingleSelection);
    m_paletteList->setResizeMode(QListView::Adjust);
    m_paletteList->setIconSize(QSize(kPaletteSwatchSize, kPaletteSwatchSize));
    const int paletteCellSize = kPaletteItemSize + 10;
    m_paletteList->setGridSize(QSize(paletteCellSize, paletteCellSize));
    m_paletteList->setSpacing(0);
    m_paletteList->setUniformItemSizes(true);
    m_paletteList->setResizeMode(QListView::Fixed);
    m_paletteList->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
    const int scrollExtent = m_paletteList->style()->pixelMetric(QStyle::PM_ScrollBarExtent, nullptr, m_paletteList);
    const int paletteGridWidth =
        paletteCellSize * 8 + m_paletteList->frameWidth() * 2 + scrollExtent + 4;
    m_paletteList->setMinimumWidth(paletteGridWidth);
    m_paletteList->setMaximumWidth(paletteGridWidth);
    m_paletteList->setItemDelegate(new PaletteSwatchDelegate(m_paletteList));
    m_paletteList->setFrameShape(QFrame::NoFrame);
    m_paletteList->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_paletteList->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    m_paletteList->setFixedHeight(paletteGridWidth);
    m_paletteList->setProperty("selectionColor", QColor(255, 140, 0));

    m_reducedSetCombo = new QComboBox(colorsTab);
    m_reducedAssignButton = new QPushButton("Set Slot", colorsTab);
    m_reducedPaletteList = new QListWidget(colorsTab);
    m_reducedPaletteList->setViewMode(QListView::IconMode);
    m_reducedPaletteList->setFlow(QListView::LeftToRight);
    m_reducedPaletteList->setWrapping(true);
    m_reducedPaletteList->setMovement(QListView::Static);
    m_reducedPaletteList->setSelectionMode(QAbstractItemView::SingleSelection);
    m_reducedPaletteList->setResizeMode(QListView::Fixed);
    m_reducedPaletteList->setIconSize(QSize(kPaletteSwatchSize, kPaletteSwatchSize));
    m_reducedPaletteList->setGridSize(QSize(paletteCellSize, paletteCellSize));
    m_reducedPaletteList->setSpacing(0);
    m_reducedPaletteList->setUniformItemSizes(true);
    m_reducedPaletteList->setItemDelegate(new PaletteSwatchDelegate(m_reducedPaletteList));
    m_reducedPaletteList->setFrameShape(QFrame::NoFrame);
    m_reducedPaletteList->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_reducedPaletteList->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    const int reducedGridWidth = paletteCellSize * 8 + m_reducedPaletteList->frameWidth() * 2 + 4;
    const int reducedGridHeight = paletteCellSize * 2 + m_reducedPaletteList->frameWidth() * 2 + 4;
    m_reducedPaletteList->setFixedSize(reducedGridWidth, reducedGridHeight);
    m_reducedPaletteList->setProperty("selectionColor", QColor(255, 140, 0));
    connect(m_reducedPaletteList, &QListWidget::currentRowChanged, this, [this](int row) {
        if (row < 0) {
            return;
        }
        m_reducedSlotIndex = row;
        if (m_reducedAssignButton) {
            m_reducedAssignButton->setEnabled(true);
        }
        if (m_dynamicPaletteList) {
            QSignalBlocker blocker(m_dynamicPaletteList);
            m_dynamicPaletteList->setCurrentRow(-1);
            m_dynamicSlotIndex = -1;
        }
        m_paletteSelectionIsReference = true;
        if (m_paletteList) {
            m_paletteList->setProperty("selectionColor", QColor(70, 150, 255));
            m_paletteList->viewport()->update();
        }
        const QColor color = reducedSlotColor(m_reducedPaletteIndex, row);
        setDrawColor(color, true);
    });
    auto* reducedLayout = new QVBoxLayout();
    auto* reducedTop = new QHBoxLayout();
    reducedTop->addWidget(new QLabel("Reduced set", colorsTab));
    reducedTop->addWidget(m_reducedSetCombo);
    reducedTop->addStretch(1);
    reducedTop->addWidget(m_reducedAssignButton);
    reducedLayout->addLayout(reducedTop);
    reducedLayout->addWidget(m_reducedPaletteList);

    m_dynamicSetCombo = new QComboBox(colorsTab);
    m_dynamicAssignButton = new QPushButton("Set Slot", colorsTab);
    m_dynamicPaletteList = new QListWidget(colorsTab);
    m_dynamicPaletteList->setViewMode(QListView::IconMode);
    m_dynamicPaletteList->setFlow(QListView::LeftToRight);
    m_dynamicPaletteList->setWrapping(true);
    m_dynamicPaletteList->setMovement(QListView::Static);
    m_dynamicPaletteList->setSelectionMode(QAbstractItemView::SingleSelection);
    m_dynamicPaletteList->setResizeMode(QListView::Fixed);
    m_dynamicPaletteList->setIconSize(QSize(kPaletteSwatchSize, kPaletteSwatchSize));
    m_dynamicPaletteList->setGridSize(QSize(paletteCellSize, paletteCellSize));
    m_dynamicPaletteList->setSpacing(0);
    m_dynamicPaletteList->setUniformItemSizes(true);
    m_dynamicPaletteList->setItemDelegate(new PaletteSwatchDelegate(m_dynamicPaletteList));
    m_dynamicPaletteList->setFrameShape(QFrame::NoFrame);
    m_dynamicPaletteList->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_dynamicPaletteList->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    const int dynamicGridWidth = paletteCellSize * 8 + m_dynamicPaletteList->frameWidth() * 2 + 4;
    const int dynamicGridHeight = paletteCellSize * 2 + m_dynamicPaletteList->frameWidth() * 2 + 4;
    m_dynamicPaletteList->setFixedSize(dynamicGridWidth, dynamicGridHeight);
    m_dynamicPaletteList->setProperty("selectionColor", QColor(255, 140, 0));
    connect(m_dynamicPaletteList, &QListWidget::currentRowChanged, this, [this](int row) {
        if (row < 0) {
            return;
        }
        m_dynamicSlotIndex = row;
        if (m_dynamicAssignButton) {
            m_dynamicAssignButton->setEnabled(true);
        }
        if (m_reducedPaletteList) {
            QSignalBlocker blocker(m_reducedPaletteList);
            m_reducedPaletteList->setCurrentRow(-1);
            m_reducedSlotIndex = -1;
        }
        m_paletteSelectionIsReference = true;
        if (m_paletteList) {
            m_paletteList->setProperty("selectionColor", QColor(70, 150, 255));
            m_paletteList->viewport()->update();
        }
        const QColor color = dynamicSlotColor(row);
        setDrawColor(color, true);
    });
    auto* dynamicLayout = new QVBoxLayout();
    auto* dynamicTop = new QHBoxLayout();
    dynamicTop->addWidget(new QLabel("Dynamic set", colorsTab));
    dynamicTop->addWidget(m_dynamicSetCombo);
    dynamicTop->addStretch(1);
    dynamicTop->addWidget(m_dynamicAssignButton);
    dynamicLayout->addLayout(dynamicTop);
    dynamicLayout->addWidget(m_dynamicPaletteList);

    auto* colorsScrollArea = new QScrollArea(colorsTab);
    colorsScrollArea->setWidgetResizable(true);
    colorsScrollArea->setFrameShape(QFrame::NoFrame);
    auto* colorsContent = new QWidget(colorsScrollArea);
    auto* colorLayout = new QVBoxLayout(colorsContent);
    auto* colorTop = new QHBoxLayout();
    colorTop->addWidget(m_currentColorButton);
    colorTop->addWidget(m_colorInfoLabel, 1);
    colorTop->addWidget(m_colorPickButton);
    colorLayout->addLayout(colorTop);
    auto* fullTop = new QHBoxLayout();
    fullTop->addWidget(new QLabel("Full palette", colorsTab));
    fullTop->addWidget(m_paletteSetCombo);
    fullTop->addStretch(1);
    fullTop->addWidget(m_paletteAssignButton);
    colorLayout->addLayout(fullTop);
    colorLayout->addWidget(m_paletteList);
    colorLayout->addSpacing(8);
    colorLayout->addLayout(reducedLayout);
    colorLayout->addSpacing(8);
    colorLayout->addLayout(dynamicLayout);
    colorsContent->setLayout(colorLayout);
    colorsScrollArea->setWidget(colorsContent);
    auto* colorsTabLayout = new QVBoxLayout(colorsTab);
    colorsTabLayout->addWidget(colorsScrollArea);
    colorsTab->setLayout(colorsTabLayout);

    toolsTabs->addTab(framesTab, "Frames");
    toolsTabs->addTab(spritesTab, "Sprites");
    toolsTabs->addTab(imagesTab, "Images");
    toolsTabs->addTab(masksTab, "Masks");
    toolsTabs->addTab(dynamicMasksTab, "Dynamic Masks");
    toolsTabs->addTab(backgroundsTab, "Backgrounds");
    toolsTabs->addTab(colorsTab, "Colors");
    toolDock->setWidget(toolsTabs);
    addDockWidget(Qt::LeftDockWidgetArea, toolDock);

    auto* inspectorDock = new QDockWidget("Inspector", this);
    inspectorDock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    auto* inspectorWidget = new QWidget(inspectorDock);
    auto* inspectorLayout = new QFormLayout(inspectorWidget);
    m_projectLabel = new QLabel("None", inspectorWidget);
    m_projectLabel->setWordWrap(true);
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
    m_frameBackgroundAssign = new QComboBox(inspectorWidget);
    m_backgroundAssignLabel = new QLabel("Background", inspectorWidget);
    m_shapeCompToggle = new QCheckBox("Shape comparison", inspectorWidget);
    m_hdSourceCombo = new QComboBox(inspectorWidget);
    m_hdScaleCombo = new QComboBox(inspectorWidget);
    m_hdCreateButton = new QPushButton("Create HD", inspectorWidget);
    m_hdDeleteButton = new QPushButton("Delete HD", inspectorWidget);
    m_hdSourceCombo->addItem("Colorized", 0);
    m_hdSourceCombo->addItem("Original", 1);
    m_hdScaleCombo->addItem("Nearest", static_cast<int>(cv::INTER_NEAREST));
    m_hdScaleCombo->addItem("Scale2x", -1);
    m_hdScaleCombo->addItem("Bilinear", static_cast<int>(cv::INTER_LINEAR));
    m_hdScaleCombo->addItem("Bicubic", static_cast<int>(cv::INTER_CUBIC));
    inspectorLayout->addRow("Project", m_projectLabel);
    inspectorLayout->addRow("Bookmarks", m_bookmarksCombo);
    inspectorLayout->addRow("Go to frame", m_frameJump);
    inspectorLayout->addRow("Counts", m_countsLabel);
    inspectorLayout->addRow("Selection", m_selectionLabel);
    inspectorLayout->addRow("Frame info", m_frameMetaLabel);
    inspectorLayout->addRow("Sprite info", m_spriteMetaLabel);
    inspectorLayout->addRow("Mask", m_frameMaskAssign);
    inspectorLayout->addRow("Dynamic mask", m_frameDynamicMaskAssign);
    inspectorLayout->addRow(m_backgroundAssignLabel, m_frameBackgroundAssign);
    inspectorLayout->addRow("Shape compare", m_shapeCompToggle);
    inspectorLayout->addRow("HD source", m_hdSourceCombo);
    inspectorLayout->addRow("HD scale", m_hdScaleCombo);
    inspectorLayout->addRow("HD create", m_hdCreateButton);
    inspectorLayout->addRow("HD delete", m_hdDeleteButton);
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
    m_previewFilterButton = new QToolButton(previewWidget);
    m_previewFilterButton->setText("Filter");
    m_previewFilterButton->setCheckable(true);
    m_previewFilterButton->setToolTip("Filter preview frames by current tool selection");
    previewBar->addWidget(m_previewFilterButton);
    m_previewFilterClearButton = new QToolButton(previewWidget);
    m_previewFilterClearButton->setText("All");
    m_previewFilterClearButton->setToolTip("Show all frames in preview");
    previewBar->addWidget(m_previewFilterClearButton);
    m_previewHdButton = new QToolButton(previewWidget);
    m_previewHdButton->setText("HD");
    m_previewHdButton->setCheckable(true);
    m_previewHdButton->setToolTip("Show only frames with HD data");
    previewBar->addWidget(m_previewHdButton);
    m_previewRefreshButton = new QToolButton(previewWidget);
    m_previewRefreshButton->setText("Refresh");
    m_previewRefreshButton->setToolTip("Refresh preview thumbnails");
    previewBar->addWidget(m_previewRefreshButton);
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
    m_framePreviewList->setUniformItemSizes(false);
    m_framePreviewList->setIconSize(QSize(kPreviewIconWidthHd, kPreviewIconHeightHd));
    m_framePreviewList->setGridSize(QSize());
    m_framePreviewList->setSpacing(1);
    m_framePreviewList->setItemDelegate(new FramePreviewDelegate(m_framePreviewList));
    previewLayout->addWidget(m_framePreviewList);

    previewWidget->setLayout(previewLayout);
    previewDock->setWidget(previewWidget);
    addDockWidget(Qt::BottomDockWidgetArea, previewDock);

    m_coordLabel = new QLabel(this);
    m_coordLabel->setMinimumWidth(140);
    statusBar()->addPermanentWidget(m_coordLabel);

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
    connect(m_shapeCompToggle, &QCheckBox::toggled, this, [this](bool enabled) {
        const int row = m_framesList ? m_framesList->currentRow() : -1;
        if (row < 0 || row >= static_cast<int>(m_frameShapeCompModes.size())) {
            return;
        }
        m_frameShapeCompModes[static_cast<std::size_t>(row)] = enabled ? 1 : 0;
    });
    connect(m_hdCreateButton, &QPushButton::clicked, this, [this]() {
        if (m_canvasTabs && m_canvasTabs->currentWidget() == m_backgroundsCanvas) {
            const int index = m_backgroundList ? m_backgroundList->currentRow() : -1;
            if (index < 0 || !m_backgroundStore) {
                return;
            }
            if (hasHdBackground(index)) {
                statusBar()->showMessage("HD background already exists.", 2000);
                return;
            }
            const cv::Mat* src = m_backgroundStore->at(index);
            if (!src || src->empty()) {
                statusBar()->showMessage("No background image to upscale.", 2000);
                return;
            }
            if (m_backgroundFramesX.size() <= static_cast<std::size_t>(index)) {
                m_backgroundFramesX.resize(static_cast<std::size_t>(index + 1));
            }
            if (m_backgroundExtraFlags.size() <= static_cast<std::size_t>(index)) {
                m_backgroundExtraFlags.resize(static_cast<std::size_t>(index + 1), 0);
            }
            const int interpolation = m_hdScaleCombo ? m_hdScaleCombo->currentData().toInt() : cv::INTER_NEAREST;
            cv::Mat hd;
            if (interpolation == -1) {
                hd = Scale2xBgr(EnsureBgr(*src));
            } else {
                cv::resize(EnsureBgr(*src), hd, cv::Size(src->cols * 2, src->rows * 2), 0.0, 0.0, interpolation);
            }
            if (hd.empty()) {
                statusBar()->showMessage("HD background create failed.", 2000);
                return;
            }
            m_backgroundFramesX[static_cast<std::size_t>(index)] = hd;
            m_backgroundExtraFlags[static_cast<std::size_t>(index)] = 1;
            updateHdControlsForContext();
            refreshBackgroundList();
            updateBackgroundCanvasImage(index);
            refreshFramePreviews();
            updateFrameCanvasImage(m_framesList ? m_framesList->currentRow() : -1);
            statusBar()->showMessage("HD background created.", 2000);
            return;
        }
        const int index = m_framesList ? m_framesList->currentRow() : -1;
        if (index < 0 || index >= m_frameStore->count()) {
            return;
        }
        if (hasHdFrame(index)) {
            statusBar()->showMessage("HD already exists for this frame.", 2000);
            return;
        }
        if (m_frameExtraFrames.size() < static_cast<std::size_t>(m_frameStore->count())) {
            m_frameExtraFrames.resize(static_cast<std::size_t>(m_frameStore->count()));
        }
        if (m_frameExtraFlags.size() < static_cast<std::size_t>(m_frameStore->count())) {
            m_frameExtraFlags.resize(static_cast<std::size_t>(m_frameStore->count()), 0);
        }

        cv::Mat source;
        const int sourceMode = m_hdSourceCombo ? m_hdSourceCombo->currentData().toInt() : 0;
        if (sourceMode == 1) {
            cv::Mat reference = buildOriginalPreviewForIndex(index);
            source = buildOriginalFrame(reference);
        } else {
            if (const cv::Mat* frame = m_frameStore->at(index)) {
                source = EnsureBgr(*frame);
            }
        }
        if (source.empty()) {
            statusBar()->showMessage("HD create failed: missing source frame.", 3000);
            return;
        }

        const int interpolation = m_hdScaleCombo ? m_hdScaleCombo->currentData().toInt() : cv::INTER_NEAREST;
        cv::Mat resized;
        if (interpolation == -1) {
            resized = Scale2xBgr(source);
        } else {
            const cv::Size targetSize(source.cols * 2, source.rows * 2);
            cv::resize(source, resized, targetSize, 0.0, 0.0, interpolation);
            if (interpolation == cv::INTER_LINEAR || interpolation == cv::INTER_CUBIC) {
                resized = AdjustBrightnessToSource(source, resized);
            }
        }
        if (resized.empty()) {
            statusBar()->showMessage("HD create failed: upscale produced empty frame.", 3000);
            return;
        }
        m_frameExtraFrames[static_cast<std::size_t>(index)] = resized;
        m_frameExtraFlags[static_cast<std::size_t>(index)] = 1;
        ensureBackgroundDataSize();
        if (index >= 0 && index < static_cast<int>(m_frameBackgroundMasks.size()) &&
            index < static_cast<int>(m_frameBackgroundMasksX.size())) {
            const cv::Mat& sdMask = m_frameBackgroundMasks[static_cast<std::size_t>(index)];
            cv::Mat& hdMask = m_frameBackgroundMasksX[static_cast<std::size_t>(index)];
            if (!sdMask.empty() && (hdMask.empty() || hdMask.size() != resized.size())) {
                cv::resize(sdMask, hdMask, resized.size(), 0.0, 0.0, cv::INTER_NEAREST);
            }
        }
        if (index < static_cast<int>(m_frameHdUndoStacks.size())) {
            m_frameHdUndoStacks[static_cast<std::size_t>(index)] = UndoStack{};
        }
        setHdMode(true);
        updateFramePreviewAt(index);
        statusBar()->showMessage("HD frame created.", 2000);
    });
    connect(m_hdDeleteButton, &QPushButton::clicked, this, [this]() {
        if (m_canvasTabs && m_canvasTabs->currentWidget() == m_backgroundsCanvas) {
            const int index = m_backgroundList ? m_backgroundList->currentRow() : -1;
            if (index < 0 || index >= static_cast<int>(m_backgroundFramesX.size())) {
                return;
            }
            m_backgroundFramesX[static_cast<std::size_t>(index)] = cv::Mat();
            if (index < static_cast<int>(m_backgroundExtraFlags.size())) {
                m_backgroundExtraFlags[static_cast<std::size_t>(index)] = 0;
            }
            updateHdControlsForContext();
            if (m_useHdBackground) {
                m_useHdBackground = false;
                if (m_backgroundsCanvas) {
                    m_backgroundsCanvas->setHdButtonChecked(false);
                }
            }
            refreshBackgroundList();
            updateBackgroundCanvasImage(index);
            refreshFramePreviews();
            updateFrameCanvasImage(m_framesList ? m_framesList->currentRow() : -1);
            statusBar()->showMessage("HD background deleted.", 2000);
            return;
        }
        const int index = m_framesList ? m_framesList->currentRow() : -1;
        if (index < 0 || index >= static_cast<int>(m_frameExtraFrames.size())) {
            return;
        }
        if (m_frameExtraFrames[static_cast<std::size_t>(index)].empty()) {
            return;
        }
        m_frameExtraFrames[static_cast<std::size_t>(index)] = cv::Mat();
        if (index < static_cast<int>(m_frameExtraFlags.size())) {
            m_frameExtraFlags[static_cast<std::size_t>(index)] = 0;
        }
        if (index < static_cast<int>(m_frameHdUndoStacks.size())) {
            m_frameHdUndoStacks[static_cast<std::size_t>(index)] = UndoStack{};
        }
        setHdMode(false);
        updateFramePreviewAt(index);
        statusBar()->showMessage("HD frame deleted.", 2000);
    });
    connect(m_frameMaskAssign, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
        setCurrentFrameMaskId(m_frameMaskAssign->currentData().toInt());
        updateMaskPreviewForFrame(m_framesList->currentRow());
        if (m_previewFilterEnabled && currentPreviewFilterKind() == PreviewFilterKind::Mask) {
            updatePreviewFilterState();
        }
    });
    connect(m_frameDynamicMaskAssign, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
        setCurrentFrameDynamicMaskId(m_frameDynamicMaskAssign->currentData().toInt());
        syncDynamicSetSelection();
        refreshDynamicPaletteButtons();
        updateMaskPreviewForFrame(m_framesList->currentRow());
        if (m_previewFilterEnabled && currentPreviewFilterKind() == PreviewFilterKind::DynamicMask) {
            updatePreviewFilterState();
        }
    });
    connect(m_frameBackgroundAssign, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
        const int row = m_framesList ? m_framesList->currentRow() : -1;
        if (row < 0) {
            return;
        }
        if (row >= static_cast<int>(m_frameBackgroundIds.size())) {
            m_frameBackgroundIds.resize(static_cast<std::size_t>(m_frameStore->count()), 0xffff);
        }
        const int id = m_frameBackgroundAssign->currentData().toInt();
        m_frameBackgroundIds[static_cast<std::size_t>(row)] = id >= 0 ? static_cast<uint16_t>(id) : 0xffff;
        updateFrameCanvasImage(row);
        updateFramePreviewAt(row);
        if (m_previewFilterEnabled && currentPreviewFilterKind() == PreviewFilterKind::Background) {
            updatePreviewFilterState();
        }
    });
    connect(m_maskList, &QListWidget::currentRowChanged, this, [this](int) {
        updateMaskPreviewForFrame(m_framesList->currentRow());
        if (m_previewFilterEnabled && currentPreviewFilterKind() == PreviewFilterKind::Mask) {
            updatePreviewFilterState();
        }
    });
    connect(m_dynamicMaskList, &QListWidget::currentRowChanged, this, [this](int) {
        syncDynamicSetSelection();
        updateMaskPreviewForFrame(m_framesList->currentRow());
        if (m_previewFilterEnabled && currentPreviewFilterKind() == PreviewFilterKind::DynamicMask) {
            updatePreviewFilterState();
        }
    });
    connect(m_backgroundList, &QListWidget::currentRowChanged, this, [this](int) {
        const int row = m_backgroundList->currentRow();
        if (row >= 0) {
            m_lastBackgroundIndex = row;
            m_backgroundsCanvas->setTitle(QString("Background canvas - BG %1").arg(row));
        }
        showBackgroundAtIndex(row);
        updateUndoActions();
        if (m_previewFilterEnabled && currentPreviewFilterKind() == PreviewFilterKind::Background) {
            updatePreviewFilterState();
        }
        updateHdControlsForContext();
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
            pushMaskUndoSnapshot(MaskMode::Comparison, frameIndex);
            mask->setTo(cv::Scalar(0));
            updateMaskPreviewIcons();
            updateMaskPreviewForFrame(m_framesList->currentRow());
        }
    });
    connect(m_dynamicMaskClearButton, &QPushButton::clicked, this, [this]() {
        const int maskId = m_dynamicMaskList->currentRow();
        if (cv::Mat* mask = activeDynamicMask()) {
            const int frameIndex = m_framesList ? m_framesList->currentRow() : -1;
            pushMaskUndoSnapshot(MaskMode::Dynamic, frameIndex);
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
        if (row < 0 || row >= m_framePreviewList->count()) {
            return;
        }
        const QListWidgetItem* item = m_framePreviewList->item(row);
        if (!item) {
            return;
        }
        const QVariant frameData = item->data(kFrameIndexRole);
        if (!frameData.isValid()) {
            return;
        }
        const int frameIndex = frameData.toInt();
        if (frameIndex < 0 || frameIndex >= m_framesList->count()) {
            return;
        }
        QSignalBlocker blocker(m_framesList);
        m_framesList->setCurrentRow(frameIndex);
        showFrameAtIndex(frameIndex);
        updateMetadataForFrame(frameIndex);
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
                refreshFramePreviewSelection();
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
            if (m_previewFilterEnabled && currentPreviewFilterKind() == PreviewFilterKind::Sprite) {
                updatePreviewFilterState();
            }
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
    connect(m_framesCanvas->canvas(), &GLCanvasWidget::imageHovered, this, [this](int x, int y, bool onImage) {
        if (!onImage) {
            m_frameHoverArea = FrameHoverArea::None;
            updateUndoActions();
            return;
        }
        const int index = m_framesList ? m_framesList->currentRow() : -1;
        if (index < 0) {
            m_frameHoverArea = FrameHoverArea::None;
            updateUndoActions();
            return;
        }
        const cv::Mat* topFrame = activeFrameImage(index, false);
        if (!topFrame || topFrame->empty()) {
            m_frameHoverArea = FrameHoverArea::None;
            updateUndoActions();
            return;
        }
        if (!m_showOriginalFrame) {
            m_frameHoverArea = FrameHoverArea::Top;
            updateUndoActions();
            return;
        }
        cv::Mat reference = buildOriginalPreviewForIndex(index);
        const int bottomWidth = reference.cols > 0 && topFrame->cols == reference.cols * 2 ? topFrame->cols : reference.cols;
        const int bottomHeight = reference.rows > 0 && topFrame->rows == reference.rows * 2 ? topFrame->rows : reference.rows;
        FrameLayout layout = BuildFrameLayout(topFrame->cols, topFrame->rows, bottomWidth, bottomHeight);
        if (y >= 0 && y < layout.topHeight && x >= layout.topX && x < layout.topX + layout.topWidth) {
            m_frameHoverArea = FrameHoverArea::Top;
        } else if (y >= layout.topHeight + kFrameGapPixels &&
                   y < layout.topHeight + kFrameGapPixels + layout.bottomHeight &&
                   x >= layout.bottomX && x < layout.bottomX + layout.bottomWidth) {
            m_frameHoverArea = FrameHoverArea::Bottom;
        } else {
            m_frameHoverArea = FrameHoverArea::None;
        }
        updateUndoActions();
    });
    connect(m_framesCanvas->canvas(), &GLCanvasWidget::imageHovered, this, [this](int x, int y, bool onImage) {
        if (!m_coordLabel) {
            return;
        }
        if (!onImage) {
            m_coordLabel->setText(QString());
            return;
        }
        const int index = m_framesList ? m_framesList->currentRow() : -1;
        const cv::Mat* frame = activeFrameImage(index, false);
        if (!frame || frame->empty()) {
            m_coordLabel->setText(QString());
            return;
        }
        cv::Mat reference = buildOriginalPreviewForIndex(index);
        cv::Mat original = buildOriginalFrame(reference);
        cv::Mat displayOriginal;
        if (m_showOriginalFrame && !original.empty()) {
            displayOriginal = BuildDisplayOriginal(original, frame->size());
        }
        FrameLayout layout = BuildFrameLayout(*frame,
                                              displayOriginal.empty() ? cv::Mat(frame->rows, frame->cols, CV_8UC3)
                                                                      : displayOriginal);
        if (y >= 0 && y < layout.topHeight &&
            x >= layout.topX && x < layout.topX + layout.topWidth) {
            const int localX = x - layout.topX;
            const int localY = y;
            const int frameX = std::clamp(localX, 0, frame->cols - 1) + 1;
            const int frameY = std::clamp(localY, 0, frame->rows - 1) + 1;
            m_coordLabel->setText(QString("Frame %1,%2").arg(frameX).arg(frameY));
            return;
        }
        if (m_showOriginalFrame && !displayOriginal.empty()) {
            const int gap = kFrameGapPixels;
            if (y >= layout.topHeight + gap &&
                y < layout.topHeight + gap + layout.bottomHeight &&
                x >= layout.bottomX && x < layout.bottomX + layout.bottomWidth) {
                const int localX = x - layout.bottomX;
                const int localY = y - (layout.topHeight + gap);
                const int scaleX = (reference.cols > 0) ? (layout.bottomWidth / reference.cols) : 1;
                const int scaleY = (reference.rows > 0) ? (layout.bottomHeight / reference.rows) : 1;
                const int mappedX = (scaleX > 1) ? localX / scaleX : localX;
                const int mappedY = (scaleY > 1) ? localY / scaleY : localY;
                const int refX = std::clamp(mappedX, 0, std::max(0, reference.cols - 1)) + 1;
                const int refY = std::clamp(mappedY, 0, std::max(0, reference.rows - 1)) + 1;
                m_coordLabel->setText(QString("Original %1,%2").arg(refX).arg(refY));
                return;
            }
        }
        m_coordLabel->setText(QString());
    });
    connect(m_spritesCanvas->canvas(), &GLCanvasWidget::imageHovered, this, [this](int x, int y, bool onImage) {
        if (!m_coordLabel) {
            return;
        }
        if (!onImage) {
            m_coordLabel->setText(QString());
            return;
        }
        const int index = m_spritesList ? m_spritesList->currentRow() : -1;
        const cv::Mat* sprite = (index >= 0) ? m_spriteStore->at(index) : nullptr;
        if (!sprite || sprite->empty()) {
            m_coordLabel->setText(QString());
            return;
        }
        const int spriteX = std::clamp(x, 0, sprite->cols - 1) + 1;
        const int spriteY = std::clamp(y, 0, sprite->rows - 1) + 1;
        m_coordLabel->setText(QString("Sprite %1,%2").arg(spriteX).arg(spriteY));
    });
    connect(m_backgroundsCanvas->canvas(), &GLCanvasWidget::imageHovered, this, [this](int x, int y, bool onImage) {
        if (!m_coordLabel) {
            return;
        }
        if (!onImage) {
            m_coordLabel->setText(QString());
            return;
        }
        const int index = m_backgroundList ? m_backgroundList->currentRow() : -1;
        const cv::Mat* bg = activeBackgroundImage(index, false);
        if (!bg || bg->empty()) {
            m_coordLabel->setText(QString());
            return;
        }
        const int bgX = std::clamp(x, 0, bg->cols - 1) + 1;
        const int bgY = std::clamp(y, 0, bg->rows - 1) + 1;
        m_coordLabel->setText(QString("Background %1,%2").arg(bgX).arg(bgY));
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
        if (kind == "background") {
            const int row = m_framesList->currentRow();
            if (index < 0 || index >= m_backgroundList->count()) {
                return;
            }
            if (row >= static_cast<int>(m_frameBackgroundIds.size())) {
                m_frameBackgroundIds.resize(static_cast<std::size_t>(m_frameStore->count()), 0xffff);
            }
            m_frameBackgroundIds[static_cast<std::size_t>(row)] = static_cast<uint16_t>(index);
            if (m_frameBackgroundAssign) {
                m_frameBackgroundAssign->setCurrentIndex(index + 1);
            }
            statusBar()->showMessage(QString("Assigned background %1").arg(index), 2000);
            updateFrameCanvasImage(row);
            updateFramePreviewAt(row);
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
    connect(m_backgroundsCanvas->canvas(), &GLCanvasWidget::imageClicked, this, [this](int x, int y, Qt::MouseButton button) {
        handleBackgroundToolPress(x, y, button);
    });
    connect(m_backgroundsCanvas->canvas(), &GLCanvasWidget::imageDragged, this, [this](int x, int y, Qt::MouseButtons buttons) {
        handleBackgroundToolDrag(x, y, buttons);
    });
    connect(m_backgroundsCanvas->canvas(), &GLCanvasWidget::imageReleased, this, [this](int x, int y, Qt::MouseButton button) {
        handleBackgroundToolRelease(x, y, button);
    });

    connect(m_framesCanvas, &CanvasWidget::fitRequested, this, [this]() {
        m_framesCanvas->canvas()->requestFitOnResize(true);
    });
    connect(m_spritesCanvas, &CanvasWidget::fitRequested, this, [this]() {
        m_spritesCanvas->canvas()->requestFitOnResize(true);
    });
    connect(m_backgroundsCanvas, &CanvasWidget::fitRequested, this, [this]() {
        m_backgroundsCanvas->canvas()->requestFitOnResize(true);
    });
    connect(m_framesCanvas, &CanvasWidget::gridToggled, this, [this](bool enabled) {
        m_framesCanvas->canvas()->setGridEnabled(enabled);
    });
    connect(m_spritesCanvas, &CanvasWidget::gridToggled, this, [this](bool enabled) {
        m_spritesCanvas->canvas()->setGridEnabled(enabled);
    });
    connect(m_backgroundsCanvas, &CanvasWidget::gridToggled, this, [this](bool enabled) {
        m_backgroundsCanvas->canvas()->setGridEnabled(enabled);
    });
    connect(m_framesCanvas, &CanvasWidget::maskToggled, this, [this](bool enabled) {
        if (enabled) {
            setMaskMode(MaskMode::Comparison);
        } else if (m_maskMode == MaskMode::Comparison) {
            setMaskMode(MaskMode::None);
        }
    });
    connect(m_framesCanvas, &CanvasWidget::dynamicToggled, this, [this](bool enabled) {
        if (enabled) {
            setMaskMode(MaskMode::Dynamic);
        } else if (m_maskMode == MaskMode::Dynamic) {
            setMaskMode(MaskMode::None);
        }
    });
    connect(m_framesCanvas, &CanvasWidget::backgroundMaskToggled, this, [this](bool enabled) {
        m_backgroundMaskMode = enabled;
        updateFrameCanvasImage(m_framesList->currentRow());
        updateMaskPreviewForFrame(m_framesList->currentRow());
        updateUndoActions();
    });
    connect(m_framesCanvas, &CanvasWidget::backgroundToggled, this, [this](bool enabled) {
        m_showBackgroundLayer = enabled;
        updateFrameCanvasImage(m_framesList->currentRow());
        updateMaskPreviewForFrame(m_framesList->currentRow());
    });
    connect(m_framesCanvas, &CanvasWidget::originalToggled, this, [this](bool enabled) {
        m_showOriginalFrame = enabled;
        updateFrameCanvasImage(m_framesList->currentRow());
        updateMaskPreviewForFrame(m_framesList->currentRow());
    });
    connect(m_framesCanvas, &CanvasWidget::hdToggled, this, [this](bool enabled) {
        setHdMode(enabled);
    });
    connect(m_backgroundsCanvas, &CanvasWidget::hdToggled, this, [this](bool enabled) {
        m_useHdBackground = enabled;
        updateBackgroundCanvasImage(m_backgroundList ? m_backgroundList->currentRow() : -1);
        updateHdControlsForContext();
    });

    connect(m_frameFilter, &QLineEdit::textChanged, this, [this](const QString& text) {
        applyFrameFilter(text);
    });
    connect(m_spriteFilter, &QLineEdit::textChanged, this, [this](const QString& text) {
        applySpriteFilter(text);
    });

    connect(m_previewFilterButton, &QToolButton::toggled, this, [this](bool enabled) {
        m_previewFilterEnabled = enabled;
        updatePreviewFilterState();
    });
    connect(m_previewFilterClearButton, &QToolButton::clicked, this, [this]() {
        m_previewFilterEnabled = false;
        if (m_previewFilterButton) {
            m_previewFilterButton->setChecked(false);
        }
        updatePreviewFilterState();
    });
    connect(m_previewHdButton, &QToolButton::toggled, this, [this](bool enabled) {
        m_previewHdOnly = enabled;
        refreshFramePreviews();
    });
    connect(m_previewRefreshButton, &QToolButton::clicked, this, [this]() {
        refreshAllPreviews();
    });
    initPalette();
    refreshPaletteList();
    refreshReducedPaletteUI();
    refreshDynamicPaletteUI();
    setDrawColor(QColor(255, 255, 255), true);
    if (m_paletteAssignButton) {
        m_paletteAssignButton->setEnabled(false);
    }
    connect(m_paletteList, &QListWidget::currentRowChanged, this, [this](int row) {
        if (row >= 0 && row < m_paletteColors.size()) {
            m_currentPaletteIndex = row;
            if (m_paletteList) {
                const QColor color = m_paletteSelectionIsReference ? QColor(70, 150, 255) : QColor(255, 140, 0);
                m_paletteList->setProperty("selectionColor", color);
                m_paletteList->viewport()->update();
            }
            if (m_reducedPaletteList) {
                QSignalBlocker blocker(m_reducedPaletteList);
                m_reducedPaletteList->setCurrentRow(-1);
                m_reducedSlotIndex = -1;
            }
            if (m_dynamicPaletteList) {
                QSignalBlocker blocker(m_dynamicPaletteList);
                m_dynamicPaletteList->setCurrentRow(-1);
                m_dynamicSlotIndex = -1;
            }
            m_paletteSelectionIsReference = false;
            setDrawColor(m_paletteColors[row], false);
            if (m_paletteAssignButton) {
                m_paletteAssignButton->setEnabled(true);
            }
        } else if (m_paletteAssignButton) {
            m_paletteAssignButton->setEnabled(false);
        }
    });
    connect(m_paletteList, &QListWidget::itemClicked, this, [this](QListWidgetItem*) {
        m_paletteSelectionIsReference = false;
        if (m_paletteList) {
            m_paletteList->setProperty("selectionColor", QColor(255, 140, 0));
            m_paletteList->viewport()->update();
        }
    });
    connect(m_paletteList, &QListWidget::itemDoubleClicked, this, [this](QListWidgetItem* item) {
        if (!item) {
            return;
        }
        const int row = item->data(Qt::UserRole).toInt();
        if (row < 0 || row >= m_paletteColors.size()) {
            return;
        }
        const QColor picked = QColorDialog::getColor(m_paletteColors[row], this, "Select palette color");
        if (!picked.isValid()) {
            return;
        }
        const cv::Vec3b quant = Rgb565ToBgr(BgrToRgb565(cv::Vec3b(picked.blue(), picked.green(), picked.red())));
        m_paletteColors[row] = QColor(quant[2], quant[1], quant[0]);
        if (m_paletteSetIndex >= 0 && m_paletteSetIndex < m_fullPalettes.size()) {
            m_fullPalettes[m_paletteSetIndex] = m_paletteColors;
        }
        refreshPaletteList();
        m_paletteList->setCurrentRow(row);
    });
    connect(m_colorPickButton, &QPushButton::clicked, this, [this]() {
        const QColor current(static_cast<int>(m_drawColor[2]),
                             static_cast<int>(m_drawColor[1]),
                             static_cast<int>(m_drawColor[0]));
        const QColor picked = QColorDialog::getColor(current, this, "Select draw color");
        if (!picked.isValid()) {
            return;
        }
        setDrawColor(picked, true);
    });
    connect(m_currentColorButton, &QToolButton::clicked, this, [this]() {
        const QColor current(static_cast<int>(m_drawColor[2]),
                             static_cast<int>(m_drawColor[1]),
                             static_cast<int>(m_drawColor[0]));
        const QColor picked = QColorDialog::getColor(current, this, "Select draw color");
        if (!picked.isValid()) {
            return;
        }
        setDrawColor(picked, true);
    });
    connect(m_paletteAssignButton, &QPushButton::clicked, this, [this]() {
        if (m_currentPaletteIndex < 0 || m_currentPaletteIndex >= m_paletteColors.size()) {
            return;
        }
        const QColor current(static_cast<int>(m_drawColor[2]),
                             static_cast<int>(m_drawColor[1]),
                             static_cast<int>(m_drawColor[0]));
        m_paletteColors[m_currentPaletteIndex] = current;
        if (m_paletteSetIndex >= 0 && m_paletteSetIndex < m_fullPalettes.size()) {
            m_fullPalettes[m_paletteSetIndex] = m_paletteColors;
        }
        refreshPaletteList();
        m_paletteList->setCurrentRow(m_currentPaletteIndex);
    });
    connect(m_paletteSetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        if (index < 0 || index >= m_fullPalettes.size()) {
            return;
        }
        m_paletteSetIndex = index;
        m_paletteColors = m_fullPalettes[m_paletteSetIndex];
        m_currentPaletteIndex = 0;
        refreshPaletteList();
        refreshReducedPaletteButtons();
        if (!m_paletteColors.isEmpty()) {
            setDrawColor(m_paletteColors.front(), true);
        }
    });
    connect(m_reducedSetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        if (index < 0 || index >= kReducedPaletteCount) {
            return;
        }
        m_reducedPaletteIndex = index;
        refreshReducedPaletteButtons();
    });
    connect(m_dynamicSetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        if (index < 0 || index >= MAX_DYNA_SETS_PER_FRAMEN) {
            return;
        }
        m_dynamicSetIndex = index;
        refreshDynamicPaletteButtons();
    });
    connect(m_reducedAssignButton, &QPushButton::clicked, this, [this]() {
        if (m_reducedSlotIndex < 0 || m_reducedSlotIndex >= 16) {
            return;
        }
        const QColor current(static_cast<int>(m_drawColor[2]),
                             static_cast<int>(m_drawColor[1]),
                             static_cast<int>(m_drawColor[0]));
        setReducedSlotColor(m_reducedPaletteIndex, m_reducedSlotIndex, current);
        refreshReducedPaletteButtons();
        setDrawColor(current, true);
    });
    connect(m_dynamicAssignButton, &QPushButton::clicked, this, [this]() {
        if (m_dynamicSlotIndex < 0 || m_dynamicSlotIndex >= 16) {
            return;
        }
        const int frameIndex = m_framesList ? m_framesList->currentRow() : -1;
        if (frameIndex < 0 || frameIndex >= static_cast<int>(m_frameDynamicColors.size())) {
            return;
        }
        const QColor current(static_cast<int>(m_drawColor[2]),
                             static_cast<int>(m_drawColor[1]),
                             static_cast<int>(m_drawColor[0]));
        setDynamicSlotColor(frameIndex, m_dynamicSetIndex, m_dynamicSlotIndex, current);
        refreshDynamicPaletteButtons();
        setDrawColor(current, true);
    });
    connect(m_toolsTabs, &QTabWidget::currentChanged, this, [this](int) {
        if (m_previewFilterEnabled) {
            updatePreviewFilterState();
        }
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
            m_backgroundStore->clear();
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
            m_frameExtraFrames.clear();
            m_frameExtraFlags.clear();
            m_backgroundFramesX.clear();
            m_backgroundExtraFlags.clear();
            m_frameBackgroundIds.clear();
            m_frameBackgroundMasks.clear();
            m_frameBackgroundMasksX.clear();
            m_useHdFrame = false;
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
        m_backgroundStore->clear();
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
        m_frameExtraFrames.clear();
        m_frameExtraFlags.clear();
        m_backgroundFramesX.clear();
        m_backgroundExtraFlags.clear();
        m_frameBackgroundIds.clear();
        m_frameBackgroundMasks.clear();
        m_frameBackgroundMasksX.clear();
        m_useHdFrame = false;
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
        m_frameExtraFrames = legacy.frames_x;
        m_frameExtraFlags = legacy.frame_extra_flags;
        m_backgroundExtraFlags = legacy.background_extra_flags;
        m_frameBackgroundIds = legacy.background_ids;
        m_frameBackgroundMasks = legacy.background_masks;
        m_frameBackgroundMasksX = legacy.background_masks_x;
        m_backgroundFramesX = legacy.background_frames_x;
        for (const auto& bg : legacy.background_frames) {
            m_backgroundStore->add(bg);
        }
        if (m_backgroundFramesX.size() < static_cast<std::size_t>(m_backgroundStore->count())) {
            m_backgroundFramesX.resize(static_cast<std::size_t>(m_backgroundStore->count()));
        }
        if (m_frameExtraFrames.size() < legacy.frames.size()) {
            m_frameExtraFrames.resize(legacy.frames.size());
        }
        if (m_frameExtraFlags.size() < legacy.frames.size()) {
            m_frameExtraFlags.resize(legacy.frames.size(), 0);
        }
        if (m_frameBackgroundIds.size() < legacy.frames.size()) {
            m_frameBackgroundIds.resize(legacy.frames.size(), 0xffff);
        }
        loadPaletteFromProject(legacy);

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
    m_backgroundStore->clear();
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
    m_frameExtraFrames.clear();
    m_frameExtraFlags.clear();
    m_backgroundFramesX.clear();
    m_backgroundExtraFlags.clear();
    m_frameBackgroundIds.clear();
    m_frameBackgroundMasks.clear();
    m_frameBackgroundMasksX.clear();
    m_useHdFrame = false;
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
    project.frames_x = m_frameExtraFrames;
    project.frame_extra_flags.assign(static_cast<std::size_t>(frameCount), 0);
    for (int i = 0; i < frameCount; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i);
        if (idx < m_frameExtraFrames.size() && !m_frameExtraFrames[idx].empty()) {
            project.frame_extra_flags[idx] = 1;
        } else if (idx < m_frameExtraFlags.size()) {
            project.frame_extra_flags[idx] = m_frameExtraFlags[idx];
        }
    }
    project.background_frames.reserve(static_cast<std::size_t>(m_backgroundStore ? m_backgroundStore->count() : 0));
    if (m_backgroundStore) {
        for (int i = 0; i < m_backgroundStore->count(); ++i) {
            if (const cv::Mat* bg = m_backgroundStore->at(i)) {
                project.background_frames.push_back(bg->clone());
            }
        }
    }
    project.background_frames_x = m_backgroundFramesX;
    project.background_extra_flags = m_backgroundExtraFlags;
    project.background_ids = m_frameBackgroundIds;
    project.background_masks = m_frameBackgroundMasks;
    project.background_masks_x = m_frameBackgroundMasksX;
    project.active_reduced_palette = static_cast<uint8_t>(std::max(0, std::min(m_reducedPaletteIndex, kReducedPaletteCount - 1)));
    project.preview_reduced_palette = project.active_reduced_palette;
    project.reduced_palettes = m_reducedPaletteIndices;
    project.reduced_palette_names = m_reducedPaletteNames;
    project.palettes.clear();
    project.palettes.resize(static_cast<std::size_t>(N_PALETTES * 64), 0);
    for (int p = 0; p < N_PALETTES && p < m_fullPalettes.size(); ++p) {
        const QVector<QColor>& colors = m_fullPalettes[p];
        for (int i = 0; i < 64 && i < colors.size(); ++i) {
            const QColor color = colors[i];
            const cv::Vec3b bgr(color.blue(), color.green(), color.red());
            project.palettes[static_cast<std::size_t>(p) * 64 + static_cast<std::size_t>(i)] = BgrToRgb565(bgr);
        }
    }
    project.palette_names.clear();
    project.palette_names.reserve(N_PALETTES);
    for (int i = 0; i < N_PALETTES && i < m_paletteNames.size(); ++i) {
        project.palette_names.push_back(m_paletteNames[i].toStdString());
    }

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

void MainWindow::refreshFramePreviewSelection()
{
    if (!m_framePreviewList || !m_framesList) {
        return;
    }
    const int row = previewRowForFrame(m_framesList->currentRow());
    if (row >= 0) {
        m_framePreviewList->setCurrentRow(row);
    } else {
        m_framePreviewList->setCurrentRow(-1);
    }
}

int MainWindow::previewRowForFrame(int index) const
{
    if (!m_framePreviewList || index < 0) {
        return -1;
    }
    for (int row = 0; row < m_framePreviewList->count(); ++row) {
        const QListWidgetItem* item = m_framePreviewList->item(row);
        if (!item) {
            continue;
        }
        if (item->data(kFrameIndexRole).toInt() == index) {
            return row;
        }
    }
    return -1;
}

MainWindow::PreviewFilterKind MainWindow::currentPreviewFilterKind() const
{
    if (!m_toolsTabs) {
        return PreviewFilterKind::None;
    }
    const QWidget* current = m_toolsTabs->currentWidget();
    if (current == m_masksTab) {
        return PreviewFilterKind::Mask;
    }
    if (current == m_dynamicMasksTab) {
        return PreviewFilterKind::DynamicMask;
    }
    if (current == m_backgroundsTab) {
        return PreviewFilterKind::Background;
    }
    if (current == m_spritesTab) {
        return PreviewFilterKind::Sprite;
    }
    return PreviewFilterKind::None;
}

void MainWindow::updatePreviewFilterState()
{
    if (!m_previewFilterButton) {
        return;
    }
    if (!m_previewFilterEnabled) {
        m_previewFilterKind = PreviewFilterKind::None;
        m_previewFilterButton->setText("Filter");
        refreshFramePreviews();
        return;
    }
    m_previewFilterKind = currentPreviewFilterKind();
    QString label("Filter");
    switch (m_previewFilterKind) {
        case PreviewFilterKind::Mask:
            label = "Filter: Mask";
            if (m_maskList && m_maskList->currentRow() < 0) {
                statusBar()->showMessage("Select a mask to filter frames.", 2000);
            }
            break;
        case PreviewFilterKind::DynamicMask:
            label = "Filter: Dynamic";
            if (m_dynamicMaskList && m_dynamicMaskList->currentRow() < 0) {
                statusBar()->showMessage("Select a dynamic mask to filter frames.", 2000);
            }
            break;
        case PreviewFilterKind::Background:
            label = "Filter: Background";
            if (m_backgroundList && m_backgroundList->currentRow() < 0) {
                statusBar()->showMessage("Select a background to filter frames.", 2000);
            }
            break;
        case PreviewFilterKind::Sprite:
            label = "Filter: Sprite";
            statusBar()->showMessage("Sprite assignments are not available yet.", 2500);
            break;
        case PreviewFilterKind::None:
            break;
    }
    m_previewFilterButton->setText(label);
    refreshFramePreviews();
}

std::vector<int> MainWindow::buildPreviewFrameIndices() const
{
    std::vector<int> indices;
    const int count = m_frameStore ? m_frameStore->count() : 0;
    if (count <= 0) {
        return indices;
    }
    if (!m_previewFilterEnabled || m_previewFilterKind == PreviewFilterKind::None) {
        indices.reserve(static_cast<std::size_t>(count));
        for (int i = 0; i < count; ++i) {
            indices.push_back(i);
        }
        if (m_previewHdOnly) {
            std::vector<int> filtered;
            filtered.reserve(indices.size());
            for (int index : indices) {
                if (hasHdFrame(index)) {
                    filtered.push_back(index);
                }
            }
            indices.swap(filtered);
        }
        return indices;
    }
    switch (m_previewFilterKind) {
        case PreviewFilterKind::Mask: {
            const int selected = m_maskList ? m_maskList->currentRow() : -1;
            if (selected < 0) {
                break;
            }
            for (int i = 0; i < count && i < static_cast<int>(m_frameCompMaskIds.size()); ++i) {
                if (m_frameCompMaskIds[static_cast<std::size_t>(i)] == selected) {
                    indices.push_back(i);
                }
            }
            break;
        }
        case PreviewFilterKind::DynamicMask: {
            const int selected = m_dynamicMaskList ? m_dynamicMaskList->currentRow() : -1;
            if (selected < 0) {
                break;
            }
            for (int i = 0; i < count && i < static_cast<int>(m_frameDynamicMaskIds.size()); ++i) {
                if (m_frameDynamicMaskIds[static_cast<std::size_t>(i)] == selected) {
                    indices.push_back(i);
                }
            }
            break;
        }
        case PreviewFilterKind::Background: {
            const int selected = m_backgroundList ? m_backgroundList->currentRow() : -1;
            if (selected < 0) {
                break;
            }
            for (int i = 0; i < count && i < static_cast<int>(m_frameBackgroundIds.size()); ++i) {
                if (m_frameBackgroundIds[static_cast<std::size_t>(i)] == selected) {
                    indices.push_back(i);
                }
            }
            break;
        }
        case PreviewFilterKind::Sprite:
            for (int i = 0; i < count; ++i) {
                indices.push_back(i);
            }
            break;
        case PreviewFilterKind::None:
            break;
    }
    if (m_previewHdOnly && !indices.empty()) {
        std::vector<int> filtered;
        filtered.reserve(indices.size());
        for (int index : indices) {
            if (hasHdFrame(index)) {
                filtered.push_back(index);
            }
        }
        indices.swap(filtered);
    }
    return indices;
}

void MainWindow::refreshAllPreviews()
{
    if (m_maskList) {
        updateMaskPreviewIcons();
    }
    if (m_dynamicMaskList) {
        updateDynamicMaskPreviewIcons();
    }
    int backgroundRow = m_backgroundList ? m_backgroundList->currentRow() : -1;
    if (backgroundRow < 0) {
        backgroundRow = m_lastBackgroundIndex;
    }
    refreshBackgroundList();
    if (m_backgroundList && backgroundRow >= 0 && backgroundRow < m_backgroundList->count()) {
        m_backgroundList->setCurrentRow(backgroundRow);
    }
    refreshFramePreviews();
    updateMaskPreviewForFrame(m_framesList ? m_framesList->currentRow() : -1);
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

    const std::vector<int> indices = buildPreviewFrameIndices();
    if (indices.empty()) {
        auto* item = new QListWidgetItem(m_previewFilterEnabled ? "No frames match filter" : "No frames");
        item->setFlags(item->flags() & ~Qt::ItemIsEnabled);
        m_framePreviewList->addItem(item);
        return;
    }

    for (int i : indices) {
        const cv::Mat* image = m_frameStore->at(i);
        if (!image || image->empty()) {
            continue;
        }

        cv::Mat reference = buildOriginalPreviewForIndex(i);
        cv::Mat hdFrame;
        if (i >= 0 && i < static_cast<int>(m_frameExtraFrames.size())) {
            hdFrame = m_frameExtraFrames[static_cast<std::size_t>(i)];
        }
        cv::Mat composed = applyBackgroundComposite(i, *image, false);
        cv::Mat hdComposed;
        if (!hdFrame.empty()) {
            hdComposed = applyBackgroundComposite(i, hdFrame, true);
        }
        cv::Mat previewMat = buildPreviewFrame(composed, reference, hdComposed);
        cv::Mat rgb;
        cv::cvtColor(previewMat, rgb, cv::COLOR_BGR2RGB);
        QImage previewImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
        const QSize iconSize = hasHdFrame(i)
            ? QSize(kPreviewIconWidthHd, kPreviewIconHeightHd)
            : QSize(kPreviewIconWidth, kPreviewIconHeight);
        QPixmap pixmap = QPixmap::fromImage(previewImage.copy());
        pixmap = pixmap.scaled(iconSize, Qt::KeepAspectRatio, Qt::FastTransformation);

        auto* item = new QListWidgetItem();
        item->setIcon(QIcon(pixmap));
        item->setText(QString());
        item->setData(kPreviewIconSizeRole, iconSize);
        item->setSizeHint(PreviewItemSizeForIcon(iconSize, m_framePreviewList->font()));
        item->setData(kFrameIndexRole, i);
        int duration = 30;
        if (i < static_cast<int>(m_frameDurations.size()) && m_frameDurations[i] > 0) {
            duration = static_cast<int>(m_frameDurations[i]);
        }
        item->setData(kFrameDurationRole, duration);
        item->setToolTip(QString("Frame %1 (%2 ms)").arg(i).arg(duration));
        m_framePreviewList->addItem(item);
    }

    refreshFramePreviewSelection();
    m_framePreviewList->doItemsLayout();
}

void MainWindow::updateFramePreviewAt(int index)
{
    const int row = previewRowForFrame(index);
    if (row < 0 || row >= m_framePreviewList->count()) {
        return;
    }
    const cv::Mat* image = m_frameStore->at(index);
    if (!image || image->empty()) {
        return;
    }
    QListWidgetItem* item = m_framePreviewList->item(row);
    if (!item) {
        return;
    }
    cv::Mat reference = buildOriginalPreviewForIndex(index);
    cv::Mat hdFrame;
    if (index >= 0 && index < static_cast<int>(m_frameExtraFrames.size())) {
        hdFrame = m_frameExtraFrames[static_cast<std::size_t>(index)];
    }
    cv::Mat composed = applyBackgroundComposite(index, *image, false);
    cv::Mat hdComposed;
    if (!hdFrame.empty()) {
        hdComposed = applyBackgroundComposite(index, hdFrame, true);
    }
    cv::Mat previewMat = buildPreviewFrame(composed, reference, hdComposed);
    cv::Mat rgb;
    cv::cvtColor(previewMat, rgb, cv::COLOR_BGR2RGB);
    QImage previewImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
    const QSize iconSize = hasHdFrame(index)
        ? QSize(kPreviewIconWidthHd, kPreviewIconHeightHd)
        : QSize(kPreviewIconWidth, kPreviewIconHeight);
    QPixmap pixmap = QPixmap::fromImage(previewImage.copy());
    pixmap = pixmap.scaled(iconSize, Qt::KeepAspectRatio, Qt::FastTransformation);
    item->setIcon(QIcon(pixmap));
    item->setData(kPreviewIconSizeRole, iconSize);
    item->setSizeHint(PreviewItemSizeForIcon(iconSize, m_framePreviewList->font()));
    m_framePreviewList->doItemsLayout();
}

cv::Mat MainWindow::buildPreviewFrame(const cv::Mat& colorized,
                                      const cv::Mat& reference,
                                      const cv::Mat& hdFrame) const
{
    cv::Mat color = EnsureBgr(colorized);
    cv::Mat hd = EnsureBgr(hdFrame);
    cv::Mat original = buildOriginalFrame(reference);
    if (color.empty() && hd.empty() && original.empty()) {
        return cv::Mat();
    }
    const QColor gap = m_framePreviewList
        ? m_framePreviewList->palette().color(QPalette::Window)
        : QApplication::palette().color(QPalette::Window);
    return BuildStackedFrames({color, original, hd}, cv::Scalar(gap.blue(), gap.green(), gap.red()));
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
    return BuildStackedFrames({top, bottom}, gapColor);
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
    if (!previewOriginal.empty() && previewOriginal.size() != colorized.size()) {
        cv::resize(previewOriginal, previewOriginal, colorized.size(), 0.0, 0.0, cv::INTER_NEAREST);
    }
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

cv::Mat MainWindow::buildReferenceForSize(int index, const cv::Size& target) const
{
    cv::Mat ref = buildOriginalPreviewForIndex(index);
    if (ref.empty()) {
        return cv::Mat();
    }
    if (ref.size() == target) {
        return ref;
    }
    cv::Mat resized;
    cv::resize(ref, resized, target, 0.0, 0.0, cv::INTER_NEAREST);
    return resized;
}

cv::Mat MainWindow::applyBackgroundComposite(int index, const cv::Mat& frame, bool useHd) const
{
    if (index < 0 || frame.empty()) {
        return frame.clone();
    }
    if (index >= static_cast<int>(m_frameBackgroundIds.size())) {
        return EnsureBgr(frame);
    }
    const uint16_t bgId = m_frameBackgroundIds[static_cast<std::size_t>(index)];
    if (bgId == 0xffff) {
        return EnsureBgr(frame);
    }
    const cv::Mat* bg = nullptr;
    if (useHd) {
        if (bgId < m_backgroundFramesX.size()) {
            bg = &m_backgroundFramesX[bgId];
        }
        if ((!bg || bg->empty()) && m_backgroundStore && bgId < m_backgroundStore->count()) {
            bg = m_backgroundStore->at(static_cast<int>(bgId));
        }
    } else if (m_backgroundStore && bgId < m_backgroundStore->count()) {
        bg = m_backgroundStore->at(static_cast<int>(bgId));
    }
    if (!bg || bg->empty()) {
        return EnsureBgr(frame);
    }

    const cv::Mat* mask = nullptr;
    const cv::Mat* sdMask = nullptr;
    if (useHd) {
        if (index < static_cast<int>(m_frameBackgroundMasksX.size())) {
            mask = &m_frameBackgroundMasksX[static_cast<std::size_t>(index)];
        }
        if (index < static_cast<int>(m_frameBackgroundMasks.size())) {
            sdMask = &m_frameBackgroundMasks[static_cast<std::size_t>(index)];
        }
        if ((!mask || mask->empty() || !MaskHasContent(*mask)) && sdMask && MaskHasContent(*sdMask)) {
            mask = sdMask;
        }
    } else if (index < static_cast<int>(m_frameBackgroundMasks.size())) {
        mask = &m_frameBackgroundMasks[static_cast<std::size_t>(index)];
    }
    if (!mask || mask->empty()) {
        return EnsureBgr(frame);
    }

    cv::Mat output = EnsureBgr(frame);
    cv::Mat ref = buildReferenceForSize(index, output.size());
    if (ref.empty()) {
        return output;
    }
    cv::Mat bgFrame = EnsureBgr(*bg);
    if (bgFrame.size() != output.size()) {
        cv::resize(bgFrame, bgFrame, output.size(), 0.0, 0.0, cv::INTER_NEAREST);
    }
    cv::Mat maskScaled = *mask;
    if (maskScaled.size() != output.size()) {
        cv::resize(maskScaled, maskScaled, output.size(), 0.0, 0.0, cv::INTER_NEAREST);
    }

    for (int y = 0; y < output.rows; ++y) {
        cv::Vec3b* row = output.ptr<cv::Vec3b>(y);
        const cv::Vec3b* brow = bgFrame.ptr<cv::Vec3b>(y);
        const uint8_t* mrow = maskScaled.ptr<uint8_t>(y);
        const uint8_t* rrow = ref.ptr<uint8_t>(y);
        for (int x = 0; x < output.cols; ++x) {
            const cv::Vec3b current = row[x];
            const bool hasColor = current[0] != 0 || current[1] != 0 || current[2] != 0;
            if (mrow[x] && rrow[x] == 0 && !hasColor) {
                row[x] = brow[x];
            }
        }
    }
    return output;
}

cv::Mat MainWindow::applyBackgroundCompositeWithMask(int index,
                                                     const cv::Mat& frame,
                                                     const cv::Mat& mask,
                                                     bool useHd) const
{
    if (index < 0 || frame.empty() || mask.empty()) {
        return EnsureBgr(frame);
    }
    if (index >= static_cast<int>(m_frameBackgroundIds.size())) {
        return EnsureBgr(frame);
    }
    const uint16_t bgId = m_frameBackgroundIds[static_cast<std::size_t>(index)];
    if (bgId == 0xffff) {
        return EnsureBgr(frame);
    }
    const cv::Mat* bg = nullptr;
    if (useHd) {
        if (bgId < m_backgroundFramesX.size()) {
            bg = &m_backgroundFramesX[bgId];
        }
    } else if (m_backgroundStore && bgId < m_backgroundStore->count()) {
        bg = m_backgroundStore->at(static_cast<int>(bgId));
    }
    if (!bg || bg->empty()) {
        return EnsureBgr(frame);
    }
    cv::Mat output = EnsureBgr(frame);
    cv::Mat ref = buildReferenceForSize(index, output.size());
    if (ref.empty()) {
        return output;
    }
    cv::Mat bgFrame = EnsureBgr(*bg);
    if (bgFrame.size() != output.size()) {
        cv::resize(bgFrame, bgFrame, output.size(), 0.0, 0.0, cv::INTER_NEAREST);
    }
    cv::Mat maskScaled = mask;
    if (maskScaled.size() != output.size()) {
        cv::resize(maskScaled, maskScaled, output.size(), 0.0, 0.0, cv::INTER_NEAREST);
    }
    for (int y = 0; y < output.rows; ++y) {
        cv::Vec3b* row = output.ptr<cv::Vec3b>(y);
        const cv::Vec3b* brow = bgFrame.ptr<cv::Vec3b>(y);
        const uint8_t* mrow = maskScaled.ptr<uint8_t>(y);
        const uint8_t* rrow = ref.ptr<uint8_t>(y);
        for (int x = 0; x < output.cols; ++x) {
        const cv::Vec3b current = row[x];
        const bool hasColor = current[0] != 0 || current[1] != 0 || current[2] != 0;
        if (mrow[x] && rrow[x] == 0 && !hasColor) {
            row[x] = brow[x];
        }
        }
    }
    return output;
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
    const cv::Mat* image = activeFrameImage(index, false);
    if (!image || image->empty()) {
        m_framesCanvas->setImage(cv::Mat());
        return;
    }
    cv::Mat composed = m_showBackgroundLayer
        ? applyBackgroundComposite(index, *image, m_useHdFrame)
        : EnsureBgr(*image);
    cv::Mat reference = buildOriginalPreviewForIndex(index);
    cv::Mat original = buildOriginalFrame(reference);
    if (original.empty() || !m_showOriginalFrame) {
        m_framesCanvas->setImage(composed);
        m_framesCanvas->canvas()->setGridSegments(0, 0, 0);
        m_framesCanvas->canvas()->setGridScales(1, 1);
        return;
    }
    cv::Mat displayOriginal = BuildDisplayOriginal(original, composed.size());
    const QColor gap = m_framesCanvas->palette().color(QPalette::Window);
    cv::Mat combined = buildCombinedFrame(composed, displayOriginal, cv::Scalar(gap.blue(), gap.green(), gap.red()));
    m_framesCanvas->setImage(combined);
    int bottomScale = 1;
    if (!displayOriginal.empty() && original.rows > 0 && original.cols > 0 &&
        displayOriginal.rows == original.rows * 2 && displayOriginal.cols == original.cols * 2) {
        bottomScale = 2;
    }
    m_framesCanvas->canvas()->setGridSegments(composed.rows, kFrameGapPixels, displayOriginal.rows);
    m_framesCanvas->canvas()->setGridScales(1, bottomScale);
}

void MainWindow::updateBackgroundCanvasImage(int index)
{
    if (!m_backgroundsCanvas || !m_backgroundStore) {
        return;
    }
    const cv::Mat* image = activeBackgroundImage(index, false);
    if (!image || image->empty()) {
        m_backgroundsCanvas->setImage(cv::Mat());
        return;
    }
    m_backgroundsCanvas->setImage(*image);
}

bool MainWindow::hasHdBackground(int index) const
{
    if (index < 0 || index >= static_cast<int>(m_backgroundFramesX.size())) {
        return false;
    }
    return !IsAllBlackFrame(m_backgroundFramesX[static_cast<std::size_t>(index)]);
}

void MainWindow::updateHdControlsForContext()
{
    if (!m_hdCreateButton || !m_hdDeleteButton || !m_hdScaleCombo || !m_hdSourceCombo) {
        return;
    }
    const bool inBackgrounds = m_canvasTabs && m_canvasTabs->currentWidget() == m_backgroundsCanvas;
    if (inBackgrounds) {
        const int bgIndex = m_backgroundList ? m_backgroundList->currentRow() : -1;
        m_hdSourceCombo->setEnabled(false);
        m_hdScaleCombo->setEnabled(bgIndex >= 0);
        m_hdCreateButton->setEnabled(bgIndex >= 0 && !hasHdBackground(bgIndex));
        m_hdDeleteButton->setEnabled(bgIndex >= 0 && hasHdBackground(bgIndex));
    } else {
        const int frameIndex = m_framesList ? m_framesList->currentRow() : -1;
        const bool hasHd = hasHdFrame(frameIndex);
        m_hdSourceCombo->setEnabled(frameIndex >= 0);
        m_hdScaleCombo->setEnabled(frameIndex >= 0);
        m_hdCreateButton->setEnabled(frameIndex >= 0 && !hasHd);
        m_hdDeleteButton->setEnabled(hasHd);
    }
    if (m_backgroundAssignLabel) {
        m_backgroundAssignLabel->setText(m_useHdFrame ? "Background (HD)" : "Background");
    }
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
    m_shapeCompToggle->setEnabled(hasFrames);
    if (m_framesCanvas) {
        m_framesCanvas->setMaskButtonsEnabled(hasFrames);
        m_framesCanvas->setBackgroundMaskEnabled(hasFrames);
    }
    m_maskList->setEnabled(hasFrames);
    m_maskMoveUp->setEnabled(hasFrames);
    m_maskMoveDown->setEnabled(hasFrames);
    m_maskClearButton->setEnabled(hasFrames);
    m_dynamicMaskList->setEnabled(hasFrames);
    m_dynamicMaskMoveUp->setEnabled(hasFrames);
    m_dynamicMaskMoveDown->setEnabled(hasFrames);
    m_dynamicMaskClearButton->setEnabled(hasFrames);
    if (m_frameBackgroundAssign) {
        m_frameBackgroundAssign->setEnabled(hasFrames);
    }

    refreshMaskCombos();
    refreshDynamicMaskCombos();
    ensureBackgroundDataSize();
}

void MainWindow::ensureBackgroundDataSize()
{
    const int frameCount = m_frameStore ? m_frameStore->count() : 0;
    if (frameCount < 0) {
        return;
    }
    const bool hasFrames = frameCount > 0 && m_framesList && !(frameCount == 1 && m_framesList->item(0)->text().startsWith("No frames"));
    if (m_frameBackgroundIds.size() != static_cast<std::size_t>(frameCount)) {
        m_frameBackgroundIds.resize(static_cast<std::size_t>(frameCount), 0xffff);
    }
    if (m_frameBackgroundMasks.size() != static_cast<std::size_t>(frameCount)) {
        m_frameBackgroundMasks.resize(static_cast<std::size_t>(frameCount));
    }
    if (m_frameBackgroundMasksX.size() != static_cast<std::size_t>(frameCount)) {
        m_frameBackgroundMasksX.resize(static_cast<std::size_t>(frameCount));
    }

    const cv::Mat* firstFrame = (frameCount > 0) ? m_frameStore->at(0) : nullptr;
    const int width = firstFrame ? firstFrame->cols : 0;
    const int height = firstFrame ? firstFrame->rows : 0;

    cv::Size hdSize;
    for (const auto& hd : m_frameExtraFrames) {
        if (!hd.empty()) {
            hdSize = hd.size();
            break;
        }
    }

    for (int i = 0; i < frameCount; ++i) {
        cv::Mat& mask = m_frameBackgroundMasks[static_cast<std::size_t>(i)];
        if (width > 0 && height > 0 && (mask.empty() || mask.cols != width || mask.rows != height)) {
            mask = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
        }
        cv::Mat& maskX = m_frameBackgroundMasksX[static_cast<std::size_t>(i)];
        if (hdSize.width > 0 && hdSize.height > 0 &&
            (maskX.empty() || maskX.cols != hdSize.width || maskX.rows != hdSize.height)) {
            if (!mask.empty()) {
                cv::resize(mask, maskX, hdSize, 0.0, 0.0, cv::INTER_NEAREST);
            } else {
                maskX = cv::Mat(hdSize.height, hdSize.width, CV_8UC1, cv::Scalar(0));
            }
        }
    }

    refreshBackgroundList();
    const bool hasBackgrounds = m_backgroundStore && m_backgroundStore->count() > 0;
    if (m_framesCanvas) {
        m_framesCanvas->setBackgroundMaskEnabled(hasFrames && hasBackgrounds);
        m_framesCanvas->setBackgroundEnabled(hasBackgrounds);
    }
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
    cv::Mat scaledMask = mask;
    if (mask.size() != preview.size()) {
        cv::resize(mask, scaledMask, preview.size(), 0.0, 0.0, cv::INTER_NEAREST);
    }
    const double alpha = 0.5;
    for (int y = 0; y < preview.rows; ++y) {
        cv::Vec3b* row = preview.ptr<cv::Vec3b>(y);
        const uint8_t* mrow = scaledMask.ptr<uint8_t>(y);
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
        m_framesCanvas->canvas()->clearMaskOutline();
        return;
    }
    const cv::Mat* frame = activeFrameImage(index, false);
    if (!frame || frame->empty()) {
        m_framesCanvas->canvas()->clearMaskOutline();
        return;
    }
    ensureMaskDataSize();
    ensureBackgroundDataSize();
    cv::Mat reference = buildOriginalPreviewForIndex(index);
    const QColor gap = m_framesCanvas->palette().color(QPalette::Window);
    const cv::Scalar gapColor(gap.blue(), gap.green(), gap.red());
    cv::Mat base = m_showBackgroundLayer
        ? applyBackgroundComposite(index, *frame, m_useHdFrame)
        : EnsureBgr(*frame);
    cv::Mat topPreview = base;
    bool hasTopMask = false;
    bool hasBottomMask = false;
    QRect topRegion;
    QRect bottomRegion;

    if (m_backgroundMaskMode) {
        if (index >= static_cast<int>(m_frameBackgroundIds.size()) ||
            m_frameBackgroundIds[static_cast<std::size_t>(index)] == 0xffff) {
            m_backgroundMaskMode = false;
        } else if (cv::Mat* bgMask = activeBackgroundMask(index)) {
            cv::Mat fallbackMask;
            if ((!bgMask || bgMask->empty() || !MaskHasContent(*bgMask)) && m_useHdFrame &&
                index < static_cast<int>(m_frameBackgroundMasks.size())) {
                const cv::Mat& sdMask = m_frameBackgroundMasks[static_cast<std::size_t>(index)];
                if (MaskHasContent(sdMask)) {
                    fallbackMask = sdMask;
                    bgMask = &fallbackMask;
                }
            }
            if (bgMask && MaskHasContent(*bgMask)) {
                topPreview = buildMaskPreview(base, *bgMask, cv::Vec3b(60, 200, 120));
                hasTopMask = true;
                if (m_showOriginalFrame) {
                    cv::Mat original = buildOriginalFrame(reference);
                    cv::Mat displayOriginal = BuildDisplayOriginal(original, topPreview.size());
                    FrameLayout layout = BuildFrameLayout(topPreview, displayOriginal);
                    topRegion = QRect(layout.topX, 0, layout.topWidth, layout.topHeight);
                }
            }
        }
    }

    if (!m_showOriginalFrame && m_maskMode != MaskMode::None && !hasTopMask) {
        m_framesCanvas->canvas()->clearPreviewImage();
        m_framesCanvas->canvas()->clearMaskOutline();
        m_framesCanvas->canvas()->setSecondaryMaskOutline(cv::Mat(), QColor());
        return;
    }

    cv::Mat bottomPreview;
    QColor bottomOutlineColor;
    if (m_showOriginalFrame) {
        cv::Mat original = buildOriginalFrame(reference);
        if (m_maskMode == MaskMode::Dynamic) {
            const int maskId = currentFrameDynamicMaskId();
            if (maskId >= 0 && maskId < static_cast<int>(m_dynamicMasks.size())) {
                const cv::Mat& mask = m_dynamicMasks[static_cast<std::size_t>(maskId)];
                bottomPreview = buildMaskPreview(original, mask, cv::Vec3b(0, 200, 255));
                hasBottomMask = true;
                bottomOutlineColor = QColor(255, 200, 0);
            }
        } else if (m_maskMode == MaskMode::Comparison) {
            const int maskId = currentFrameMaskId();
            if (maskId >= 0 && maskId < static_cast<int>(m_compMasks.size())) {
                const cv::Mat& mask = m_compMasks[static_cast<std::size_t>(maskId)];
                bottomPreview = buildMaskPreview(original, mask, cv::Vec3b(200, 0, 200));
                hasBottomMask = true;
                bottomOutlineColor = QColor(200, 0, 200);
            }
        }
        if (bottomPreview.empty()) {
            bottomPreview = original;
        }
        cv::Mat displayOriginal = BuildDisplayOriginal(bottomPreview, topPreview.size());
        FrameLayout layout = BuildFrameLayout(topPreview, displayOriginal);
        bottomRegion = QRect(layout.bottomX,
                             layout.topHeight + kFrameGapPixels,
                             layout.bottomWidth,
                             layout.bottomHeight);
        cv::Mat combined = buildCombinedFrame(topPreview, displayOriginal, gapColor);
        m_framesCanvas->canvas()->setPreviewImage(combined);
    } else {
        if (hasTopMask) {
            m_framesCanvas->canvas()->setPreviewImage(topPreview);
        } else {
            m_framesCanvas->canvas()->clearPreviewImage();
        }
    }

    if (hasBottomMask && m_maskMode != MaskMode::None) {
        const cv::Mat* mask = (m_maskMode == MaskMode::Dynamic)
            ? (currentFrameDynamicMaskId() >= 0 ? &m_dynamicMasks[static_cast<std::size_t>(currentFrameDynamicMaskId())] : nullptr)
            : (currentFrameMaskId() >= 0 ? &m_compMasks[static_cast<std::size_t>(currentFrameMaskId())] : nullptr);
        if (mask && !mask->empty()) {
            m_framesCanvas->canvas()->setMaskOutline(*mask, bottomOutlineColor, bottomRegion);
        }
    } else {
        m_framesCanvas->canvas()->clearPrimaryOutline();
    }
    if (hasTopMask && m_backgroundMaskMode) {
        cv::Mat* bgMask = activeBackgroundMask(index);
        cv::Mat fallbackMask;
        if ((!bgMask || bgMask->empty() || !MaskHasContent(*bgMask)) && m_useHdFrame &&
            index < static_cast<int>(m_frameBackgroundMasks.size())) {
            const cv::Mat& sdMask = m_frameBackgroundMasks[static_cast<std::size_t>(index)];
            if (MaskHasContent(sdMask)) {
                fallbackMask = sdMask;
                bgMask = &fallbackMask;
            }
        }
        if (bgMask && MaskHasContent(*bgMask)) {
            m_framesCanvas->canvas()->setSecondaryMaskOutline(*bgMask, QColor(120, 200, 60), topRegion);
        }
    }
    if (!(hasTopMask && m_backgroundMaskMode)) {
        m_framesCanvas->canvas()->setSecondaryMaskOutline(cv::Mat(), QColor());
    }
    if (!hasBottomMask && !(hasTopMask && m_backgroundMaskMode)) {
        m_framesCanvas->canvas()->clearMaskOutline();
    }
}

void MainWindow::setMaskMode(MaskMode mode)
{
    if (m_maskMode == mode) {
        return;
    }
    m_maskMode = mode;
    cancelCurrentDraw();
    if (m_framesCanvas) {
        const bool comp = (m_maskMode == MaskMode::Comparison);
        const bool dyn = (m_maskMode == MaskMode::Dynamic);
        m_framesCanvas->setMaskButtonsChecked(comp, dyn);
    }
    if (m_framesList && m_framesList->currentRow() >= 0) {
        if (m_maskMode == MaskMode::Comparison && m_maskList) {
            const int assigned = currentFrameMaskId();
            if (assigned >= 0 && assigned < m_maskList->count()) {
                m_maskList->setCurrentRow(assigned);
            }
        } else if (m_maskMode == MaskMode::Dynamic && m_dynamicMaskList) {
            const int assigned = currentFrameDynamicMaskId();
            if (assigned >= 0 && assigned < m_dynamicMaskList->count()) {
                m_dynamicMaskList->setCurrentRow(assigned);
            }
        }
    }
    updateMaskPreviewForFrame(m_framesList ? m_framesList->currentRow() : -1);
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
    syncDynamicSetSelection();
    refreshDynamicPaletteButtons();
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
        pixmap = pixmap.scaled(kPreviewIconWidthHd, kPreviewIconHeightHd, Qt::KeepAspectRatio, Qt::FastTransformation);
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

cv::Mat* MainWindow::activeBackgroundMask(int index)
{
    if (index < 0) {
        return nullptr;
    }
    if (m_useHdFrame && index < static_cast<int>(m_frameBackgroundMasksX.size())) {
        return &m_frameBackgroundMasksX[static_cast<std::size_t>(index)];
    }
    if (index < static_cast<int>(m_frameBackgroundMasks.size())) {
        return &m_frameBackgroundMasks[static_cast<std::size_t>(index)];
    }
    return nullptr;
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
    m_frameHdUndoStacks.clear();
    m_spriteUndoStacks.clear();
    m_backgroundUndoStacks.clear();
    m_backgroundHdUndoStacks.clear();
    m_compMaskUndoStacks.clear();
    m_dynMaskUndoStacks.clear();
    m_backgroundMaskUndoStacks.clear();
    m_frameUndoActive = false;
    m_spriteUndoActive = false;
    m_backgroundUndoActive = false;
    updateUndoActions();
}

void MainWindow::ensureUndoStacksSize()
{
    const int frameCount = m_frameStore ? m_frameStore->count() : 0;
    const int spriteCount = m_spriteStore ? m_spriteStore->count() : 0;
    const int backgroundCount = m_backgroundStore ? m_backgroundStore->count() : 0;
    if (frameCount >= 0) {
        m_frameUndoStacks.resize(static_cast<std::size_t>(frameCount));
        m_frameHdUndoStacks.resize(static_cast<std::size_t>(frameCount));
        m_compMaskUndoStacks.resize(static_cast<std::size_t>(frameCount));
        m_dynMaskUndoStacks.resize(static_cast<std::size_t>(frameCount));
        m_backgroundMaskUndoStacks.resize(static_cast<std::size_t>(frameCount));
    }
    if (spriteCount >= 0) {
        m_spriteUndoStacks.resize(static_cast<std::size_t>(spriteCount));
    }
    if (backgroundCount >= 0) {
        m_backgroundUndoStacks.resize(static_cast<std::size_t>(backgroundCount));
        m_backgroundHdUndoStacks.resize(static_cast<std::size_t>(backgroundCount));
    }
    updateUndoActions();
}

void MainWindow::pushUndoSnapshot(bool isFrame, int index)
{
    if (index < 0) {
        return;
    }
    std::vector<UndoStack>* stacks = nullptr;
    if (isFrame) {
        if (m_useHdFrame && hasHdFrame(index)) {
            stacks = &m_frameHdUndoStacks;
        } else {
            stacks = &m_frameUndoStacks;
        }
    } else {
        stacks = &m_spriteUndoStacks;
    }
    if (!stacks) {
        return;
    }
    if (index >= static_cast<int>(stacks->size())) {
        return;
    }
    cv::Mat* image = isFrame ? activeFrameImage(index, true) : m_spriteStore->atMutable(index);
    if (!image || image->empty()) {
        return;
    }
    UndoStack& stack = (*stacks)[static_cast<std::size_t>(index)];
    UndoState state;
    state.image = image->clone();
    stack.undo.push_back(std::move(state));
    if (stack.undo.size() > kMaxUndoDepth) {
        stack.undo.erase(stack.undo.begin());
    }
    stack.redo.clear();
    updateUndoActions();
}

void MainWindow::pushMaskUndoSnapshot(MaskMode mode, int index)
{
    if (index < 0) {
        return;
    }
    std::vector<UndoStack>* stacks = nullptr;
    if (mode == MaskMode::Comparison) {
        stacks = &m_compMaskUndoStacks;
    } else if (mode == MaskMode::Dynamic) {
        stacks = &m_dynMaskUndoStacks;
    } else {
        return;
    }
    if (index >= static_cast<int>(stacks->size())) {
        return;
    }
    UndoStack& stack = (*stacks)[static_cast<std::size_t>(index)];
    UndoState state;
    if (mode == MaskMode::Comparison) {
        const int maskId = m_maskList ? m_maskList->currentRow() : -1;
        if (maskId >= 0 && maskId < static_cast<int>(m_compMasks.size())) {
            state.mask = m_compMasks[static_cast<std::size_t>(maskId)].clone();
            state.mask_kind = MaskKind::Comparison;
            state.mask_index = maskId;
        }
    } else if (mode == MaskMode::Dynamic) {
        const int maskId = m_dynamicMaskList ? m_dynamicMaskList->currentRow() : -1;
        if (maskId >= 0 && maskId < static_cast<int>(m_dynamicMasks.size())) {
            state.mask = m_dynamicMasks[static_cast<std::size_t>(maskId)].clone();
            state.mask_kind = MaskKind::Dynamic;
            state.mask_index = maskId;
        }
    }
    if (state.mask.empty()) {
        return;
    }
    stack.undo.push_back(std::move(state));
    if (stack.undo.size() > kMaxUndoDepth) {
        stack.undo.erase(stack.undo.begin());
    }
    stack.redo.clear();
    updateUndoActions();
}

void MainWindow::pushBackgroundUndoSnapshot(int index)
{
    if (index < 0 || !m_backgroundStore) {
        return;
    }
    std::vector<UndoStack>* stacks = nullptr;
    if (m_useHdBackground && hasHdBackground(index)) {
        stacks = &m_backgroundHdUndoStacks;
    } else {
        stacks = &m_backgroundUndoStacks;
    }
    if (!stacks || index >= static_cast<int>(stacks->size())) {
        return;
    }
    cv::Mat* image = activeBackgroundImage(index, true);
    if (!image || image->empty()) {
        return;
    }
    UndoStack& stack = (*stacks)[static_cast<std::size_t>(index)];
    UndoState state;
    state.image = image->clone();
    stack.undo.push_back(std::move(state));
    if (stack.undo.size() > kMaxUndoDepth) {
        stack.undo.erase(stack.undo.begin());
    }
    stack.redo.clear();
    updateUndoActions();
}

void MainWindow::pushBackgroundMaskUndoSnapshot(int index)
{
    if (index < 0 || index >= static_cast<int>(m_backgroundMaskUndoStacks.size())) {
        return;
    }
    cv::Mat* mask = activeBackgroundMask(index);
    if (!mask || mask->empty()) {
        return;
    }
    UndoStack& stack = m_backgroundMaskUndoStacks[static_cast<std::size_t>(index)];
    UndoState state;
    state.mask = mask->clone();
    state.mask_kind = MaskKind::None;
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
    std::vector<UndoStack>* stacks = nullptr;
    if (isFrame) {
        if (m_useHdFrame && hasHdFrame(index)) {
            stacks = &m_frameHdUndoStacks;
        } else {
            stacks = &m_frameUndoStacks;
        }
    } else {
        stacks = &m_spriteUndoStacks;
    }
    if (!stacks || index < 0 || index >= static_cast<int>(stacks->size())) {
        return false;
    }
    UndoStack& stack = (*stacks)[static_cast<std::size_t>(index)];
    if (stack.undo.empty()) {
        return false;
    }
    cv::Mat* image = isFrame ? activeFrameImage(index, true) : m_spriteStore->atMutable(index);
    if (!image || image->empty()) {
        return false;
    }
    UndoState current;
    current.image = image->clone();
    stack.redo.push_back(std::move(current));
    UndoState previous = stack.undo.back();
    stack.undo.pop_back();
    *image = previous.image.clone();
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
    std::vector<UndoStack>* stacks = nullptr;
    if (isFrame) {
        if (m_useHdFrame && hasHdFrame(index)) {
            stacks = &m_frameHdUndoStacks;
        } else {
            stacks = &m_frameUndoStacks;
        }
    } else {
        stacks = &m_spriteUndoStacks;
    }
    if (!stacks || index < 0 || index >= static_cast<int>(stacks->size())) {
        return false;
    }
    UndoStack& stack = (*stacks)[static_cast<std::size_t>(index)];
    if (stack.redo.empty()) {
        return false;
    }
    cv::Mat* image = isFrame ? activeFrameImage(index, true) : m_spriteStore->atMutable(index);
    if (!image || image->empty()) {
        return false;
    }
    UndoState current;
    current.image = image->clone();
    stack.undo.push_back(std::move(current));
    if (stack.undo.size() > kMaxUndoDepth) {
        stack.undo.erase(stack.undo.begin());
    }
    UndoState next = stack.redo.back();
    stack.redo.pop_back();
    *image = next.image.clone();
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

bool MainWindow::undoBackgroundEdit()
{
    const int index = m_backgroundList ? m_backgroundList->currentRow() : -1;
    std::vector<UndoStack>* stacks = nullptr;
    if (m_useHdBackground && hasHdBackground(index)) {
        stacks = &m_backgroundHdUndoStacks;
    } else {
        stacks = &m_backgroundUndoStacks;
    }
    if (!stacks || index < 0 || index >= static_cast<int>(stacks->size())) {
        return false;
    }
    UndoStack& stack = (*stacks)[static_cast<std::size_t>(index)];
    if (stack.undo.empty()) {
        return false;
    }
    cv::Mat* image = activeBackgroundImage(index, true);
    if (!image || image->empty()) {
        return false;
    }
    UndoState current;
    current.image = image->clone();
    stack.redo.push_back(std::move(current));
    UndoState previous = stack.undo.back();
    stack.undo.pop_back();
    *image = previous.image.clone();
    updateBackgroundCanvasImage(index);
    updateUndoActions();
    return true;
}

bool MainWindow::redoBackgroundEdit()
{
    const int index = m_backgroundList ? m_backgroundList->currentRow() : -1;
    std::vector<UndoStack>* stacks = nullptr;
    if (m_useHdBackground && hasHdBackground(index)) {
        stacks = &m_backgroundHdUndoStacks;
    } else {
        stacks = &m_backgroundUndoStacks;
    }
    if (!stacks || index < 0 || index >= static_cast<int>(stacks->size())) {
        return false;
    }
    UndoStack& stack = (*stacks)[static_cast<std::size_t>(index)];
    if (stack.redo.empty()) {
        return false;
    }
    cv::Mat* image = activeBackgroundImage(index, true);
    if (!image || image->empty()) {
        return false;
    }
    UndoState current;
    current.image = image->clone();
    stack.undo.push_back(std::move(current));
    if (stack.undo.size() > kMaxUndoDepth) {
        stack.undo.erase(stack.undo.begin());
    }
    UndoState next = stack.redo.back();
    stack.redo.pop_back();
    *image = next.image.clone();
    updateBackgroundCanvasImage(index);
    updateUndoActions();
    return true;
}

bool MainWindow::undoBackgroundMaskEdit()
{
    const int index = m_framesList ? m_framesList->currentRow() : -1;
    if (index < 0 || index >= static_cast<int>(m_backgroundMaskUndoStacks.size())) {
        return false;
    }
    UndoStack& stack = m_backgroundMaskUndoStacks[static_cast<std::size_t>(index)];
    if (stack.undo.empty()) {
        return false;
    }
    cv::Mat* mask = activeBackgroundMask(index);
    if (!mask || mask->empty()) {
        return false;
    }
    UndoState current;
    current.mask = mask->clone();
    stack.redo.push_back(std::move(current));
    UndoState previous = stack.undo.back();
    stack.undo.pop_back();
    *mask = previous.mask.clone();
    updateFrameCanvasImage(index);
    updateFramePreviewAt(index);
    updateUndoActions();
    return true;
}

bool MainWindow::redoBackgroundMaskEdit()
{
    const int index = m_framesList ? m_framesList->currentRow() : -1;
    if (index < 0 || index >= static_cast<int>(m_backgroundMaskUndoStacks.size())) {
        return false;
    }
    UndoStack& stack = m_backgroundMaskUndoStacks[static_cast<std::size_t>(index)];
    if (stack.redo.empty()) {
        return false;
    }
    cv::Mat* mask = activeBackgroundMask(index);
    if (!mask || mask->empty()) {
        return false;
    }
    UndoState current;
    current.mask = mask->clone();
    stack.undo.push_back(std::move(current));
    if (stack.undo.size() > kMaxUndoDepth) {
        stack.undo.erase(stack.undo.begin());
    }
    UndoState next = stack.redo.back();
    stack.redo.pop_back();
    *mask = next.mask.clone();
    updateFrameCanvasImage(index);
    updateFramePreviewAt(index);
    updateUndoActions();
    return true;
}

bool MainWindow::undoMaskEdit(MaskMode mode)
{
    const int index = m_framesList ? m_framesList->currentRow() : -1;
    std::vector<UndoStack>* stacks = nullptr;
    if (mode == MaskMode::Comparison) {
        stacks = &m_compMaskUndoStacks;
    } else if (mode == MaskMode::Dynamic) {
        stacks = &m_dynMaskUndoStacks;
    } else {
        return false;
    }
    if (index < 0 || index >= static_cast<int>(stacks->size())) {
        return false;
    }
    UndoStack& stack = (*stacks)[static_cast<std::size_t>(index)];
    if (stack.undo.empty()) {
        return false;
    }
    UndoState current;
    if (mode == MaskMode::Comparison) {
        const int maskId = m_maskList ? m_maskList->currentRow() : -1;
        if (maskId >= 0 && maskId < static_cast<int>(m_compMasks.size())) {
            current.mask = m_compMasks[static_cast<std::size_t>(maskId)].clone();
            current.mask_kind = MaskKind::Comparison;
            current.mask_index = maskId;
        }
    } else if (mode == MaskMode::Dynamic) {
        const int maskId = m_dynamicMaskList ? m_dynamicMaskList->currentRow() : -1;
        if (maskId >= 0 && maskId < static_cast<int>(m_dynamicMasks.size())) {
            current.mask = m_dynamicMasks[static_cast<std::size_t>(maskId)].clone();
            current.mask_kind = MaskKind::Dynamic;
            current.mask_index = maskId;
        }
    }
    if (!current.mask.empty()) {
        stack.redo.push_back(std::move(current));
    }
    UndoState previous = stack.undo.back();
    stack.undo.pop_back();
    if (previous.mask_kind == MaskKind::Comparison &&
        previous.mask_index >= 0 && previous.mask_index < static_cast<int>(m_compMasks.size())) {
        m_compMasks[static_cast<std::size_t>(previous.mask_index)] = previous.mask.clone();
        updateMaskPreviewIcons();
    } else if (previous.mask_kind == MaskKind::Dynamic &&
               previous.mask_index >= 0 && previous.mask_index < static_cast<int>(m_dynamicMasks.size())) {
        m_dynamicMasks[static_cast<std::size_t>(previous.mask_index)] = previous.mask.clone();
        updateDynamicMaskPreviewIcons();
    }
    updateMaskPreviewForFrame(index);
    updateUndoActions();
    return true;
}

bool MainWindow::redoMaskEdit(MaskMode mode)
{
    const int index = m_framesList ? m_framesList->currentRow() : -1;
    std::vector<UndoStack>* stacks = nullptr;
    if (mode == MaskMode::Comparison) {
        stacks = &m_compMaskUndoStacks;
    } else if (mode == MaskMode::Dynamic) {
        stacks = &m_dynMaskUndoStacks;
    } else {
        return false;
    }
    if (index < 0 || index >= static_cast<int>(stacks->size())) {
        return false;
    }
    UndoStack& stack = (*stacks)[static_cast<std::size_t>(index)];
    if (stack.redo.empty()) {
        return false;
    }
    UndoState current;
    if (mode == MaskMode::Comparison) {
        const int maskId = m_maskList ? m_maskList->currentRow() : -1;
        if (maskId >= 0 && maskId < static_cast<int>(m_compMasks.size())) {
            current.mask = m_compMasks[static_cast<std::size_t>(maskId)].clone();
            current.mask_kind = MaskKind::Comparison;
            current.mask_index = maskId;
        }
    } else if (mode == MaskMode::Dynamic) {
        const int maskId = m_dynamicMaskList ? m_dynamicMaskList->currentRow() : -1;
        if (maskId >= 0 && maskId < static_cast<int>(m_dynamicMasks.size())) {
            current.mask = m_dynamicMasks[static_cast<std::size_t>(maskId)].clone();
            current.mask_kind = MaskKind::Dynamic;
            current.mask_index = maskId;
        }
    }
    if (!current.mask.empty()) {
        stack.undo.push_back(std::move(current));
    }
    if (stack.undo.size() > kMaxUndoDepth) {
        stack.undo.erase(stack.undo.begin());
    }
    UndoState next = stack.redo.back();
    stack.redo.pop_back();
    if (next.mask_kind == MaskKind::Comparison &&
        next.mask_index >= 0 && next.mask_index < static_cast<int>(m_compMasks.size())) {
        m_compMasks[static_cast<std::size_t>(next.mask_index)] = next.mask.clone();
        updateMaskPreviewIcons();
    } else if (next.mask_kind == MaskKind::Dynamic &&
               next.mask_index >= 0 && next.mask_index < static_cast<int>(m_dynamicMasks.size())) {
        m_dynamicMasks[static_cast<std::size_t>(next.mask_index)] = next.mask.clone();
        updateDynamicMaskPreviewIcons();
    }
    updateMaskPreviewForFrame(index);
    updateUndoActions();
    return true;
}

void MainWindow::updateUndoActions()
{
    const UndoTarget target = currentUndoTarget();
    int index = -1;
    const std::vector<UndoStack>* stacks = nullptr;
    switch (target) {
        case UndoTarget::Frame:
            index = m_framesList ? m_framesList->currentRow() : -1;
            if (m_useHdFrame && hasHdFrame(index)) {
                stacks = &m_frameHdUndoStacks;
            } else {
                stacks = &m_frameUndoStacks;
            }
            break;
        case UndoTarget::Sprite:
            index = m_spritesList ? m_spritesList->currentRow() : -1;
            stacks = &m_spriteUndoStacks;
            break;
        case UndoTarget::Background:
            index = m_backgroundList ? m_backgroundList->currentRow() : -1;
            if (m_useHdBackground && hasHdBackground(index)) {
                stacks = &m_backgroundHdUndoStacks;
            } else {
                stacks = &m_backgroundUndoStacks;
            }
            break;
        case UndoTarget::CompMask:
            index = m_framesList ? m_framesList->currentRow() : -1;
            stacks = &m_compMaskUndoStacks;
            break;
        case UndoTarget::DynMask:
            index = m_framesList ? m_framesList->currentRow() : -1;
            stacks = &m_dynMaskUndoStacks;
            break;
        case UndoTarget::BackgroundMask:
            index = m_framesList ? m_framesList->currentRow() : -1;
            stacks = &m_backgroundMaskUndoStacks;
            break;
    }
    bool canUndo = false;
    bool canRedo = false;
    if (stacks && index >= 0 && index < static_cast<int>(stacks->size())) {
        const UndoStack& stack = (*stacks)[static_cast<std::size_t>(index)];
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
    if (m_canvasTabs &&
        (m_canvasTabs->currentWidget() == m_spritesCanvas ||
         m_canvasTabs->currentWidget() == m_backgroundsCanvas)) {
        return false;
    }
    return m_framesList && m_framesList->currentRow() >= 0;
}

MainWindow::UndoTarget MainWindow::currentUndoTarget() const
{
    if (m_canvasTabs && m_canvasTabs->currentWidget() == m_backgroundsCanvas) {
        return UndoTarget::Background;
    }
    if (m_canvasTabs && m_canvasTabs->currentWidget() == m_spritesCanvas) {
        return UndoTarget::Sprite;
    }
    if (!isFrameContext()) {
        return UndoTarget::Sprite;
    }
    if (!m_showOriginalFrame) {
        return UndoTarget::Frame;
    }
    if (m_backgroundMaskMode) {
        return UndoTarget::BackgroundMask;
    }
    if (m_maskMode != MaskMode::None && m_frameHoverArea == FrameHoverArea::Bottom) {
        return (m_maskMode == MaskMode::Dynamic) ? UndoTarget::DynMask : UndoTarget::CompMask;
    }
    return UndoTarget::Frame;
}

void MainWindow::setHdMode(bool enabled)
{
    const int index = m_framesList ? m_framesList->currentRow() : -1;
    const cv::Mat* beforeImage = activeFrameImage(index, false);
    const cv::Size beforeSize = beforeImage ? beforeImage->size() : cv::Size();
    const bool canEnable = enabled && hasHdFrame(index);
    m_useHdFrame = canEnable;
    if (m_framesCanvas) {
        m_framesCanvas->setHdButtonChecked(m_useHdFrame);
        m_framesCanvas->setHdButtonEnabled(hasHdFrame(index));
    }
    updateHdControlsForContext();
    ensureBackgroundDataSize();
    if (m_useHdFrame && index >= 0 &&
        index < static_cast<int>(m_frameBackgroundMasks.size()) &&
        index < static_cast<int>(m_frameBackgroundMasksX.size())) {
        const cv::Mat& sdMask = m_frameBackgroundMasks[static_cast<std::size_t>(index)];
        cv::Mat& hdMask = m_frameBackgroundMasksX[static_cast<std::size_t>(index)];
        if (MaskHasContent(sdMask) && (hdMask.empty() || !MaskHasContent(hdMask))) {
            const cv::Mat* hdFrame = activeFrameImage(index, false);
            if (hdFrame && !hdFrame->empty()) {
                cv::resize(sdMask, hdMask, hdFrame->size(), 0.0, 0.0, cv::INTER_NEAREST);
            }
        }
    }
    const cv::Mat* afterImage = activeFrameImage(index, false);
    const cv::Size afterSize = afterImage ? afterImage->size() : cv::Size();
    if (m_framesCanvas && beforeSize.width > 0 && afterSize.width > 0 &&
        (beforeSize.width != afterSize.width || beforeSize.height != afterSize.height)) {
        const double factor = static_cast<double>(beforeSize.width) /
            static_cast<double>(afterSize.width);
        m_framesCanvas->canvas()->scaleZoom(factor);
    }
    updateFrameCanvasImage(index);
    updateMaskPreviewForFrame(index);
    updateUndoActions();
}

bool MainWindow::hasHdFrame(int index) const
{
    if (index < 0 || index >= static_cast<int>(m_frameExtraFrames.size())) {
        return false;
    }
    return !m_frameExtraFrames[static_cast<std::size_t>(index)].empty();
}

cv::Mat* MainWindow::activeFrameImage(int index, bool forEdit)
{
    if (index < 0) {
        return nullptr;
    }
    if (m_useHdFrame && hasHdFrame(index)) {
        return &m_frameExtraFrames[static_cast<std::size_t>(index)];
    }
    if (forEdit) {
        return m_frameStore->atMutable(index);
    }
    return m_frameStore->atMutable(index);
}

cv::Mat* MainWindow::activeBackgroundImage(int index, bool forEdit)
{
    if (index < 0) {
        return nullptr;
    }
    if (m_useHdBackground && hasHdBackground(index)) {
        return &m_backgroundFramesX[static_cast<std::size_t>(index)];
    }
    if (!m_backgroundStore) {
        return nullptr;
    }
    if (forEdit) {
        return m_backgroundStore->atMutable(index);
    }
    return m_backgroundStore->atMutable(index);
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

void MainWindow::initPalette()
{
    if (!m_fullPalettes.isEmpty()) {
        return;
    }
    m_fullPalettes.resize(N_PALETTES);
    m_paletteNames.resize(N_PALETTES);
    for (int p = 0; p < N_PALETTES; ++p) {
        m_paletteNames[p] = QString("Palette %1").arg(p);
        m_fullPalettes[p].reserve(64);
        for (int i = 0; i < 64; ++i) {
            const int value = static_cast<int>(std::round(255.0 * i / 63.0));
            QColor color(value, value, value);
            const cv::Vec3b quant = Rgb565ToBgr(BgrToRgb565(cv::Vec3b(color.blue(), color.green(), color.red())));
            m_fullPalettes[p].push_back(QColor(quant[2], quant[1], quant[0]));
        }
    }
    m_paletteSetIndex = 0;
    m_paletteColors = m_fullPalettes[m_paletteSetIndex];
}

void MainWindow::refreshPaletteList()
{
    if (!m_paletteList) {
        return;
    }
    QSignalBlocker blocker(m_paletteList);
    m_paletteList->clear();
    for (int i = 0; i < m_paletteColors.size(); ++i) {
        QPixmap pixmap(kPaletteSwatchSize, kPaletteSwatchSize);
        pixmap.fill(m_paletteColors[i]);
        auto* item = new QListWidgetItem(QIcon(pixmap), QString());
        item->setSizeHint(m_paletteList->gridSize());
        item->setData(Qt::UserRole, i);
        m_paletteList->addItem(item);
    }
    if (m_currentPaletteIndex >= 0 && m_currentPaletteIndex < m_paletteList->count()) {
        m_paletteList->setCurrentRow(m_currentPaletteIndex);
    }
    if (m_paletteSetCombo) {
        QSignalBlocker blockCombo(m_paletteSetCombo);
        m_paletteSetCombo->clear();
        for (int i = 0; i < m_paletteNames.size(); ++i) {
            const QString name = m_paletteNames[i].trimmed();
            const QString label = name.isEmpty() ? QString::number(i + 1) : name;
            m_paletteSetCombo->addItem(label, i);
        }
        if (m_paletteSetIndex >= 0 && m_paletteSetIndex < m_paletteSetCombo->count()) {
            m_paletteSetCombo->setCurrentIndex(m_paletteSetIndex);
        }
    }
}

int MainWindow::reducedSlotCount() const
{
    const int count = static_cast<int>(m_noColors);
    if (count <= 0) {
        return 16;
    }
    return std::min(16, count);
}

int MainWindow::dynamicSlotCount() const
{
    const int count = static_cast<int>(m_noColors);
    if (count <= 0) {
        return 16;
    }
    return std::min(16, count);
}

int MainWindow::dynamicColorsPerSet(const std::vector<uint16_t>& colors) const
{
    if (colors.empty()) {
        return 0;
    }
    if (colors.size() % MAX_DYNA_SETS_PER_FRAMEN == 0) {
        return static_cast<int>(colors.size() / MAX_DYNA_SETS_PER_FRAMEN);
    }
    return 16;
}

QColor MainWindow::reducedSlotColor(int setIndex, int slot) const
{
    if (setIndex < 0 || setIndex >= kReducedPaletteCount || slot < 0 || slot >= 16) {
        return QColor(0, 0, 0);
    }
    const std::size_t offset = static_cast<std::size_t>(setIndex) * 16 + static_cast<std::size_t>(slot);
    if (offset >= m_reducedPaletteIndices.size()) {
        return QColor(0, 0, 0);
    }
    const uint16_t value = m_reducedPaletteIndices[offset];
    const cv::Vec3b bgr = Rgb565ToBgr(value);
    return QColor(bgr[2], bgr[1], bgr[0]);
}

QColor MainWindow::dynamicSlotColor(int slot) const
{
    if (slot < 0 || slot >= 16) {
        return QColor(0, 0, 0);
    }
    const int frameIndex = m_framesList ? m_framesList->currentRow() : -1;
    if (frameIndex < 0 || frameIndex >= static_cast<int>(m_frameDynamicColors.size())) {
        return QColor(0, 0, 0);
    }
    const std::vector<uint16_t>& colors = m_frameDynamicColors[static_cast<std::size_t>(frameIndex)];
    const int stride = dynamicColorsPerSet(colors);
    if (stride <= 0 || slot >= stride) {
        return QColor(0, 0, 0);
    }
    const std::size_t offset = static_cast<std::size_t>(m_dynamicSetIndex) * stride + slot;
    if (offset >= colors.size()) {
        return QColor(0, 0, 0);
    }
    const uint16_t value = colors[offset];
    const cv::Vec3b bgr = Rgb565ToBgr(value);
    return QColor(bgr[2], bgr[1], bgr[0]);
}

void MainWindow::setReducedSlotColor(int setIndex, int slot, const QColor& color)
{
    if (setIndex < 0 || setIndex >= kReducedPaletteCount || slot < 0 || slot >= 16) {
        return;
    }
    if (m_reducedPaletteIndices.size() < static_cast<std::size_t>(kReducedPaletteCount * 16)) {
        m_reducedPaletteIndices.resize(static_cast<std::size_t>(kReducedPaletteCount * 16), 0);
    }
    const cv::Vec3b bgr(color.blue(), color.green(), color.red());
    const uint16_t value = BgrToRgb565(bgr);
    const std::size_t offset = static_cast<std::size_t>(setIndex) * 16 + static_cast<std::size_t>(slot);
    if (offset < m_reducedPaletteIndices.size()) {
        m_reducedPaletteIndices[offset] = value;
    }
}

void MainWindow::setDynamicSlotColor(int frameIndex, int setIndex, int slot, const QColor& color)
{
    if (frameIndex < 0 || frameIndex >= static_cast<int>(m_frameDynamicColors.size())) {
        return;
    }
    if (setIndex < 0 || setIndex >= MAX_DYNA_SETS_PER_FRAMEN || slot < 0 || slot >= 16) {
        return;
    }
    std::vector<uint16_t>& colors = m_frameDynamicColors[static_cast<std::size_t>(frameIndex)];
    const int stride = dynamicColorsPerSet(colors);
    if (stride <= 0 || slot >= stride) {
        return;
    }
    const std::size_t offset = static_cast<std::size_t>(setIndex) * stride + slot;
    if (offset >= colors.size()) {
        return;
    }
    const cv::Vec3b bgr(color.blue(), color.green(), color.red());
    colors[offset] = BgrToRgb565(bgr);
}

void MainWindow::refreshReducedPaletteUI()
{
    if (!m_reducedSetCombo) {
        return;
    }
    if (m_reducedPaletteNames.size() < static_cast<std::size_t>(kReducedPaletteCount)) {
        m_reducedPaletteNames.resize(kReducedPaletteCount);
    }
    if (m_reducedPaletteIndices.size() < static_cast<std::size_t>(kReducedPaletteCount * 16)) {
        m_reducedPaletteIndices.resize(static_cast<std::size_t>(kReducedPaletteCount * 16), 0);
    }
    QSignalBlocker blocker(m_reducedSetCombo);
    m_reducedSetCombo->clear();
    for (int i = 0; i < kReducedPaletteCount; ++i) {
        QString name;
        if (static_cast<std::size_t>(i) < m_reducedPaletteNames.size()) {
            name = QString::fromStdString(m_reducedPaletteNames[static_cast<std::size_t>(i)]).trimmed();
        }
        const QString label = name.isEmpty() ? QString::number(i + 1) : name;
        m_reducedSetCombo->addItem(label, i);
    }
    if (m_reducedPaletteIndex < 0 || m_reducedPaletteIndex >= kReducedPaletteCount) {
        m_reducedPaletteIndex = 0;
    }
    if (m_reducedPaletteIndex >= 0 && m_reducedPaletteIndex < m_reducedSetCombo->count()) {
        m_reducedSetCombo->setCurrentIndex(m_reducedPaletteIndex);
    }
    refreshReducedPaletteButtons();
}

void MainWindow::refreshReducedPaletteButtons()
{
    if (!m_reducedPaletteList) {
        return;
    }
    const int slotCount = reducedSlotCount();
    if (m_reducedSlotIndex >= slotCount) {
        m_reducedSlotIndex = -1;
    }
    QSignalBlocker blocker(m_reducedPaletteList);
    if (m_reducedPaletteList->count() != 16) {
        m_reducedPaletteList->clear();
        for (int i = 0; i < 16; ++i) {
            auto* item = new QListWidgetItem();
            item->setData(Qt::UserRole, i);
            item->setSizeHint(m_reducedPaletteList->gridSize());
            m_reducedPaletteList->addItem(item);
        }
    }
    for (int i = 0; i < m_reducedPaletteList->count(); ++i) {
        QListWidgetItem* item = m_reducedPaletteList->item(i);
        if (!item) {
            continue;
        }
        const bool enabled = i < slotCount;
        Qt::ItemFlags flags = item->flags();
        if (enabled) {
            flags |= Qt::ItemIsEnabled | Qt::ItemIsSelectable;
        } else {
            flags &= ~Qt::ItemIsEnabled;
            flags &= ~Qt::ItemIsSelectable;
        }
        item->setFlags(flags);
        item->setData(kPaletteDisabledRole, !enabled);
        QPixmap pixmap(kPaletteSwatchSize, kPaletteSwatchSize);
        const QColor color = enabled ? reducedSlotColor(m_reducedPaletteIndex, i) : QColor(32, 32, 32);
        pixmap.fill(color);
        item->setIcon(QIcon(pixmap));
    }
    if (m_reducedSlotIndex >= 0 && m_reducedSlotIndex < m_reducedPaletteList->count()) {
        m_reducedPaletteList->setCurrentRow(m_reducedSlotIndex);
    } else {
        m_reducedPaletteList->setCurrentRow(-1);
    }
    if (m_reducedAssignButton) {
        m_reducedAssignButton->setEnabled(m_reducedSlotIndex >= 0);
    }
}

void MainWindow::refreshDynamicPaletteUI()
{
    if (!m_dynamicSetCombo) {
        return;
    }
    QSignalBlocker blocker(m_dynamicSetCombo);
    m_dynamicSetCombo->clear();
    for (int i = 0; i < MAX_DYNA_SETS_PER_FRAMEN; ++i) {
        m_dynamicSetCombo->addItem(QString("Dynamic %1").arg(i + 1), i);
    }
    if (m_dynamicSetIndex < 0 || m_dynamicSetIndex >= MAX_DYNA_SETS_PER_FRAMEN) {
        m_dynamicSetIndex = 0;
    }
    m_dynamicSetCombo->setCurrentIndex(m_dynamicSetIndex);
    refreshDynamicPaletteButtons();
}

void MainWindow::refreshDynamicPaletteButtons()
{
    if (!m_dynamicPaletteList) {
        return;
    }
    const int frameIndex = m_framesList ? m_framesList->currentRow() : -1;
    const bool hasFrame = frameIndex >= 0 && frameIndex < static_cast<int>(m_frameDynamicColors.size());
    const int slotCount = dynamicSlotCount();
    if (m_dynamicSlotIndex >= slotCount) {
        m_dynamicSlotIndex = -1;
    }
    QSignalBlocker blocker(m_dynamicPaletteList);
    if (m_dynamicPaletteList->count() != 16) {
        m_dynamicPaletteList->clear();
        for (int i = 0; i < 16; ++i) {
            auto* item = new QListWidgetItem();
            item->setData(Qt::UserRole, i);
            item->setSizeHint(m_dynamicPaletteList->gridSize());
            m_dynamicPaletteList->addItem(item);
        }
    }
    for (int i = 0; i < m_dynamicPaletteList->count(); ++i) {
        QListWidgetItem* item = m_dynamicPaletteList->item(i);
        if (!item) {
            continue;
        }
        const bool enabled = hasFrame && i < slotCount;
        Qt::ItemFlags flags = item->flags();
        if (enabled) {
            flags |= Qt::ItemIsEnabled | Qt::ItemIsSelectable;
        } else {
            flags &= ~Qt::ItemIsEnabled;
            flags &= ~Qt::ItemIsSelectable;
        }
        item->setFlags(flags);
        item->setData(kPaletteDisabledRole, !enabled);
        QPixmap pixmap(kPaletteSwatchSize, kPaletteSwatchSize);
        const QColor color = enabled ? dynamicSlotColor(i) : QColor(32, 32, 32);
        pixmap.fill(color);
        item->setIcon(QIcon(pixmap));
    }
    if (m_dynamicSlotIndex >= 0 && m_dynamicSlotIndex < m_dynamicPaletteList->count()) {
        m_dynamicPaletteList->setCurrentRow(m_dynamicSlotIndex);
    } else {
        m_dynamicPaletteList->setCurrentRow(-1);
    }
    if (m_dynamicAssignButton) {
        m_dynamicAssignButton->setEnabled(m_dynamicSlotIndex >= 0 && hasFrame);
    }
}

void MainWindow::syncDynamicSetSelection()
{
    int desired = -1;
    if (m_dynamicMaskList && m_dynamicMaskList->currentRow() >= 0) {
        desired = m_dynamicMaskList->currentRow();
    }
    if (desired < 0) {
        desired = m_dynamicSetIndex;
    }
    if (desired < 0 || desired >= MAX_DYNA_SETS_PER_FRAMEN) {
        desired = 0;
    }
    m_dynamicSetIndex = desired;
    if (m_dynamicSetCombo) {
        QSignalBlocker blocker(m_dynamicSetCombo);
        m_dynamicSetCombo->setCurrentIndex(m_dynamicSetIndex);
    }
}

void MainWindow::setDrawColor(const QColor& color, bool updatePaletteSelection)
{
    const cv::Vec3b bgr = cv::Vec3b(color.blue(), color.green(), color.red());
    const uint16_t rgb565 = BgrToRgb565(bgr);
    const cv::Vec3b quant = Rgb565ToBgr(rgb565);
    m_drawColor = cv::Scalar(quant[0], quant[1], quant[2], 255);
    updateCurrentColorSwatch();

    if (m_colorInfoLabel) {
        m_colorInfoLabel->setText(QString("RGB565: 0x%1\nRGB: %2,%3,%4")
                                      .arg(rgb565, 4, 16, QLatin1Char('0'))
                                      .arg(quant[2])
                                      .arg(quant[1])
                                      .arg(quant[0])
                                      .toUpper());
    }
        if (updatePaletteSelection && m_paletteList) {
            const QColor quantColor(quant[2], quant[1], quant[0]);
            int match = -1;
        for (int i = 0; i < m_paletteColors.size(); ++i) {
            if (m_paletteColors[i] == quantColor) {
                match = i;
                break;
            }
        }
        if (match >= 0) {
            m_currentPaletteIndex = match;
            m_paletteList->setCurrentRow(match);
        }
    }
}

void MainWindow::updateCurrentColorSwatch()
{
    if (!m_currentColorButton) {
        return;
    }
    const QColor color(static_cast<int>(m_drawColor[2]),
                       static_cast<int>(m_drawColor[1]),
                       static_cast<int>(m_drawColor[0]));
    m_currentColorButton->setStyleSheet(
        QString("QToolButton { background-color: %1; border: 1px solid #444; }").arg(color.name()));
}

void MainWindow::loadPaletteFromProject(const LegacyProject& legacy)
{
    initPalette();
    if (!legacy.palettes.empty() && legacy.palettes.size() >= static_cast<std::size_t>(N_PALETTES * 64)) {
        QVector<QVector<QColor>> loaded;
        loaded.resize(N_PALETTES);
        for (int p = 0; p < N_PALETTES; ++p) {
            loaded[p].reserve(64);
            const std::size_t base = static_cast<std::size_t>(p) * 64;
            for (int i = 0; i < 64; ++i) {
                const uint16_t value = legacy.palettes[base + static_cast<std::size_t>(i)];
                const cv::Vec3b bgr = Rgb565ToBgr(value);
                loaded[p].push_back(QColor(bgr[2], bgr[1], bgr[0]));
            }
        }
        m_fullPalettes = loaded;
    }
    if (!legacy.palette_names.empty() && legacy.palette_names.size() >= static_cast<std::size_t>(N_PALETTES)) {
        m_paletteNames.resize(N_PALETTES);
        for (int i = 0; i < N_PALETTES; ++i) {
            m_paletteNames[i] = QString::fromStdString(legacy.palette_names[static_cast<std::size_t>(i)]);
        }
    }
    if (m_paletteSetIndex < 0 || m_paletteSetIndex >= m_fullPalettes.size()) {
        m_paletteSetIndex = 0;
    }
    if (!m_fullPalettes.isEmpty()) {
        m_paletteColors = m_fullPalettes[m_paletteSetIndex];
    }
    m_reducedPaletteIndices = legacy.reduced_palettes;
    m_reducedPaletteNames = legacy.reduced_palette_names;
    if (m_reducedPaletteIndices.size() < static_cast<std::size_t>(kReducedPaletteCount * 16)) {
        m_reducedPaletteIndices.resize(static_cast<std::size_t>(kReducedPaletteCount * 16), 0);
    }
    if (m_reducedPaletteNames.size() < static_cast<std::size_t>(kReducedPaletteCount)) {
        m_reducedPaletteNames.resize(static_cast<std::size_t>(kReducedPaletteCount), std::string());
    }
    m_reducedPaletteIndex = legacy.active_reduced_palette < kReducedPaletteCount
        ? legacy.active_reduced_palette
        : 0;
    m_currentPaletteIndex = 0;
    refreshPaletteList();
    refreshReducedPaletteUI();
    refreshDynamicPaletteUI();
    if (!m_paletteColors.isEmpty()) {
        setDrawColor(m_paletteColors.front(), true);
    } else {
        updateCurrentColorSwatch();
    }
}

void MainWindow::refreshBackgroundList()
{
    if (!m_backgroundList) {
        return;
    }
    QSignalBlocker blocker(m_backgroundList);
    m_backgroundList->clear();

    const int count = m_backgroundStore ? m_backgroundStore->count() : 0;
    if (count <= 0) {
        auto* item = new QListWidgetItem("No backgrounds");
        item->setFlags(item->flags() & ~Qt::ItemIsEnabled);
        m_backgroundList->addItem(item);
        if (m_backgroundsCanvas) {
            m_backgroundsCanvas->setTitle("Backgrounds canvas (placeholder)");
            m_backgroundsCanvas->setStatusText("No backgrounds loaded");
            m_backgroundsCanvas->setImage(cv::Mat());
            m_backgroundsCanvas->setHdButtonEnabled(false);
            m_backgroundsCanvas->setHdButtonChecked(false);
        }
        m_useHdBackground = false;
        if (m_frameBackgroundAssign) {
            QSignalBlocker blockAssign(m_frameBackgroundAssign);
            m_frameBackgroundAssign->clear();
            m_frameBackgroundAssign->addItem("None", -1);
            m_frameBackgroundAssign->setEnabled(false);
        }
        return;
    }

    for (int i = 0; i < count; ++i) {
        const cv::Mat* image = m_backgroundStore->at(i);
        if (!image || image->empty()) {
            continue;
        }
        cv::Mat rgb;
        const bool hasHd = hasHdBackground(i);
        cv::Mat hd;
        if (hasHd) {
            hd = m_backgroundFramesX[static_cast<std::size_t>(i)];
        }
        const QColor gap = m_backgroundList->palette().color(QPalette::Window);
        cv::Mat previewMat = BuildBackgroundPreview(*image,
                                                    hd,
                                                    cv::Scalar(gap.blue(), gap.green(), gap.red()));
        cv::cvtColor(previewMat, rgb, cv::COLOR_BGR2RGB);
        QImage previewImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
        const QSize iconSize = hasHd
            ? QSize(kPreviewIconWidthHd, kPreviewIconHeightHd)
            : QSize(kPreviewIconWidth, kPreviewIconHeight);
        QPixmap pixmap = QPixmap::fromImage(previewImage.copy());
        pixmap = pixmap.scaled(iconSize, Qt::KeepAspectRatio, Qt::FastTransformation);

        auto* item = new QListWidgetItem();
        item->setIcon(QIcon(pixmap));
        item->setText(QString("BG %1").arg(i));
        item->setData(kPreviewIconSizeRole, iconSize);
        item->setSizeHint(PreviewItemSizeForIcon(iconSize, m_backgroundList->font()));
        item->setData(Qt::UserRole, i);
        item->setData(Qt::UserRole + 1, QStringLiteral("background"));
        m_backgroundList->addItem(item);
    }

    int targetRow = m_backgroundList->currentRow();
    if (targetRow < 0) {
        targetRow = m_lastBackgroundIndex;
    }
    if (targetRow >= 0 && targetRow < m_backgroundList->count()) {
        m_backgroundList->setCurrentRow(targetRow);
    } else if (m_backgroundList->count() > 0 && m_lastBackgroundIndex < 0) {
        m_backgroundList->setCurrentRow(0);
    }
    if (m_frameBackgroundAssign) {
        QSignalBlocker blockAssign(m_frameBackgroundAssign);
        m_frameBackgroundAssign->clear();
        m_frameBackgroundAssign->addItem("None", -1);
        for (int i = 0; i < count; ++i) {
            m_frameBackgroundAssign->addItem(QString("BG %1").arg(i), i);
        }
        m_frameBackgroundAssign->setEnabled(true);
        const int currentFrame = m_framesList ? m_framesList->currentRow() : -1;
        if (currentFrame >= 0 && currentFrame < static_cast<int>(m_frameBackgroundIds.size())) {
            const uint16_t bgId = m_frameBackgroundIds[static_cast<std::size_t>(currentFrame)];
            const int comboIndex = (bgId == 0xffff || bgId >= static_cast<uint16_t>(count))
                ? 0
                : static_cast<int>(bgId) + 1;
            m_frameBackgroundAssign->setCurrentIndex(comboIndex);
        }
    }
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
        if (m_framesCanvas) {
            m_framesCanvas->setHdButtonEnabled(false);
            m_framesCanvas->setHdButtonChecked(false);
        }
        if (m_hdCreateButton) {
            m_hdCreateButton->setEnabled(false);
        }
        if (m_hdDeleteButton) {
            m_hdDeleteButton->setEnabled(false);
        }
        if (m_hdSourceCombo) {
            m_hdSourceCombo->setEnabled(false);
        }
        if (m_hdScaleCombo) {
            m_hdScaleCombo->setEnabled(false);
        }
        updateHdControlsForContext();
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

    const int frameCount = m_frameStore->count();
    if (frameCount >= 0) {
        if (m_frameExtraFrames.size() != static_cast<std::size_t>(frameCount)) {
            m_frameExtraFrames.resize(static_cast<std::size_t>(frameCount));
        }
        if (m_frameExtraFlags.size() != static_cast<std::size_t>(frameCount)) {
            m_frameExtraFlags.resize(static_cast<std::size_t>(frameCount), 0);
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
    if (m_backgroundList && m_backgroundList->currentRow() >= 0) {
        setInspectorSelection(QString("Background: %1").arg(m_backgroundList->currentItem()->text()));
        showBackgroundAtIndex(m_backgroundList->currentRow());
        return;
    }
    setInspectorSelection("None");
    updateMetadataForFrame(-1);
    updateMetadataForSprite(-1);
    updateUndoActions();
    if (m_maskMode == MaskMode::None) {
        m_framesCanvas->canvas()->clearPreviewImage();
    }

    refreshBackgroundList();
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
        if (m_frameBackgroundAssign && index >= 0 && index < static_cast<int>(m_frameBackgroundIds.size())) {
            QSignalBlocker blockAssign(m_frameBackgroundAssign);
            const uint16_t bgId = m_frameBackgroundIds[static_cast<std::size_t>(index)];
            m_frameBackgroundAssign->setCurrentIndex(bgId == 0xffff ? 0 : static_cast<int>(bgId) + 1);
        }
        if (m_shapeCompToggle && index >= 0 && index < static_cast<int>(m_frameShapeCompModes.size())) {
            QSignalBlocker blockShape(m_shapeCompToggle);
            const uint8_t value = m_frameShapeCompModes[static_cast<std::size_t>(index)];
            m_shapeCompToggle->setChecked(value != 0);
        }
        const bool hasHd = hasHdFrame(index);
        if (!hasHd && m_useHdFrame) {
            m_useHdFrame = false;
        }
        if (m_framesCanvas) {
            m_framesCanvas->setHdButtonEnabled(hasHd);
            m_framesCanvas->setHdButtonChecked(m_useHdFrame);
            m_framesCanvas->setBackgroundChecked(m_showBackgroundLayer);
        }
        updateHdControlsForContext();
        updateMaskPreviewForFrame(index);
        syncDynamicSetSelection();
        refreshDynamicPaletteButtons();
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

void MainWindow::showBackgroundAtIndex(int index)
{
    if (!m_backgroundsCanvas) {
        return;
    }
    const bool hasHd = hasHdBackground(index);
    if (!hasHd && m_useHdBackground) {
        m_useHdBackground = false;
    }
    m_backgroundsCanvas->setHdButtonEnabled(hasHd);
    m_backgroundsCanvas->setHdButtonChecked(m_useHdBackground && hasHd);
    m_backgroundsCanvas->canvas()->clearPreviewImage();
    updateBackgroundCanvasImage(index);
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
    if (isFrame) {
        const int index = m_framesList->currentRow();
        if (index < 0) {
            return;
        }
        const cv::Mat* topFrame = activeFrameImage(index, false);
        if (!topFrame || topFrame->empty()) {
            return;
        }
        cv::Mat reference = buildOriginalPreviewForIndex(index);
        FrameLayout layout = BuildFrameLayout(*topFrame, reference);
        int localX = x;
        int localY = y;
        m_frameDrawOnMask = false;
        if (m_backgroundMaskMode) {
            ensureBackgroundDataSize();
            if (index >= static_cast<int>(m_frameBackgroundIds.size()) ||
                m_frameBackgroundIds[static_cast<std::size_t>(index)] == 0xffff) {
                statusBar()->showMessage("Assign a background to this frame before editing its mask.", 2000);
                return;
            }
            if (y < 0 || y >= layout.topHeight ||
                x < layout.topX || x >= layout.topX + layout.topWidth) {
                return;
            }
            localX = x - layout.topX;
            localY = y;
            cv::Mat* mask = activeBackgroundMask(index);
            if (!mask || mask->empty()) {
                return;
            }
            if (m_drawTool == DrawTool::MagicFill || m_drawTool == DrawTool::Point) {
                if (localX < 0 || localY < 0 || localX >= mask->cols || localY >= mask->rows) {
                    return;
                }
                if (!m_frameUndoActive) {
                    pushBackgroundMaskUndoSnapshot(index);
                    m_frameUndoActive = true;
                }
                if (m_drawTool == DrawTool::MagicFill) {
                    applyMaskFill(*mask, localX, localY, button == Qt::RightButton);
                } else {
                    applyToolToMask(*mask,
                                    DrawTool::Point,
                                    QPoint(localX, localY),
                                    QPoint(localX, localY),
                                    button == Qt::RightButton);
                }
                updateFrameCanvasImage(index);
                updateFramePreviewAt(index);
                updateMaskPreviewForFrame(index);
                return;
            }
            if (m_drawTool == DrawTool::ColorPicker) {
                return;
            }
            if (!m_frameUndoActive) {
                pushBackgroundMaskUndoSnapshot(index);
                m_frameUndoActive = true;
            }
            m_frameStart = QPoint(localX, localY);
            m_frameHasStart = true;
            m_frameStartButton = button;
            return;
        }
        if (!m_showOriginalFrame) {
            m_frameDrawOnMask = false;
        } else if (m_maskMode != MaskMode::None) {
            const int gap = kFrameGapPixels;
            if (y >= layout.topHeight + gap &&
                y < layout.topHeight + gap + layout.bottomHeight &&
                x >= layout.bottomX && x < layout.bottomX + layout.bottomWidth) {
                m_frameDrawOnMask = true;
                localX = x - layout.bottomX;
                localY = y - (layout.topHeight + gap);
            } else {
                m_frameDrawOnMask = false;
            }
        } else {
            m_frameDrawOnMask = false;
        }
        if (m_frameDrawOnMask) {
            ensureMaskDataSize();
            cv::Mat* mask = nullptr;
            if (m_maskMode == MaskMode::Comparison) {
                const int assigned = currentFrameMaskId();
                const int selected = m_maskList->currentRow();
                if (assigned < 0 || assigned != selected) {
                    statusBar()->showMessage("Assign the selected mask to this frame before editing.", 2000);
                    return;
                }
                mask = activeComparisonMask();
            } else if (m_maskMode == MaskMode::Dynamic) {
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
                const int scaleX = (layout.bottomWidth > 0 && mask->cols > 0) ? layout.bottomWidth / mask->cols : 1;
                const int scaleY = (layout.bottomHeight > 0 && mask->rows > 0) ? layout.bottomHeight / mask->rows : 1;
                const int mappedX = (scaleX > 1) ? localX / scaleX : localX;
                const int mappedY = (scaleY > 1) ? localY / scaleY : localY;
                if (mappedX < 0 || mappedY < 0 || mappedX >= mask->cols || mappedY >= mask->rows) {
                    return;
                }
                if (!m_frameUndoActive) {
                    pushMaskUndoSnapshot(m_maskMode, index);
                    m_frameUndoActive = true;
                }
                if (m_drawTool == DrawTool::MagicFill) {
                    applyMaskFill(*mask, mappedX, mappedY, button == Qt::RightButton);
                } else {
                    applyToolToMask(*mask,
                                    DrawTool::Point,
                                    QPoint(mappedX, mappedY),
                                    QPoint(mappedX, mappedY),
                                    button == Qt::RightButton);
                }
                if (m_maskMode == MaskMode::Comparison) {
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
                pushMaskUndoSnapshot(m_maskMode, index);
                m_frameUndoActive = true;
            }
            const int scaleX = (layout.bottomWidth > 0 && mask->cols > 0) ? layout.bottomWidth / mask->cols : 1;
            const int scaleY = (layout.bottomHeight > 0 && mask->rows > 0) ? layout.bottomHeight / mask->rows : 1;
            const int mappedX = (scaleX > 1) ? localX / scaleX : localX;
            const int mappedY = (scaleY > 1) ? localY / scaleY : localY;
            m_frameStart = QPoint(mappedX, mappedY);
            m_frameHasStart = true;
            m_frameStartButton = button;
            return;
        }
    }
    cv::Mat* image = isFrame ? activeFrameImage(m_framesList->currentRow(), true)
                             : m_spriteStore->atMutable(m_spritesList->currentRow());
    if (!image || image->empty()) {
        return;
    }
    if (isFrame) {
        const int index = m_framesList->currentRow();
        cv::Mat reference = buildOriginalPreviewForIndex(index);
        FrameLayout layout = BuildFrameLayout(*image, reference);
        if (y < 0 || y >= layout.topHeight ||
            x < layout.topX || x >= layout.topX + layout.topWidth) {
            statusBar()->showMessage("Color edits apply to the top frame only.", 2000);
            return;
        }
        x -= layout.topX;
        m_frameDrawOnMask = false;
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
    if (isFrame && m_backgroundMaskMode) {
        const int index = m_framesList->currentRow();
        if (index < 0) {
            return;
        }
        const cv::Mat* frameImage = activeFrameImage(index, false);
        if (!frameImage || frameImage->empty()) {
            return;
        }
        cv::Mat reference = buildOriginalPreviewForIndex(index);
        FrameLayout layout = BuildFrameLayout(*frameImage, reference);
        if (y < 0 || y >= layout.topHeight ||
            x < layout.topX || x >= layout.topX + layout.topWidth) {
            return;
        }
        x -= layout.topX;
        cv::Mat* mask = activeBackgroundMask(index);
        if (!mask || mask->empty()) {
            return;
        }
        if (m_drawTool == DrawTool::Point) {
            const bool erase = buttons.testFlag(Qt::RightButton);
            applyToolToMask(*mask, DrawTool::Point, QPoint(x, y), QPoint(x, y), erase);
            updateFrameCanvasImage(index);
            updateFramePreviewAt(index);
            return;
        }
        if (!m_frameHasStart) {
            return;
        }
        cv::Mat previewMask = mask->clone();
        const bool erase = buttons.testFlag(Qt::RightButton);
        applyToolToMask(previewMask, m_drawTool, m_frameStart, QPoint(x, y), erase);
        cv::Mat base = m_showBackgroundLayer
            ? applyBackgroundComposite(index, *frameImage, m_useHdFrame)
            : EnsureBgr(*frameImage);
        cv::Mat topPreview = buildMaskPreview(base, previewMask, cv::Vec3b(60, 200, 120));
        QRect outlineRegion;
        if (m_showOriginalFrame) {
            cv::Mat original = buildOriginalFrame(reference);
            cv::Mat displayOriginal = BuildDisplayOriginal(original, topPreview.size());
            FrameLayout layout = BuildFrameLayout(topPreview, displayOriginal);
            outlineRegion = QRect(layout.topX, 0, layout.topWidth, layout.topHeight);
        }
        m_framesCanvas->canvas()->setMaskOutline(previewMask, QColor(120, 200, 60), outlineRegion);
        if (m_showOriginalFrame) {
            cv::Mat original = buildOriginalFrame(reference);
            cv::Mat displayOriginal = BuildDisplayOriginal(original, topPreview.size());
            const QColor gap = m_framesCanvas->palette().color(QPalette::Window);
            cv::Mat combined = buildCombinedFrame(topPreview,
                                                  displayOriginal,
                                                  cv::Scalar(gap.blue(), gap.green(), gap.red()));
            m_framesCanvas->canvas()->setPreviewImage(combined);
        } else {
            m_framesCanvas->canvas()->setPreviewImage(topPreview);
        }
        return;
    }
    if (isFrame && m_frameDrawOnMask) {
        const int index = m_framesList->currentRow();
        if (index < 0) {
            return;
        }
        const cv::Mat* frameImage = activeFrameImage(index, false);
        if (!frameImage || frameImage->empty()) {
            return;
        }
        cv::Mat reference = buildOriginalPreviewForIndex(index);
        const int bottomWidth = reference.cols > 0 && frameImage->cols == reference.cols * 2 ? frameImage->cols : reference.cols;
        const int bottomHeight = reference.rows > 0 && frameImage->rows == reference.rows * 2 ? frameImage->rows : reference.rows;
        FrameLayout layout = BuildFrameLayout(frameImage->cols, frameImage->rows, bottomWidth, bottomHeight);
        const int gap = kFrameGapPixels;
        if (y < layout.topHeight + gap || y >= layout.topHeight + gap + layout.bottomHeight ||
            x < layout.bottomX || x >= layout.bottomX + layout.bottomWidth) {
            return;
        }
        x -= layout.bottomX;
        y -= (layout.topHeight + gap);
        cv::Mat* mask = nullptr;
        cv::Vec3b color(200, 0, 200);
        if (m_maskMode == MaskMode::Comparison) {
            const int assigned = currentFrameMaskId();
            const int selected = m_maskList->currentRow();
            if (assigned < 0 || assigned != selected) {
                return;
            }
            mask = activeComparisonMask();
            color = cv::Vec3b(200, 0, 200);
        } else if (m_maskMode == MaskMode::Dynamic) {
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
        const int scaleX = (layout.bottomWidth > 0 && mask->cols > 0) ? layout.bottomWidth / mask->cols : 1;
        const int scaleY = (layout.bottomHeight > 0 && mask->rows > 0) ? layout.bottomHeight / mask->rows : 1;
        if (scaleX > 1) {
            x /= scaleX;
        }
        if (scaleY > 1) {
            y /= scaleY;
        }
        if (m_drawTool == DrawTool::Point) {
            const bool erase = buttons.testFlag(Qt::RightButton);
            applyToolToMask(*mask, DrawTool::Point, QPoint(x, y), QPoint(x, y), erase);
            if (m_maskMode == MaskMode::Comparison) {
                updateMaskPreviewIcons();
                m_framesCanvas->canvas()->setMaskOutline(*mask, QColor(200, 0, 200));
            } else {
                updateDynamicMaskPreviewIcons();
                m_framesCanvas->canvas()->setMaskOutline(*mask, QColor(255, 200, 0));
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
        if (const cv::Mat* frame = activeFrameImage(index, false)) {
            const QColor gap = m_framesCanvas->palette().color(QPalette::Window);
            const cv::Scalar gapColor(gap.blue(), gap.green(), gap.red());
            const cv::Mat base = m_showBackgroundLayer
                ? applyBackgroundComposite(index, *frame, m_useHdFrame)
                : EnsureBgr(*frame);
            cv::Mat preview = buildCombinedMaskPreview(base, reference, previewMask, color, gapColor);
            m_framesCanvas->canvas()->setPreviewImage(preview);
            QRect outlineRegion;
            if (m_showOriginalFrame) {
                outlineRegion = QRect(layout.bottomX,
                                      layout.topHeight + kFrameGapPixels,
                                      layout.bottomWidth,
                                      layout.bottomHeight);
            }
            if (m_maskMode == MaskMode::Comparison) {
                m_framesCanvas->canvas()->setMaskOutline(previewMask, QColor(200, 0, 200), outlineRegion);
            } else {
                m_framesCanvas->canvas()->setMaskOutline(previewMask, QColor(255, 200, 0), outlineRegion);
            }
        }
        return;
    }
    cv::Mat* image = isFrame ? activeFrameImage(m_framesList->currentRow(), true)
                             : m_spriteStore->atMutable(m_spritesList->currentRow());
    if (!image || image->empty()) {
        return;
    }
    if (isFrame) {
        const int index = m_framesList->currentRow();
        cv::Mat reference = buildOriginalPreviewForIndex(index);
        FrameLayout layout = BuildFrameLayout(*image, reference);
        if (y < 0 || y >= layout.topHeight ||
            x < layout.topX || x >= layout.topX + layout.topWidth) {
            return;
        }
        x -= layout.topX;
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
        cv::Mat composed = m_showBackgroundLayer
            ? applyBackgroundComposite(m_framesList->currentRow(), preview, m_useHdFrame)
            : EnsureBgr(preview);
        if (original.empty()) {
            m_framesCanvas->canvas()->setPreviewImage(composed);
        } else {
            const QColor gap = m_framesCanvas->palette().color(QPalette::Window);
            cv::Mat displayOriginal = BuildDisplayOriginal(original, composed.size());
            cv::Mat combined = buildCombinedFrame(composed,
                                                  displayOriginal,
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
    if (isFrame && m_backgroundMaskMode) {
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
        const cv::Mat* frameImage = activeFrameImage(index, false);
        if (!frameImage || frameImage->empty()) {
            m_frameHasStart = false;
            return;
        }
        cv::Mat reference = buildOriginalPreviewForIndex(index);
        FrameLayout layout = BuildFrameLayout(*frameImage, reference);
        if (y < 0 || y >= layout.topHeight ||
            x < layout.topX || x >= layout.topX + layout.topWidth) {
            m_frameHasStart = false;
            return;
        }
        x -= layout.topX;
        cv::Mat* mask = activeBackgroundMask(index);
        if (!mask || mask->empty()) {
            m_frameHasStart = false;
            return;
        }
        const bool erase = (m_frameStartButton == Qt::RightButton || button == Qt::RightButton);
        applyToolToMask(*mask, m_drawTool, m_frameStart, QPoint(x, y), erase);
        m_frameHasStart = false;
        m_framesCanvas->canvas()->clearPreviewImage();
        updateFrameCanvasImage(index);
        updateFramePreviewAt(index);
        updateMaskPreviewForFrame(index);
        m_frameUndoActive = false;
        return;
    }
    if (isFrame && m_frameDrawOnMask) {
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
        const cv::Mat* frameImage = activeFrameImage(index, false);
        if (!frameImage || frameImage->empty()) {
            m_frameHasStart = false;
            return;
        }
        cv::Mat reference = buildOriginalPreviewForIndex(index);
        const int bottomWidth = reference.cols > 0 && frameImage->cols == reference.cols * 2 ? frameImage->cols : reference.cols;
        const int bottomHeight = reference.rows > 0 && frameImage->rows == reference.rows * 2 ? frameImage->rows : reference.rows;
        FrameLayout layout = BuildFrameLayout(frameImage->cols, frameImage->rows, bottomWidth, bottomHeight);
        const int gap = kFrameGapPixels;
        if (y < layout.topHeight + gap || y >= layout.topHeight + gap + layout.bottomHeight ||
            x < layout.bottomX || x >= layout.bottomX + layout.bottomWidth) {
            m_frameHasStart = false;
            return;
        }
        x -= layout.bottomX;
        y -= (layout.topHeight + gap);
        cv::Mat* mask = nullptr;
        if (m_maskMode == MaskMode::Comparison) {
            const int assigned = currentFrameMaskId();
            const int selected = m_maskList->currentRow();
            if (assigned < 0 || assigned != selected) {
                m_frameHasStart = false;
                return;
            }
            mask = activeComparisonMask();
        } else if (m_maskMode == MaskMode::Dynamic) {
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
        const int scaleX = (layout.bottomWidth > 0 && mask->cols > 0) ? layout.bottomWidth / mask->cols : 1;
        const int scaleY = (layout.bottomHeight > 0 && mask->rows > 0) ? layout.bottomHeight / mask->rows : 1;
        if (scaleX > 1) {
            x /= scaleX;
        }
        if (scaleY > 1) {
            y /= scaleY;
        }
        const bool erase = (m_frameStartButton == Qt::RightButton || button == Qt::RightButton);
        applyToolToMask(*mask, m_drawTool, m_frameStart, QPoint(x, y), erase);
        m_frameHasStart = false;
        m_framesCanvas->canvas()->clearPreviewImage();
        if (m_maskMode == MaskMode::Comparison) {
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
    cv::Mat* image = isFrame ? activeFrameImage(m_framesList->currentRow(), true)
                             : m_spriteStore->atMutable(m_spritesList->currentRow());
    if (!image || image->empty()) {
        return;
    }
    if (isFrame) {
        const int index = m_framesList->currentRow();
        cv::Mat reference = buildOriginalPreviewForIndex(index);
        FrameLayout layout = BuildFrameLayout(*image, reference);
        if (y < 0 || y >= layout.topHeight ||
            x < layout.topX || x >= layout.topX + layout.topWidth) {
            if (isFrame) {
                m_frameHasStart = false;
            }
            return;
        }
        x -= layout.topX;
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

void MainWindow::handleBackgroundToolPress(int x, int y, Qt::MouseButton button)
{
    if (!m_drawPointEnabled || !m_backgroundStore) {
        return;
    }
    const int index = m_backgroundList ? m_backgroundList->currentRow() : -1;
    if (index < 0) {
        return;
    }
    cv::Mat* image = activeBackgroundImage(index, true);
    if (!image || image->empty()) {
        return;
    }
    if (m_drawTool != DrawTool::ColorPicker) {
        if (!m_backgroundUndoActive) {
            pushBackgroundUndoSnapshot(index);
            m_backgroundUndoActive = true;
        }
    }
    if (m_drawTool == DrawTool::Point) {
        applyToolToImage(*image, DrawTool::Point, QPoint(x, y), QPoint(x, y), button == Qt::RightButton);
    } else if (m_drawTool == DrawTool::ColorPicker) {
        pickColorFromImage(*image, x, y);
    } else if (m_drawTool == DrawTool::MagicFill) {
        applyMagicFill(*image, x, y);
    } else {
        m_spriteStart = QPoint(x, y);
        m_spriteHasStart = true;
        m_spriteStartButton = button;
    }
    updateBackgroundCanvasImage(index);
}

void MainWindow::handleBackgroundToolDrag(int x, int y, Qt::MouseButtons buttons)
{
    if (!m_drawPointEnabled || !m_backgroundStore) {
        return;
    }
    const int index = m_backgroundList ? m_backgroundList->currentRow() : -1;
    if (index < 0) {
        return;
    }
    cv::Mat* image = activeBackgroundImage(index, true);
    if (!image || image->empty()) {
        return;
    }
    if (m_drawTool == DrawTool::Point) {
        const bool erase = buttons.testFlag(Qt::RightButton);
        applyToolToImage(*image, DrawTool::Point, QPoint(x, y), QPoint(x, y), erase);
        updateBackgroundCanvasImage(index);
        return;
    }
    if (!m_spriteHasStart) {
        return;
    }
    const QPoint start = m_spriteStart;
    const bool erase = buttons.testFlag(Qt::RightButton);
    cv::Mat preview = image->clone();
    applyToolToImage(preview, m_drawTool, start, QPoint(x, y), erase);
    m_backgroundsCanvas->canvas()->setPreviewImage(preview);
}

void MainWindow::handleBackgroundToolRelease(int x, int y, Qt::MouseButton button)
{
    if (!m_drawPointEnabled || !m_backgroundStore) {
        return;
    }
    const int index = m_backgroundList ? m_backgroundList->currentRow() : -1;
    if (index < 0) {
        return;
    }
    cv::Mat* image = activeBackgroundImage(index, true);
    if (!image || image->empty()) {
        return;
    }
    if (m_drawTool == DrawTool::Point ||
        m_drawTool == DrawTool::ColorPicker ||
        m_drawTool == DrawTool::MagicFill) {
        m_backgroundUndoActive = false;
        return;
    }
    if (!m_spriteHasStart) {
        return;
    }
    const QPoint start = m_spriteStart;
    const Qt::MouseButton startButton = m_spriteStartButton;
    m_spriteHasStart = false;
    const bool erase = (startButton == Qt::RightButton || button == Qt::RightButton);
    applyToolToImage(*image, m_drawTool, start, QPoint(x, y), erase);
    m_backgroundsCanvas->canvas()->clearPreviewImage();
    updateBackgroundCanvasImage(index);
    m_backgroundUndoActive = false;
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
    if (x < 0 || y < 0 || x >= image.cols || y >= image.rows) {
        return;
    }
    QColor picked;
    if (image.type() == CV_8UC3) {
        const cv::Vec3b color = image.at<cv::Vec3b>(y, x);
        picked = QColor(color[2], color[1], color[0]);
    } else if (image.type() == CV_8UC4) {
        const cv::Vec4b color = image.at<cv::Vec4b>(y, x);
        picked = QColor(color[2], color[1], color[0], color[3]);
    } else if (image.type() == CV_8UC1) {
        const uint8_t value = image.at<uint8_t>(y, x);
        picked = QColor(value, value, value);
    } else {
        return;
    }
    setDrawColor(picked, true);
    statusBar()->showMessage(QString("Picked color: R%1 G%2 B%3")
                                 .arg(static_cast<int>(m_drawColor[2]))
                                 .arg(static_cast<int>(m_drawColor[1]))
                                 .arg(static_cast<int>(m_drawColor[0])),
                             2000);
}

void MainWindow::cancelCurrentDraw()
{
    m_frameHasStart = false;
    m_spriteHasStart = false;
    m_frameUndoActive = false;
    m_spriteUndoActive = false;
    m_frameDrawOnMask = false;
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
