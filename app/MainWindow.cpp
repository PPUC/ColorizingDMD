#include "MainWindow.h"

#include <QAction>
#include <QApplication>
#include <QActionGroup>
#include <QAbstractItemView>
#include <QAbstractItemModel>
#include <QComboBox>
#include <QDockWidget>
#include <QDialog>
#include <QDialogButtonBox>
#include <QLabel>
#include <QProgressDialog>
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
#include <QKeyEvent>
#include <QMenu>
#include <QMenuBar>
#include <QDialog>
#include <QTextBrowser>
#include <QVBoxLayout>
#include <QMessageBox>
#include <QInputDialog>
#include <QSettings>
#include <QDateTime>
#include <QDir>
#include <QStandardPaths>
#include <QClipboard>
#include <QTimer>
#include <QCloseEvent>
#include <cstdint>
#include <cstdio>

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#if defined(_MSC_VER)
#pragma comment(lib, "psapi.lib")
#endif
#elif defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <unistd.h>
#endif
#include <QScopedValueRollback>
#include <QStackedWidget>
#include <QButtonGroup>
#include <QLayout>
#include <QDrag>
#include <QMouseEvent>
#include <QSpinBox>
#include <QStatusBar>
#include <QStyle>
#include <QTabWidget>
#include <QTimer>
#include <QToolBar>
#include <QToolButton>
#include <QScrollBar>
#include <QSize>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <cstring>
#include <memory>
#include <QFileInfo>
#include <QSignalBlocker>
#include <QStyledItemDelegate>
#include <QPainter>
#include <QPalette>
#include <QDragEnterEvent>
#include <QDragMoveEvent>
#include <QDropEvent>
#include <QMimeData>
#include <QDataStream>
#include <QMap>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstring>
#include <array>
#include <unordered_set>
#include <unordered_map>
#include <map>

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
#include "serum-editor.h"
#ifdef MIN
#undef MIN
#endif
#include "serum-decode.h"

namespace {
constexpr int kDefaultFrameWidth = 128;
constexpr int kDefaultFrameHeight = 32;
constexpr int kDefaultSpriteWidth = 64;
constexpr int kDefaultSpriteHeight = 64;
constexpr int kPreviewIconWidth = 160;
constexpr int kMonochromeTriggerId = 65432;
constexpr int kPreviewIconHeight = 120;
constexpr int kPreviewItemWidth = 180;
constexpr int kPreviewItemHeight = 150;
constexpr int kPreviewIconWidthHd = kPreviewIconWidth * 2;
constexpr int kPreviewIconHeightHd = kPreviewIconHeight * 2;
constexpr int kPreviewItemWidthHd = kPreviewIconWidthHd + 12;
constexpr int kPreviewItemHeightHd = kPreviewIconHeightHd + 28;
constexpr int kDynamicColorStripWidth = 64;
constexpr int kDynamicMaskIconWidth = kPreviewIconWidth + kDynamicColorStripWidth + 8;
constexpr int kDynamicMaskItemWidth = kDynamicMaskIconWidth + 12;
constexpr int kPaletteSwatchSize = 26;
constexpr int kPaletteItemSize = 32;
constexpr int kReducedPaletteCount = 64;
constexpr int kDefaultUndoDepth = 50;
constexpr int kDefaultHistoryDepth = 100;
constexpr int kFrameGapPixels = 4;

bool EnsureConcentrateExists(const QString& cromPath,
                             const QString& cromcPath,
                             QString* errorMessage);

constexpr int kFrameIndexRole = Qt::UserRole + 1;
constexpr int kFrameDurationRole = Qt::UserRole + 2;
constexpr int kPreviewIconSizeRole = Qt::UserRole + 3;
constexpr int kPaletteDisabledRole = Qt::UserRole + 4;
constexpr int kPreviewSecondarySelectedRole = Qt::UserRole + 5;
constexpr int kSpriteZoneIndexRole = Qt::UserRole + 6;
constexpr int kPreviewUsageRole = Qt::UserRole + 7;
constexpr int kSpriteZoneSlotRole = Qt::UserRole + 8;
constexpr int kSpriteZoneSpriteIndexRole = Qt::UserRole + 9;
constexpr int kPreviewPlaybackRole = Qt::UserRole + 10;

cv::Mat EnsureBgr(const cv::Mat& source);
uint16_t BgrToRgb565(const cv::Vec3b& color);
cv::Vec3b Rgb565ToBgr(uint16_t value);
std::vector<uint16_t> ConvertBgrMatToRgb565(const cv::Mat& source);
cv::Mat ConvertRgb565ToBgrMat(const uint16_t* data, int width, int height);
cv::Mat Scale2xBgr(const cv::Mat& source);

bool LoadReferenceFramesFromRp(const QString& rpPath,
                               int width,
                               int height,
                               int frameCount,
                               std::vector<cv::Mat>& out,
                               QString* error)
{
    if (rpPath.isEmpty()) {
        if (error) {
            *error = "Missing .cRP path";
        }
        return false;
    }
    QFile file(rpPath);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) {
            *error = QString("Could not open .cRP file: %1").arg(rpPath);
        }
        return false;
    }
    if (width <= 0 || height <= 0 || frameCount <= 0) {
        if (error) {
            *error = "Invalid frame dimensions";
        }
        return false;
    }
    char header[64] = {};
    if (file.read(header, sizeof(header)) != static_cast<qint64>(sizeof(header))) {
        if (error) {
            *error = "Invalid .cRP header";
        }
        return false;
    }
    const std::size_t pixelCount = static_cast<std::size_t>(frameCount) *
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (pixelCount == 0) {
        if (error) {
            *error = "Invalid .cRP frame size";
        }
        return false;
    }
    QByteArray data = file.read(static_cast<qint64>(pixelCount));
    if (data.size() != static_cast<qint64>(pixelCount)) {
        if (error) {
            *error = "Unexpected end of .cRP reference frames";
        }
        return false;
    }
    out.clear();
    out.reserve(static_cast<std::size_t>(frameCount));
    const uint8_t* src = reinterpret_cast<const uint8_t*>(data.constData());
    const std::size_t framePixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    for (int i = 0; i < frameCount; ++i) {
        cv::Mat ref(height, width, CV_8UC1);
        std::memcpy(ref.data, src + static_cast<std::size_t>(i) * framePixels, framePixels);
        out.push_back(ref);
    }
    return true;
}

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

int FrameGapForWidth(int width)
{
    (void)width;
    return kFrameGapPixels;
}

bool SpriteHasNonTransparentPixel(const uint8_t* data, std::size_t count)
{
    if (!data || count == 0) {
        return false;
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (data[i] != 255) {
            return true;
        }
    }
    return false;
}


cv::Mat BuildStackedFrames(const std::vector<cv::Mat>& frames,
                           const cv::Scalar& gapColor,
                           int gapPixels)
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
    height += static_cast<int>(valid.size() - 1) * gapPixels;
    cv::Mat combined(height, width, CV_8UC3, gapColor);
    int y = 0;
    for (const auto& frame : valid) {
        const int x = (width - frame.cols) / 2;
        frame.copyTo(combined(cv::Rect(x, y, frame.cols, frame.rows)));
        y += frame.rows + gapPixels;
    }
    return combined;
}

QPoint ScalePointToSize(const QPoint& point, const QSize& from, const QSize& to)
{
    if (from.width() <= 0 || from.height() <= 0 || to.width() <= 0 || to.height() <= 0) {
        return point;
    }
    const int x = std::clamp(static_cast<int>(std::llround(
        static_cast<double>(point.x()) * to.width() / from.width())), 0, to.width() - 1);
    const int y = std::clamp(static_cast<int>(std::llround(
        static_cast<double>(point.y()) * to.height() / from.height())), 0, to.height() - 1);
    return QPoint(x, y);
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

cv::Mat UpscaleSpriteToHd(const cv::Mat& source,
                          const QRect& contentRect,
                          int interpolation,
                          bool isMask,
                          uint8_t fillValue)
{
    if (source.empty()) {
        return cv::Mat();
    }
    const cv::Size targetSize(MAX_SPRITE_WIDTH, MAX_SPRITE_HEIGHT);
    cv::Mat output;
    if (source.channels() == 1) {
        output = cv::Mat(targetSize, CV_8UC1, cv::Scalar(fillValue));
    } else {
        output = cv::Mat(targetSize, CV_8UC3, cv::Scalar(0, 0, 0));
    }
    QRect srcRect = contentRect;
    if (!srcRect.isValid() || srcRect.isEmpty()) {
        srcRect = QRect(0, 0, source.cols, source.rows);
    }
    srcRect = srcRect.intersected(QRect(0, 0, source.cols, source.rows));
    if (srcRect.width() <= 0 || srcRect.height() <= 0) {
        return output;
    }
    const cv::Rect roi(srcRect.x(), srcRect.y(), srcRect.width(), srcRect.height());
    cv::Mat cropped = source(roi).clone();
    const int scaledWidth = srcRect.width() * 2;
    const int scaledHeight = srcRect.height() * 2;
    cv::Mat scaled;
    if (scaledWidth > targetSize.width || scaledHeight > targetSize.height) {
        if (isMask) {
            cv::resize(cropped, scaled, targetSize, 0.0, 0.0, cv::INTER_NEAREST);
        } else if (interpolation == -1) {
            scaled = Scale2xBgr(EnsureBgr(cropped));
            if (scaled.size() != targetSize) {
                cv::resize(EnsureBgr(cropped), scaled, targetSize, 0.0, 0.0, cv::INTER_NEAREST);
            }
        } else {
            cv::resize(EnsureBgr(cropped), scaled, targetSize, 0.0, 0.0, interpolation);
            if (interpolation == cv::INTER_LINEAR || interpolation == cv::INTER_CUBIC) {
                scaled = AdjustBrightnessToSource(EnsureBgr(cropped), scaled);
            }
        }
        if (!scaled.empty()) {
            scaled.copyTo(output);
        }
        return output;
    }

    if (isMask) {
        cv::resize(cropped, scaled, cv::Size(scaledWidth, scaledHeight), 0.0, 0.0, cv::INTER_NEAREST);
    } else if (interpolation == -1) {
        scaled = Scale2xBgr(EnsureBgr(cropped));
        if (scaled.size() != cv::Size(scaledWidth, scaledHeight)) {
            cv::resize(EnsureBgr(cropped), scaled, cv::Size(scaledWidth, scaledHeight), 0.0, 0.0, cv::INTER_NEAREST);
        }
    } else {
        cv::resize(EnsureBgr(cropped), scaled, cv::Size(scaledWidth, scaledHeight), 0.0, 0.0, interpolation);
        if (interpolation == cv::INTER_LINEAR || interpolation == cv::INTER_CUBIC) {
            scaled = AdjustBrightnessToSource(EnsureBgr(cropped), scaled);
        }
    }

    if (scaled.empty()) {
        return output;
    }
    const QRect dstRect(srcRect.x() * 2, srcRect.y() * 2, scaled.cols, scaled.rows);
    const QRect targetRect(0, 0, targetSize.width, targetSize.height);
    const QRect clipped = dstRect.intersected(targetRect);
    if (clipped.isEmpty()) {
        return output;
    }
    const cv::Rect srcClip(clipped.x() - dstRect.x(),
                           clipped.y() - dstRect.y(),
                           clipped.width(),
                           clipped.height());
    const cv::Rect dstClip(clipped.x(), clipped.y(), clipped.width(), clipped.height());
    scaled(srcClip).copyTo(output(dstClip));
    return output;
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
        opt.showDecorationSelected = true;
        QStyle* style = opt.widget ? opt.widget->style() : QApplication::style();

        const bool secondarySelected = index.data(kPreviewSecondarySelectedRole).toBool();
        const bool playbackActive = index.data(kPreviewPlaybackRole).toBool();
        const bool primarySelected = opt.state.testFlag(QStyle::State_Selected) && !secondarySelected;
        opt.state &= ~QStyle::State_Selected;
        style->drawPrimitive(QStyle::PE_PanelItemViewItem, &opt, painter, opt.widget);
        if (primarySelected) {
            painter->fillRect(opt.rect, opt.palette.highlight().color());
        } else if (secondarySelected) {
            painter->fillRect(opt.rect, QColor(140, 190, 255, 160));
        }
        if (playbackActive) {
            const QPen pen(QColor(255, 210, 0), 3);
            painter->setPen(pen);
            painter->setBrush(Qt::NoBrush);
            painter->drawRect(opt.rect.adjusted(2, 2, -2, -2));
        }
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
        const bool selected = (opt.state & QStyle::State_Selected);
        const bool used = index.data(kPreviewUsageRole).toBool();
        const QColor usedColor(170, 200, 255);
        painter->fillRect(opt.rect,
                          selected ? opt.palette.highlight()
                                   : used ? usedColor
                                          : opt.palette.base());

        const QFontMetrics metrics(opt.font);
        const int padding = 6;
        const int textHeight = metrics.height() + 4;
        const int gap = 2;
        QRect contentRect = opt.rect.adjusted(padding, padding, -padding, -padding);
        const QSize iconSize = index.data(kPreviewIconSizeRole).toSize().isValid()
            ? index.data(kPreviewIconSizeRole).toSize()
            : opt.decorationSize.isValid() ? opt.decorationSize : QSize(kPreviewIconWidth, kPreviewIconHeight);
        const int iconX = contentRect.left() + (contentRect.width() - iconSize.width()) / 2;
        QRect iconRect(iconX,
                       contentRect.top(),
                       iconSize.width(),
                       iconSize.height());
        QRect textRect(iconRect.left(),
                       iconRect.bottom() + gap,
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
        painter->setPen(selected ? opt.palette.highlightedText().color()
                                 : opt.palette.text().color());
        painter->drawText(textRect, Qt::AlignHCenter | Qt::AlignVCenter, text);

        painter->restore();
    }

    QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const override
    {
        const QFontMetrics metrics(option.font);
        const int padding = 6;
        const int textHeight = metrics.height() + 4;
        const int gap = 2;
        const QSize iconSize = index.data(kPreviewIconSizeRole).toSize().isValid()
            ? index.data(kPreviewIconSizeRole).toSize()
            : option.decorationSize.isValid() ? option.decorationSize : QSize(kPreviewIconWidth, kPreviewIconHeight);
        const int height = iconSize.height() + textHeight + padding * 2 + gap;
        const int width = iconSize.width() + padding * 2;
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

void ApplySpriteMaskConstraints(cv::Mat& map, const cv::Mat& spriteMask)
{
    if (map.empty() || spriteMask.empty()) {
        return;
    }
    cv::Mat maskScaled;
    if (spriteMask.size() != map.size()) {
        cv::resize(spriteMask, maskScaled, map.size(), 0.0, 0.0, cv::INTER_NEAREST);
    } else {
        maskScaled = spriteMask;
    }
    for (int y = 0; y < map.rows; ++y) {
        uint8_t* row = map.ptr<uint8_t>(y);
        const uint8_t* mrow = maskScaled.ptr<uint8_t>(y);
        for (int x = 0; x < map.cols; ++x) {
            if (mrow[x] == 255) {
                row[x] = 255;
            }
        }
    }
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
    const int gap = FrameGapForWidth(std::max(sd.cols, hd.cols));
    return BuildStackedFrames({sd, hd}, gapColor, gap);
}

class FlowLayout : public QLayout
{
public:
    explicit FlowLayout(QWidget* parent = nullptr, int margin = -1, int hSpacing = -1, int vSpacing = -1)
        : QLayout(parent),
          m_hSpace(hSpacing),
          m_vSpace(vSpacing)
    {
        setContentsMargins(margin, margin, margin, margin);
    }

    ~FlowLayout() override
    {
        QLayoutItem* item = nullptr;
        while ((item = takeAt(0)) != nullptr) {
            delete item;
        }
    }

    void addItem(QLayoutItem* item) override
    {
        m_itemList.append(item);
    }

    int count() const override
    {
        return m_itemList.size();
    }

    QLayoutItem* itemAt(int index) const override
    {
        return m_itemList.value(index);
    }

    QLayoutItem* takeAt(int index) override
    {
        if (index < 0 || index >= m_itemList.size()) {
            return nullptr;
        }
        return m_itemList.takeAt(index);
    }

    Qt::Orientations expandingDirections() const override
    {
        return {};
    }

    bool hasHeightForWidth() const override
    {
        return true;
    }

    int heightForWidth(int width) const override
    {
        return doLayout(QRect(0, 0, width, 0), true);
    }

    QSize minimumSize() const override
    {
        QSize size;
        for (const QLayoutItem* item : m_itemList) {
            size = size.expandedTo(item->minimumSize());
        }
        const QMargins margins = contentsMargins();
        size += QSize(margins.left() + margins.right(), margins.top() + margins.bottom());
        return size;
    }

    QSize sizeHint() const override
    {
        return minimumSize();
    }

    void setGeometry(const QRect& rect) override
    {
        QLayout::setGeometry(rect);
        doLayout(rect, false);
    }

private:
    int horizontalSpacing() const
    {
        if (m_hSpace >= 0) {
            return m_hSpace;
        }
        return smartSpacing(QStyle::PM_LayoutHorizontalSpacing);
    }

    int verticalSpacing() const
    {
        if (m_vSpace >= 0) {
            return m_vSpace;
        }
        return smartSpacing(QStyle::PM_LayoutVerticalSpacing);
    }

    int doLayout(const QRect& rect, bool testOnly) const
    {
        int x = rect.x();
        int y = rect.y();
        int lineHeight = 0;
        const int spaceX = horizontalSpacing();
        const int spaceY = verticalSpacing();
        for (QLayoutItem* item : m_itemList) {
            const QWidget* widget = item->widget();
            const int nextX = x + item->sizeHint().width() + spaceX;
            if (nextX - spaceX > rect.right() && lineHeight > 0) {
                x = rect.x();
                y += lineHeight + spaceY;
                lineHeight = 0;
            }
            if (!testOnly) {
                item->setGeometry(QRect(QPoint(x, y), item->sizeHint()));
            }
            x += item->sizeHint().width() + spaceX;
            lineHeight = std::max(lineHeight, item->sizeHint().height());
        }
        return y + lineHeight - rect.y();
    }

    int smartSpacing(QStyle::PixelMetric pm) const
    {
        const QObject* parentObject = parent();
        if (!parentObject) {
            return -1;
        }
        if (parentObject->isWidgetType()) {
            const QWidget* parentWidget = static_cast<const QWidget*>(parentObject);
            return parentWidget->style()->pixelMetric(pm, nullptr, parentWidget);
        }
        return static_cast<const QLayout*>(parentObject)->spacing();
    }

    QList<QLayoutItem*> m_itemList;
    int m_hSpace;
    int m_vSpace;
};

QSize PreviewItemSizeForIcon(const QSize& iconSize, const QFont& font)
{
    const QFontMetrics metrics(font);
    const int padding = 6;
    const int textHeight = metrics.height() + 4;
    const int height = iconSize.height() + textHeight + padding * 2;
    const int width = std::max(iconSize.width() + padding * 2, kPreviewItemWidth);
    return QSize(width, height);
}

constexpr const char* kPaletteColorMime = "application/x-ppuc-color";

QByteArray EncodePaletteColor(const QColor& color)
{
    QByteArray data;
    QDataStream stream(&data, QIODevice::WriteOnly);
    stream << static_cast<quint8>(color.red())
           << static_cast<quint8>(color.green())
           << static_cast<quint8>(color.blue());
    return data;
}

bool DecodePaletteColor(const QMimeData* mimeData, QColor& color)
{
    if (!mimeData || !mimeData->hasFormat(kPaletteColorMime)) {
        return false;
    }
    const QByteArray data = mimeData->data(kPaletteColorMime);
    QDataStream stream(data);
    quint8 r = 0;
    quint8 g = 0;
    quint8 b = 0;
    stream >> r >> g >> b;
    color = QColor(r, g, b);
    return true;
}

bool ExtractListDrop(const QMimeData* mimeData, QString& kind, int& index)
{
    if (!mimeData || !mimeData->hasFormat("application/x-qabstractitemmodeldatalist")) {
        return false;
    }
    const QByteArray encoded = mimeData->data("application/x-qabstractitemmodeldatalist");
    QDataStream stream(encoded);
    while (!stream.atEnd()) {
        int row = 0;
        int col = 0;
        QMap<int, QVariant> roleData;
        stream >> row >> col >> roleData;
        if (roleData.contains(Qt::UserRole + 1) && roleData.contains(Qt::UserRole)) {
            kind = roleData.value(Qt::UserRole + 1).toString();
            index = roleData.value(Qt::UserRole).toInt();
            if (!kind.isEmpty() && index >= 0) {
                return true;
            }
        }
    }
    return false;
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
    const QString appName = QCoreApplication::applicationName().isEmpty()
        ? QString("PPUC-Serum-Colorizer")
        : QCoreApplication::applicationName();
    const QString appVersion = QCoreApplication::applicationVersion();
    setWindowTitle(appVersion.isEmpty() ? appName : QString("%1 v%2").arg(appName, appVersion));
    setWindowIcon(QIcon(":/app/app.png"));
    setMinimumSize(1200, 800);

    auto* fileMenu = menuBar()->addMenu("&File");
    auto* editMenu = menuBar()->addMenu("&Edit");
    auto* viewMenu = menuBar()->addMenu("&View");
    auto* helpMenu = menuBar()->addMenu("&Help");

    auto* newAction = new QAction(QIcon(":/icons/new.png"), "&New", this);
    auto* openAction = new QAction(QIcon(":/icons/open.png"), "&Open...", this);
    auto* saveAction = new QAction(QIcon(":/icons/save.png"), "&Save", this);
    auto* saveAsAction = new QAction("Save &As...", this);
    saveAction->setShortcut(QKeySequence::Save);
    saveAction->setShortcutContext(Qt::ApplicationShortcut);
    saveAsAction->setShortcut(QKeySequence::SaveAs);
    saveAsAction->setShortcutContext(Qt::ApplicationShortcut);
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
    toolPointAction->setIcon(QIcon(":/icons/crayon.png"));
    auto* toolLineAction = new QAction("&Line", this);
    toolLineAction->setCheckable(true);
    toolLineAction->setIcon(QIcon(":/icons/trait.png"));
    auto* toolRectAction = new QAction("&Rectangle", this);
    toolRectAction->setCheckable(true);
    toolRectAction->setIcon(QIcon(":/icons/select.png"));
    auto* toolRectFillAction = new QAction("Rectangle &Fill", this);
    toolRectFillAction->setCheckable(true);
    toolRectFillAction->setIcon(QIcon(":/icons/drawall.png"));
    auto* toolCircleAction = new QAction("&Circle", this);
    toolCircleAction->setCheckable(true);
    toolCircleAction->setIcon(QIcon(":/icons/cercle.png"));
    auto* toolCircleFillAction = new QAction("Circle F&ill", this);
    toolCircleFillAction->setCheckable(true);
    toolCircleFillAction->setIcon(QIcon(":/icons/drawall.png"));
    auto* toolEllipseAction = new QAction("&Ellipse", this);
    toolEllipseAction->setCheckable(true);
    toolEllipseAction->setIcon(QIcon(":/icons/ellipse.png"));
    auto* toolEllipseFillAction = new QAction("Ellipse Fi&ll", this);
    toolEllipseFillAction->setCheckable(true);
    toolEllipseFillAction->setIcon(QIcon(":/icons/drawall.png"));
    auto* toolColorPickerAction = new QAction("Color &Picker", this);
    toolColorPickerAction->setCheckable(true);
    toolColorPickerAction->setIcon(QIcon(":/icons/colpick.png"));
    auto* toolMagicFillAction = new QAction("&Magic Fill", this);
    toolMagicFillAction->setCheckable(true);
    toolMagicFillAction->setIcon(QIcon(":/icons/drawall.png"));
    auto* cancelDrawAction = new QAction("&Cancel Draw", this);
    cancelDrawAction->setShortcut(QKeySequence(Qt::Key_Escape));
    cancelDrawAction->setShortcutContext(Qt::WidgetWithChildrenShortcut);
    m_settingsAction = new QAction("&Settings...", this);
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
    editMenu->addSeparator();
    editMenu->addAction(m_settingsAction);
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
            case UndoTarget::Palette:
                handled = undoPaletteEdit();
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
            case UndoTarget::Palette:
                handled = redoPaletteEdit();
                break;
        }
        if (handled) {
            statusBar()->showMessage("Redo", 1500);
        }
    });
    connect(newAction, &QAction::triggered, this, [this]() {
        stopPlayback();
        m_state->newProject();
        m_imageStore->clear();
        m_frameStore->clear();
        m_spriteStore->clear();
        m_frameDurations.clear();
        m_frameTriggerIds.clear();
        m_spriteNames.clear();
        m_spriteColored.clear();
        m_spriteColoredX.clear();
        m_spriteOriginals.clear();
        m_spriteMasksX.clear();
        m_spriteDynamicMasks.clear();
        m_spriteDynamicMasksX.clear();
        m_spriteDynamicColors.clear();
        m_spriteDynamicColorsX.clear();
        m_spriteExtraFlags.clear();
        m_spriteShapeModes.clear();
        m_spriteDetAreas.clear();
        m_spriteDetDwords.clear();
        m_spriteDetDwordPos.clear();
        m_frameSpriteAssignments.clear();
        m_frameSpriteBBoxes.clear();
        m_spriteColFromFrame.clear();
        m_spriteRects.clear();
        m_spriteRectMirror.clear();
        m_sectionStarts.clear();
        m_sectionNames.clear();
        m_frameRefs.clear();
        m_frameDynamicColors.clear();
        m_compMasks.clear();
        m_frameCompMaskIds.clear();
        m_frameDynamicMaskMaps.clear();
        m_frameDynamicMaskMapsX.clear();
        m_frameShapeCompModes.clear();
        m_noColors = 64;
        m_projectDir.clear();
        m_projectBaseName.clear();
        resetUndoStacks();
        resetNavigationHistory();
        updateMetadataForFrame(-1);
        updateMetadataForSprite(-1);
        populateBookmarks({}, {});
        m_hasLegacyRoundTrip = false;
        m_legacyRoundTrip = LegacyRoundTripData{};
        statusBar()->showMessage("New project (stub)", 3000);
    });
    connect(openAction, &QAction::triggered, this, [this]() {
        const QString filename = QFileDialog::getOpenFileName(
            this,
            "Open Project",
            QString(),
            "Serum Projects (*.crom *.cROM *.crp *.cRP *.cROMc);;All Files (*.*)");
        if (!filename.isEmpty()) {
            openProjectFile(filename);
        }
    });
    connect(saveAction, &QAction::triggered, this, [this]() {
        QString filename = m_state->projectPath();
        if (filename.isEmpty()) {
            filename = QFileDialog::getSaveFileName(this,
                                                    "Save Project",
                                                    QString(),
                                                    "Serum Projects (*.cRP *.cROMc)");
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
        const QString filename = QFileDialog::getSaveFileName(
            this,
            "Save Project As",
            QString(),
            "Serum Projects (*.cRP *.cROMc)");
        if (filename.isEmpty()) {
            return;
        }
        if (filename.isEmpty()) {
            return;
        }
        if (saveProjectToPath(filename)) {
            QFileInfo info(filename);
            QString suffix = info.suffix().toLower();
            QString base = info.completeBaseName();
            QString dir = info.absolutePath();
            if (suffix != "cromc" && suffix != "crp" && suffix != "crom") {
                suffix = "crp";
                base = info.fileName();
            }
            m_projectDir = dir;
            m_projectBaseName = base;
            m_serumRuntimeAltDir = dir;
            m_serumRuntimeRomName = base;
            updateWindowTitle();
            persistRecentFiles();
            statusBar()->showMessage(QString("Save As: %1").arg(filename), 5000);
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
            markProjectDirty();
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
        if (m_serumDataLoaded) {
            m_serumData.nframes = static_cast<uint32_t>(m_frameStore->count());
        }
        m_state->addFrame();
        markProjectDirty();
        m_frameTriggerIds.push_back(0xffffffffu);
        if (m_hasLegacyRoundTrip) {
            m_legacyRoundTrip.trigger_ids = m_frameTriggerIds;
        }
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
        markProjectDirty();
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
    connect(cancelDrawAction, &QAction::triggered, this, [this]() {
        if (m_framePreviewList &&
            (m_framePreviewList->hasFocus() || m_framePreviewList->viewport()->hasFocus())) {
            return;
        }
        cancelCurrentDraw();
    });
    connect(m_settingsAction, &QAction::triggered, this, [this]() { showSettingsDialog(); });
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
            if (m_serumDataLoaded) {
                m_serumData.nframes = static_cast<uint32_t>(m_frameStore->count());
            }
            m_state->removeFrame(row);
            markProjectDirty();
            if (row >= 0 && row < static_cast<int>(m_frameRefs.size())) {
                m_frameRefs.erase(m_frameRefs.begin() + row);
            }
            if (row >= 0 && row < static_cast<int>(m_frameDynamicColors.size())) {
                m_frameDynamicColors.erase(m_frameDynamicColors.begin() + row);
            }
            if (row >= 0 && row < static_cast<int>(m_frameCompMaskIds.size())) {
                m_frameCompMaskIds.erase(m_frameCompMaskIds.begin() + row);
            }
            if (row >= 0 && row < static_cast<int>(m_frameDynamicMaskMaps.size())) {
                m_frameDynamicMaskMaps.erase(m_frameDynamicMaskMaps.begin() + row);
            }
            if (row >= 0 && row < static_cast<int>(m_frameDynamicMaskMapsX.size())) {
                m_frameDynamicMaskMapsX.erase(m_frameDynamicMaskMapsX.begin() + row);
            }
            if (row >= 0 && row < static_cast<int>(m_frameTriggerIds.size())) {
                m_frameTriggerIds.erase(m_frameTriggerIds.begin() + row);
                if (m_hasLegacyRoundTrip) {
                    m_legacyRoundTrip.trigger_ids = m_frameTriggerIds;
                }
            }
            ensureUndoStacksSize();
        } else if (m_spritesList->hasFocus()) {
            const int row = m_spritesList->currentRow();
            m_spriteStore->removeAt(row);
            m_state->removeSprite(row);
            markProjectDirty();
            if (row >= 0 && row < static_cast<int>(m_spriteNames.size())) {
                m_spriteNames.erase(m_spriteNames.begin() + row);
            }
            if (row >= 0 && row < static_cast<int>(m_spriteColored.size())) {
                m_spriteColored.erase(m_spriteColored.begin() + row);
            }
            if (row >= 0 && row < static_cast<int>(m_spriteColoredX.size())) {
                m_spriteColoredX.erase(m_spriteColoredX.begin() + row);
            }
            if (row >= 0 && row < static_cast<int>(m_spriteOriginals.size())) {
                m_spriteOriginals.erase(m_spriteOriginals.begin() + row);
            }
            if (row >= 0 && row < static_cast<int>(m_spriteMasksX.size())) {
                m_spriteMasksX.erase(m_spriteMasksX.begin() + row);
            }
            if (row >= 0 && row < static_cast<int>(m_spriteDynamicMasks.size())) {
                m_spriteDynamicMasks.erase(m_spriteDynamicMasks.begin() + row);
            }
            if (row >= 0 && row < static_cast<int>(m_spriteDynamicMasksX.size())) {
                m_spriteDynamicMasksX.erase(m_spriteDynamicMasksX.begin() + row);
            }
            if (row >= 0 && row < static_cast<int>(m_spriteDynamicColors.size())) {
                m_spriteDynamicColors.erase(m_spriteDynamicColors.begin() + row);
            }
            if (row >= 0 && row < static_cast<int>(m_spriteDynamicColorsX.size())) {
                m_spriteDynamicColorsX.erase(m_spriteDynamicColorsX.begin() + row);
            }
            if (row >= 0 && row < static_cast<int>(m_spriteExtraFlags.size())) {
                m_spriteExtraFlags.erase(m_spriteExtraFlags.begin() + row);
            }
            if (row >= 0 && row < static_cast<int>(m_spriteShapeModes.size())) {
                m_spriteShapeModes.erase(m_spriteShapeModes.begin() + row);
            }
            if (!m_frameSpriteAssignments.empty()) {
                for (auto& entry : m_frameSpriteAssignments) {
                    if (entry == 255) {
                        continue;
                    }
                    if (entry == static_cast<uint8_t>(row)) {
                        entry = 255;
                    } else if (entry > static_cast<uint8_t>(row)) {
                        entry = static_cast<uint8_t>(entry - 1);
                    }
                }
            }
            ensureUndoStacksSize();
        } else if (m_imagesList->hasFocus()) {
            const int row = m_imagesList->currentRow();
            if (row >= 0 && row < m_state->images().size()) {
                const QString path = m_state->images().at(row);
                m_state->removeImage(row);
                markProjectDirty();
                m_imageStore->removeByPath(path);
            }
        }
        updateSelectionFromLists();
    });

    auto* tabs = new QTabWidget(this);
    m_canvasTabs = tabs;
    m_framesCanvas = new CanvasWidget("Frames canvas (placeholder)", tabs);
    m_spritesCanvas = new CanvasWidget("Sprites canvas (placeholder)", tabs);
    m_spritesCanvas->setMaskButtonText("Detect");
    m_spritesCanvas->setMaskButtonToolTip("Edit sprite detection areas");
    m_imagesCanvas = new CanvasWidget("Images canvas (placeholder)", tabs);
    m_backgroundsCanvas = new CanvasWidget("Backgrounds canvas (placeholder)", tabs);
    m_playbackCanvas = new CanvasWidget("Playback canvas (placeholder)", tabs);
    m_framesCanvas->addAction(cancelDrawAction);
    m_spritesCanvas->addAction(cancelDrawAction);
    m_backgroundsCanvas->addAction(cancelDrawAction);
    m_framesCanvas->setMaskButtonsVisible(true);
    m_framesCanvas->setZoneButtonVisible(true);
    m_spritesCanvas->setMaskButtonsVisible(true);
    m_spritesCanvas->setZoneButtonVisible(false);
    m_imagesCanvas->setMaskButtonsVisible(false);
    m_imagesCanvas->setZoneButtonVisible(false);
    m_backgroundsCanvas->setMaskButtonsVisible(false);
    m_backgroundsCanvas->setZoneButtonVisible(false);
    m_framesCanvas->setBackgroundMaskVisible(true);
    m_spritesCanvas->setBackgroundMaskVisible(false);
    m_imagesCanvas->setBackgroundMaskVisible(false);
    m_backgroundsCanvas->setBackgroundMaskVisible(false);
    m_framesCanvas->setBackgroundVisible(true);
    m_spritesCanvas->setBackgroundVisible(false);
    m_imagesCanvas->setBackgroundVisible(false);
    m_backgroundsCanvas->setBackgroundVisible(false);
    m_playbackCanvas->setMaskButtonsVisible(false);
    m_playbackCanvas->setZoneButtonVisible(false);
    m_playbackCanvas->setBackgroundVisible(false);
    m_playbackCanvas->setBackgroundMaskVisible(false);
    m_playbackCanvas->setOriginalVisible(true);
    m_playbackCanvas->setHdButtonEnabled(false);
    m_playbackCanvas->setRotateEnabled(false);
    m_backgroundsCanvas->setHdButtonEnabled(true);
    m_framesCanvas->setRotateEnabled(true);
    m_spritesCanvas->setRotateEnabled(false);
    m_imagesCanvas->setRotateEnabled(false);
    m_backgroundsCanvas->setRotateEnabled(false);
    m_playbackTab = new QWidget(tabs);
    auto* playbackLayout = new QVBoxLayout(m_playbackTab);
    playbackLayout->setContentsMargins(0, 0, 0, 0);
    playbackLayout->setSpacing(6);
    playbackLayout->addWidget(m_playbackCanvas, 1);
    auto* playbackControls = new QWidget(m_playbackTab);
    auto* playbackControlsLayout = new QHBoxLayout(playbackControls);
    playbackControlsLayout->setContentsMargins(0, 0, 0, 0);
    playbackControlsLayout->setSpacing(8);
    m_previewRewindButton = new QToolButton(playbackControls);
    m_previewRewindButton->setText("Rew");
    m_previewRewindButton->setToolTip("Jump back 10 frames while playing");
    m_previewPrevButton = new QToolButton(playbackControls);
    m_previewPrevButton->setText("Prev");
    m_previewPrevButton->setToolTip("Step to previous frame");
    m_previewPlayButton = new QToolButton(playbackControls);
    m_previewPlayButton->setText("Play");
    m_previewPlayButton->setToolTip("Play selected frames or full ROM");
    m_previewStopButton = new QToolButton(playbackControls);
    m_previewStopButton->setText("Stop");
    m_previewStopButton->setToolTip("Stop playback");
    m_previewPauseButton = new QToolButton(playbackControls);
    m_previewPauseButton->setText("Pause");
    m_previewPauseButton->setToolTip("Pause playback");
    m_previewNextButton = new QToolButton(playbackControls);
    m_previewNextButton->setText("Next");
    m_previewNextButton->setToolTip("Step to next frame");
    m_previewFastForwardButton = new QToolButton(playbackControls);
    m_previewFastForwardButton->setText("Fwd");
    m_previewFastForwardButton->setToolTip("Jump forward 10 frames while playing");
    m_playbackOverrideDurationCheck = new QCheckBox("Fixed", playbackControls);
    m_playbackOverrideDurationCheck->setToolTip("Override frame duration during playback");
    m_playbackOverrideDurationSpin = new QSpinBox(playbackControls);
    m_playbackOverrideDurationSpin->setRange(1, 10000);
    m_playbackOverrideDurationSpin->setValue(33);
    m_playbackOverrideDurationSpin->setSuffix(" ms");
    m_playbackOverrideDurationSpin->setEnabled(false);
    playbackControlsLayout->addStretch(1);
    playbackControlsLayout->addWidget(m_previewRewindButton);
    playbackControlsLayout->addWidget(m_previewPrevButton);
    playbackControlsLayout->addWidget(m_previewPlayButton);
    playbackControlsLayout->addWidget(m_previewPauseButton);
    playbackControlsLayout->addWidget(m_previewStopButton);
    playbackControlsLayout->addWidget(m_previewNextButton);
    playbackControlsLayout->addWidget(m_previewFastForwardButton);
    playbackControlsLayout->addSpacing(12);
    playbackControlsLayout->addWidget(m_playbackOverrideDurationCheck);
    playbackControlsLayout->addWidget(m_playbackOverrideDurationSpin);
    playbackControlsLayout->addStretch(1);
    playbackControls->setLayout(playbackControlsLayout);
    playbackLayout->addWidget(playbackControls, 0, Qt::AlignHCenter);
    m_playbackTab->setLayout(playbackLayout);
    tabs->addTab(m_framesCanvas, "Frames");
    tabs->addTab(m_spritesCanvas, "Sprites");
    tabs->addTab(m_imagesCanvas, "Images");
    tabs->addTab(m_backgroundsCanvas, "Backgrounds");
    tabs->addTab(m_playbackTab, "Playback");
    setCentralWidget(tabs);
    connect(tabs, &QTabWidget::currentChanged, this, [this](int) {
        updateUndoActions();
        updateHdControlsForContext();
        if (m_canvasTabs && m_canvasTabs->currentWidget() == m_framesCanvas) {
            const int previewRow = m_framePreviewList ? m_framePreviewList->currentRow() : -1;
            if (m_framePreviewList && previewRow >= 0 && previewRow < m_framePreviewList->count()) {
                const QListWidgetItem* item = m_framePreviewList->item(previewRow);
                if (item) {
                    const int frameIndex = item->data(kFrameIndexRole).toInt();
                    if (frameIndex >= 0 && frameIndex < m_framesList->count()) {
                        QSignalBlocker blocker(m_framesList);
                        m_framesList->setCurrentRow(frameIndex);
                        showFrameAtIndex(frameIndex);
                    }
                }
            } else if (m_framesList && m_framesList->currentRow() >= 0) {
                showFrameAtIndex(m_framesList->currentRow());
            }
        }
    });

    auto* toolDock = new QDockWidget("Components", this);
    toolDock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    auto* toolsContainer = new QWidget(toolDock);
    auto* toolsLayout = new QVBoxLayout(toolsContainer);
    toolsLayout->setContentsMargins(4, 4, 4, 4);
    toolsLayout->setSpacing(4);
    auto* toolsTabBar = new QWidget(toolsContainer);
    auto* toolsTabLayout = new FlowLayout(toolsTabBar, 0, 6, 6);
    toolsTabBar->setLayout(toolsTabLayout);
    m_toolsTabs = new QStackedWidget(toolsContainer);
    auto* toolsButtonGroup = new QButtonGroup(toolsContainer);
    toolsButtonGroup->setExclusive(true);
    auto addToolsTab = [this, toolsTabLayout, toolsButtonGroup](QWidget* page, const QString& label) {
        if (!page) {
            return -1;
        }
        const int index = m_toolsTabs->addWidget(page);
        auto* button = new QToolButton();
        button->setText(label);
        button->setCheckable(true);
        button->setAutoRaise(true);
        button->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
        toolsTabLayout->addWidget(button);
        toolsButtonGroup->addButton(button, index);
        connect(button, &QToolButton::clicked, this, [this, index]() {
            m_toolsTabs->setCurrentIndex(index);
        });
        if (m_toolsTabs->currentIndex() == index) {
            button->setChecked(true);
        }
        return index;
    };
    connect(m_toolsTabs, &QStackedWidget::currentChanged, this, [toolsButtonGroup](int index) {
        if (QAbstractButton* button = toolsButtonGroup->button(index)) {
            button->setChecked(true);
        }
    });
    toolsLayout->addWidget(toolsTabBar);
    toolsLayout->addWidget(m_toolsTabs, 1);
    toolsContainer->setLayout(toolsLayout);
    auto* framesTab = new QWidget(m_toolsTabs);
    auto* spritesTab = new QWidget(m_toolsTabs);
    auto* imagesTab = new QWidget(m_toolsTabs);
    auto* masksTab = new QWidget(m_toolsTabs);
    auto* dynamicMasksTab = new QWidget(m_toolsTabs);
    auto* backgroundsTab = new QWidget(m_toolsTabs);
    auto* spriteZonesTab = new QWidget();
    m_masksTab = masksTab;
    m_dynamicMasksTab = dynamicMasksTab;
    m_backgroundsTab = backgroundsTab;
    m_spritesTab = spritesTab;
    m_spriteZonesTab = spriteZonesTab;

    m_frameFilter = new QLineEdit(framesTab);
    m_frameFilter->setPlaceholderText("Filter frames...");
    m_framesList = new QListWidget(framesTab);
    m_framesList->setSelectionMode(QAbstractItemView::SingleSelection);
    m_framesList->setViewMode(QListView::IconMode);
    m_framesList->setFlow(QListView::TopToBottom);
    m_framesList->setWrapping(false);
    m_framesList->setMovement(QListView::Snap);
    m_framesList->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_framesList->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    m_framesList->setResizeMode(QListView::Adjust);
    m_framesList->setUniformItemSizes(false);
    m_framesList->setDragDropMode(QAbstractItemView::DragOnly);
    m_framesList->setDragDropOverwriteMode(false);
    m_framesList->setDragEnabled(true);
    m_framesList->setAcceptDrops(false);
    m_framesList->setDropIndicatorShown(false);
    m_framesList->setIconSize(QSize(kPreviewIconWidthHd, kPreviewIconHeightHd));
    m_framesList->setGridSize(QSize());
    m_framesList->setSpacing(4);
    m_framesList->setItemDelegate(new ToolPreviewDelegate(m_framesList));
    m_bookmarksCombo = new QComboBox(framesTab);
    m_bookmarksCombo->setEnabled(false);
    auto* framesLayout = new QVBoxLayout(framesTab);
    framesLayout->addWidget(m_frameFilter);
    framesLayout->addWidget(m_framesList, 1);
    auto* bookmarksRow = new QHBoxLayout();
    auto* bookmarksLabel = new QLabel("Bookmarks", framesTab);
    bookmarksRow->addWidget(bookmarksLabel);
    bookmarksRow->addWidget(m_bookmarksCombo, 1);
    framesLayout->addLayout(bookmarksRow);
    framesTab->setLayout(framesLayout);

    m_spriteFilter = new QLineEdit(spritesTab);
    m_spriteFilter->setPlaceholderText("Filter sprites...");
    m_spritesList = new QListWidget(spritesTab);
    m_spritesList->setSelectionMode(QAbstractItemView::SingleSelection);
    m_spritesList->setViewMode(QListView::IconMode);
    m_spritesList->setFlow(QListView::TopToBottom);
    m_spritesList->setWrapping(false);
    m_spritesList->setMovement(QListView::Snap);
    m_spritesList->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_spritesList->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    m_spritesList->setResizeMode(QListView::Adjust);
    m_spritesList->setUniformItemSizes(false);
    m_spritesList->setDragDropMode(QAbstractItemView::DragOnly);
    m_spritesList->setDragDropOverwriteMode(false);
    m_spritesList->setDragEnabled(true);
    m_spritesList->setAcceptDrops(false);
    m_spritesList->setDropIndicatorShown(false);
    m_spritesList->setIconSize(QSize(kPreviewIconWidthHd, kPreviewIconHeightHd));
    m_spritesList->setGridSize(QSize());
    m_spritesList->setSpacing(4);
    m_spritesList->setItemDelegate(new ToolPreviewDelegate(m_spritesList));
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

    m_spriteZoneList = new QListWidget(spriteZonesTab);
    m_spriteZoneList->setSelectionMode(QAbstractItemView::SingleSelection);
    m_spriteZoneList->setViewMode(QListView::IconMode);
    m_spriteZoneList->setFlow(QListView::TopToBottom);
    m_spriteZoneList->setWrapping(false);
    m_spriteZoneList->setMovement(QListView::Snap);
    m_spriteZoneList->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_spriteZoneList->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    m_spriteZoneList->setResizeMode(QListView::Adjust);
    m_spriteZoneList->setUniformItemSizes(false);
    m_spriteZoneList->setIconSize(QSize(kPreviewIconWidth, kPreviewIconHeight));
    m_spriteZoneList->setGridSize(QSize());
    m_spriteZoneList->setSpacing(4);
    m_spriteZoneList->setItemDelegate(new ToolPreviewDelegate(m_spriteZoneList));
    m_spriteZoneAddButton = new QToolButton(spriteZonesTab);
    m_spriteZoneAddButton->setText("Add Zone");
    m_spriteZoneRemoveButton = new QToolButton(spriteZonesTab);
    m_spriteZoneRemoveButton->setText("Remove Zone");
    auto* spriteZoneButtons = new QHBoxLayout();
    spriteZoneButtons->addWidget(m_spriteZoneAddButton);
    spriteZoneButtons->addWidget(m_spriteZoneRemoveButton);
    spriteZoneButtons->addStretch(1);
    auto* spriteZoneListLayout = new QVBoxLayout();
    spriteZoneListLayout->addWidget(m_spriteZoneList, 1);
    spriteZoneListLayout->addLayout(spriteZoneButtons);

    m_spriteZoneSpritesList = new QListWidget(spriteZonesTab);
    m_spriteZoneSpritesList->setSelectionMode(QAbstractItemView::SingleSelection);
    m_spriteZoneSpritesList->setViewMode(QListView::IconMode);
    m_spriteZoneSpritesList->setFlow(QListView::TopToBottom);
    m_spriteZoneSpritesList->setWrapping(false);
    m_spriteZoneSpritesList->setMovement(QListView::Snap);
    m_spriteZoneSpritesList->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_spriteZoneSpritesList->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    m_spriteZoneSpritesList->setResizeMode(QListView::Adjust);
    m_spriteZoneSpritesList->setUniformItemSizes(false);
    m_spriteZoneSpritesList->setAcceptDrops(true);
    m_spriteZoneSpritesList->setDropIndicatorShown(true);
    m_spriteZoneSpritesList->setDragDropMode(QAbstractItemView::DropOnly);
    m_spriteZoneSpritesList->setIconSize(QSize(kPreviewIconWidth, kPreviewIconHeight));
    m_spriteZoneSpritesList->setGridSize(QSize());
    m_spriteZoneSpritesList->setSpacing(4);
    m_spriteZoneSpritesList->setItemDelegate(new ToolPreviewDelegate(m_spriteZoneSpritesList));
    m_spriteZoneSpriteUp = new QToolButton(spriteZonesTab);
    m_spriteZoneSpriteUp->setText("Up");
    m_spriteZoneSpriteDown = new QToolButton(spriteZonesTab);
    m_spriteZoneSpriteDown->setText("Down");
    m_spriteZoneSpriteRemove = new QToolButton(spriteZonesTab);
    m_spriteZoneSpriteRemove->setText("Remove");
    auto* spriteZoneSpriteButtons = new QHBoxLayout();
    spriteZoneSpriteButtons->addWidget(m_spriteZoneSpriteUp);
    spriteZoneSpriteButtons->addWidget(m_spriteZoneSpriteDown);
    spriteZoneSpriteButtons->addWidget(m_spriteZoneSpriteRemove);
    spriteZoneSpriteButtons->addStretch(1);
    auto* spriteZoneSpritesLayout = new QVBoxLayout();
    spriteZoneSpritesLayout->addWidget(m_spriteZoneSpritesList, 1);
    spriteZoneSpritesLayout->addLayout(spriteZoneSpriteButtons);

    auto* spriteZonesLayout = new QHBoxLayout(spriteZonesTab);
    spriteZonesLayout->addLayout(spriteZoneListLayout, 1);
    spriteZonesLayout->addLayout(spriteZoneSpritesLayout, 1);
    spriteZonesTab->setLayout(spriteZonesLayout);

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
    m_maskList->setGridSize(QSize());
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
    m_dynamicMaskList->setIconSize(QSize(kDynamicMaskIconWidth, kPreviewIconHeight));
    m_dynamicMaskList->setGridSize(QSize());
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

    auto* colorsTab = new QWidget(m_toolsTabs);
    m_colorsTab = colorsTab;
    m_currentColorButton = new QToolButton(colorsTab);
    m_currentColorButton->setAutoRaise(true);
    m_currentColorButton->setFixedSize(32, 32);
    m_currentColorButton->setToolTip("Current color");
    m_currentColorButton->installEventFilter(this);
    m_colorInfoLabel = new QLabel("RGB565: 0xFFFF\nRGB: 255,255,255", colorsTab);
    m_colorInfoLabel->setFixedWidth(160);
    m_paletteSetCombo = new QComboBox(colorsTab);
    m_colorPickButton = new QPushButton("Pick Color...", colorsTab);
    m_paletteAssignButton = new QPushButton("Set Slot", colorsTab);
    m_paletteGradientButton = new QPushButton("Gradient", colorsTab);
    m_paletteGradientButton->setEnabled(false);
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
    m_paletteList->setAcceptDrops(true);
    m_paletteList->viewport()->installEventFilter(this);
    m_paletteList->viewport()->setAcceptDrops(true);

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
    m_reducedPaletteList->setAcceptDrops(true);
    m_reducedPaletteList->viewport()->installEventFilter(this);
    m_reducedPaletteList->viewport()->setAcceptDrops(true);
    const int reducedGridWidth = paletteCellSize * 8 + m_reducedPaletteList->frameWidth() * 2 + 4;
    const int reducedGridHeight = paletteCellSize * 2 + m_reducedPaletteList->frameWidth() * 2 + 4;
    m_reducedPaletteList->setFixedSize(reducedGridWidth, reducedGridHeight);
    m_reducedPaletteList->setProperty("selectionColor", QColor(255, 140, 0));
    connect(m_reducedPaletteList, &QListWidget::currentRowChanged, this, [this](int row) {
        if (row < 0) {
            return;
        }
        if (m_paletteSetSlotActive) {
            cancelPaletteSetSlot();
            return;
        }
        if (m_dynamicSetSlotActive || m_rotationSetSlotActive) {
            cancelPaletteSetSlot();
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
        if (m_paletteGradientActive) {
            cancelPaletteGradient();
        }
        if (m_reducedSetSlotActive) {
            const QColor current(static_cast<int>(m_drawColor[2]),
                                 static_cast<int>(m_drawColor[1]),
                                 static_cast<int>(m_drawColor[0]));
            pushReducedUndoSnapshot(m_reducedPaletteIndex);
            setReducedSlotColor(m_reducedPaletteIndex, row, current);
            refreshReducedPaletteButtons();
            cancelPaletteSetSlot();
            return;
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
    reducedTop->addWidget(new QLabel("Replacement Set", colorsTab));
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
    m_dynamicPaletteList->setAcceptDrops(true);
    m_dynamicPaletteList->viewport()->installEventFilter(this);
    m_dynamicPaletteList->viewport()->setAcceptDrops(true);
    const int dynamicGridWidth = paletteCellSize * 8 + m_dynamicPaletteList->frameWidth() * 2 + 4;
    const int dynamicGridHeight = paletteCellSize * 2 + m_dynamicPaletteList->frameWidth() * 2 + 4;
    m_dynamicPaletteList->setFixedSize(dynamicGridWidth, dynamicGridHeight);
    m_dynamicPaletteList->setProperty("selectionColor", QColor(255, 140, 0));
    connect(m_dynamicPaletteList, &QListWidget::currentRowChanged, this, [this](int row) {
        if (row < 0) {
            return;
        }
        if (m_paletteSetSlotActive) {
            cancelPaletteSetSlot();
            return;
        }
        if (m_reducedSetSlotActive || m_rotationSetSlotActive) {
            cancelPaletteSetSlot();
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
        if (m_paletteGradientActive) {
            cancelPaletteGradient();
        }
        if (m_dynamicSetSlotActive) {
            const std::vector<int> targets = targetFrameIndices();
            const QColor current(static_cast<int>(m_drawColor[2]),
                                 static_cast<int>(m_drawColor[1]),
                                 static_cast<int>(m_drawColor[0]));
            for (int frameIndex : targets) {
                if (frameIndex < 0 || frameIndex >= static_cast<int>(m_frameDynamicColors.size())) {
                    continue;
                }
                pushDynamicUndoSnapshot(frameIndex, m_dynamicSetIndex);
                setDynamicSlotColor(frameIndex, m_dynamicSetIndex, row, current);
            }
            refreshDynamicPaletteButtons();
            cancelPaletteSetSlot();
            return;
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
    dynamicTop->addWidget(new QLabel("Dynamic Set", colorsTab));
    dynamicTop->addWidget(m_dynamicSetCombo);
    dynamicTop->addStretch(1);
    dynamicTop->addWidget(m_dynamicAssignButton);
    dynamicLayout->addLayout(dynamicTop);
    dynamicLayout->addWidget(m_dynamicPaletteList);

    m_rotationSetCombo = new QComboBox(colorsTab);
    for (int i = 0; i < MAX_COLOR_ROTATIONN; ++i) {
        m_rotationSetCombo->addItem(QString("Rotation %1").arg(i + 1), i);
    }
    m_rotationDelaySpin = new QSpinBox(colorsTab);
    m_rotationDelaySpin->setRange(0, 60000);
    m_rotationDelaySpin->setSuffix(" ms");
    m_rotationList = new QListWidget(colorsTab);
    m_rotationList->setViewMode(QListView::IconMode);
    m_rotationList->setFlow(QListView::LeftToRight);
    m_rotationList->setWrapping(true);
    m_rotationList->setMovement(QListView::Static);
    m_rotationList->setSelectionMode(QAbstractItemView::SingleSelection);
    m_rotationList->setResizeMode(QListView::Fixed);
    m_rotationList->setIconSize(QSize(kPaletteSwatchSize, kPaletteSwatchSize));
    m_rotationList->setGridSize(QSize(paletteCellSize, paletteCellSize));
    m_rotationList->setSpacing(0);
    m_rotationList->setUniformItemSizes(true);
    m_rotationList->setItemDelegate(new PaletteSwatchDelegate(m_rotationList));
    m_rotationList->setFrameShape(QFrame::NoFrame);
    m_rotationList->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_rotationList->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_rotationList->setAcceptDrops(true);
    m_rotationList->viewport()->installEventFilter(this);
    m_rotationList->viewport()->setAcceptDrops(true);
    m_rotationList->setProperty("selectionColor", QColor(255, 140, 0));
    m_rotationAddButton = new QPushButton("Add Color", colorsTab);
    m_rotationRemoveButton = new QPushButton("Remove", colorsTab);
    m_rotationUpButton = new QPushButton("Up", colorsTab);
    m_rotationDownButton = new QPushButton("Down", colorsTab);
    m_rotationClearButton = new QPushButton("Clear", colorsTab);
    m_rotationAssignButton = new QPushButton("Set Slot", colorsTab);
    connect(m_rotationSetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        m_rotationSetIndex = std::max(0, index);
        refreshRotationList();
    });
    connect(m_rotationDelaySpin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        updateRotationDelay(value);
    });
    connect(m_rotationList, &QListWidget::currentRowChanged, this, [this](int row) {
        const bool hasSelection = row >= 0;
        m_rotationRemoveButton->setEnabled(hasSelection);
        m_rotationUpButton->setEnabled(hasSelection && row > 0);
        m_rotationDownButton->setEnabled(hasSelection && row + 1 < m_rotationList->count());
        if (!hasSelection) {
            return;
        }
        if (m_paletteSetSlotActive || m_reducedSetSlotActive || m_dynamicSetSlotActive) {
            cancelPaletteSetSlot();
            return;
        }
        if (m_rotationSetSlotActive) {
            const int frameIndex = m_framesList ? m_framesList->currentRow() : -1;
            if (frameIndex < 0) {
                return;
            }
            const bool useHd = m_useHdFrame && hasHdFrame(frameIndex);
            pushRotationUndoSnapshot(frameIndex, m_rotationSetIndex, useHd);
            uint16_t* rotations = rotationBlockForEdit(frameIndex, useHd);
            const std::size_t base = rotationSetOffset(m_rotationSetIndex);
            if (!rotations || base + MAX_LENGTH_COLOR_ROTATION > rotationBlockSize()) {
                cancelPaletteSetSlot();
                return;
            }
            const uint16_t length = rotations[base];
            if (row >= static_cast<int>(length)) {
                cancelPaletteSetSlot();
                return;
            }
            const cv::Vec3b bgr(static_cast<uint8_t>(m_drawColor[0]),
                                static_cast<uint8_t>(m_drawColor[1]),
                                static_cast<uint8_t>(m_drawColor[2]));
            rotations[base + 2 + static_cast<std::size_t>(row)] = BgrToRgb565(bgr);
            refreshRotationList();
            if (m_rotationList) {
                QSignalBlocker blocker(m_rotationList);
                if (row >= 0 && row < m_rotationList->count()) {
                    m_rotationList->setCurrentRow(row);
                }
            }
            resetCanvasRotationState();
            cancelPaletteSetSlot();
            return;
        }
        const QListWidgetItem* item = m_rotationList->item(row);
        if (!item) {
            return;
        }
        const int value = item->data(Qt::UserRole).toInt();
        const cv::Vec3b bgr = Rgb565ToBgr(static_cast<uint16_t>(value));
        const QColor color(bgr[2], bgr[1], bgr[0]);
        setDrawColor(color, true);
    });
    connect(m_rotationAddButton, &QPushButton::clicked, this, [this]() {
        updateRotationDataFromList();
        const int frameIndex = m_framesList ? m_framesList->currentRow() : -1;
        if (frameIndex < 0) {
            return;
        }
        const bool useHd = m_useHdFrame && hasHdFrame(frameIndex);
        pushRotationUndoSnapshot(frameIndex, m_rotationSetIndex, useHd);
        uint16_t* rotations = rotationBlockForEdit(frameIndex, useHd);
        const std::size_t base = rotationSetOffset(m_rotationSetIndex);
        if (!rotations || base + MAX_LENGTH_COLOR_ROTATION > rotationBlockSize()) {
            return;
        }
        uint16_t length = rotations[base];
        if (length >= MAX_LENGTH_COLOR_ROTATION - 2) {
            return;
        }
        const cv::Vec3b bgr(static_cast<uint8_t>(m_drawColor[0]),
                            static_cast<uint8_t>(m_drawColor[1]),
                            static_cast<uint8_t>(m_drawColor[2]));
        const uint16_t value = BgrToRgb565(bgr);
        rotations[base + 2 + length] = value;
        rotations[base] = static_cast<uint16_t>(length + 1);
        refreshRotationList();
        resetCanvasRotationState();
    });
    connect(m_rotationRemoveButton, &QPushButton::clicked, this, [this]() {
        const int index = m_rotationList ? m_rotationList->currentRow() : -1;
        if (index < 0) {
            return;
        }
        const int frameIndex = m_framesList ? m_framesList->currentRow() : -1;
        if (frameIndex < 0) {
            return;
        }
        const bool useHd = m_useHdFrame && hasHdFrame(frameIndex);
        pushRotationUndoSnapshot(frameIndex, m_rotationSetIndex, useHd);
        uint16_t* rotations = rotationBlockForEdit(frameIndex, useHd);
        const std::size_t base = rotationSetOffset(m_rotationSetIndex);
        if (!rotations || base + MAX_LENGTH_COLOR_ROTATION > rotationBlockSize()) {
            return;
        }
        uint16_t length = rotations[base];
        if (index >= length) {
            return;
        }
        for (uint16_t i = static_cast<uint16_t>(index); i + 1 < length; ++i) {
            rotations[base + 2 + i] = rotations[base + 2 + i + 1];
        }
        rotations[base + 2 + length - 1] = 0;
        rotations[base] = static_cast<uint16_t>(length - 1);
        refreshRotationList();
        resetCanvasRotationState();
    });
    connect(m_rotationUpButton, &QPushButton::clicked, this, [this]() {
        const int index = m_rotationList ? m_rotationList->currentRow() : -1;
        if (index <= 0) {
            return;
        }
        const int frameIndex = m_framesList ? m_framesList->currentRow() : -1;
        if (frameIndex < 0) {
            return;
        }
        const bool useHd = m_useHdFrame && hasHdFrame(frameIndex);
        pushRotationUndoSnapshot(frameIndex, m_rotationSetIndex, useHd);
        uint16_t* rotations = rotationBlockForEdit(frameIndex, useHd);
        const std::size_t base = rotationSetOffset(m_rotationSetIndex);
        if (!rotations || base + MAX_LENGTH_COLOR_ROTATION > rotationBlockSize()) {
            return;
        }
        const uint16_t length = rotations[base];
        if (index >= length) {
            return;
        }
        std::swap(rotations[base + 2 + index], rotations[base + 2 + index - 1]);
        refreshRotationList();
        m_rotationList->setCurrentRow(index - 1);
        resetCanvasRotationState();
    });
    connect(m_rotationDownButton, &QPushButton::clicked, this, [this]() {
        const int index = m_rotationList ? m_rotationList->currentRow() : -1;
        if (index < 0) {
            return;
        }
        const int frameIndex = m_framesList ? m_framesList->currentRow() : -1;
        if (frameIndex < 0) {
            return;
        }
        const bool useHd = m_useHdFrame && hasHdFrame(frameIndex);
        pushRotationUndoSnapshot(frameIndex, m_rotationSetIndex, useHd);
        uint16_t* rotations = rotationBlockForEdit(frameIndex, useHd);
        const std::size_t base = rotationSetOffset(m_rotationSetIndex);
        if (!rotations || base + MAX_LENGTH_COLOR_ROTATION > rotationBlockSize()) {
            return;
        }
        const uint16_t length = rotations[base];
        if (index + 1 >= length) {
            return;
        }
        std::swap(rotations[base + 2 + index], rotations[base + 2 + index + 1]);
        refreshRotationList();
        m_rotationList->setCurrentRow(index + 1);
        resetCanvasRotationState();
    });
    connect(m_rotationClearButton, &QPushButton::clicked, this, [this]() {
        const int frameIndex = m_framesList ? m_framesList->currentRow() : -1;
        if (frameIndex < 0) {
            return;
        }
        const bool useHd = m_useHdFrame && hasHdFrame(frameIndex);
        pushRotationUndoSnapshot(frameIndex, m_rotationSetIndex, useHd);
        uint16_t* rotations = rotationBlockForEdit(frameIndex, useHd);
        const std::size_t base = rotationSetOffset(m_rotationSetIndex);
        if (!rotations || base + MAX_LENGTH_COLOR_ROTATION > rotationBlockSize()) {
            return;
        }
        rotations[base] = 0;
        rotations[base + 1] = 0;
        for (int i = 0; i < MAX_LENGTH_COLOR_ROTATION - 2; ++i) {
            rotations[base + 2 + i] = 0;
        }
        refreshRotationList();
        resetCanvasRotationState();
    });
    connect(m_rotationAssignButton, &QPushButton::clicked, this, [this]() {
        startRotationSetSlot();
    });

    auto* rotationLayout = new QVBoxLayout();
    auto* rotationTop = new QHBoxLayout();
    rotationTop->addWidget(new QLabel("Rotations", colorsTab));
    rotationTop->addWidget(m_rotationSetCombo);
    rotationTop->addWidget(new QLabel("Delay", colorsTab));
    rotationTop->addWidget(m_rotationDelaySpin);
    rotationTop->addStretch(1);
    rotationTop->addWidget(m_rotationAssignButton);
    rotationLayout->addLayout(rotationTop);
    rotationLayout->addWidget(m_rotationList);
    auto* rotationButtons = new QHBoxLayout();
    rotationButtons->addWidget(m_rotationAddButton);
    rotationButtons->addWidget(m_rotationRemoveButton);
    rotationButtons->addWidget(m_rotationUpButton);
    rotationButtons->addWidget(m_rotationDownButton);
    rotationButtons->addStretch(1);
    rotationButtons->addWidget(m_rotationClearButton);
    rotationLayout->addLayout(rotationButtons);

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
    fullTop->addWidget(m_paletteGradientButton);
    fullTop->addWidget(m_paletteAssignButton);
    colorLayout->addLayout(fullTop);
    colorLayout->addWidget(m_paletteList);
    colorsContent->setLayout(colorLayout);
    colorsScrollArea->setWidget(colorsContent);
    auto* colorsTabLayout = new QVBoxLayout(colorsTab);
    colorsTabLayout->addWidget(colorsScrollArea);
    colorsTab->setLayout(colorsTabLayout);

    addToolsTab(framesTab, "Frames");
    addToolsTab(spritesTab, "Sprites");
    addToolsTab(imagesTab, "Images");
    addToolsTab(masksTab, "Masks");
    addToolsTab(dynamicMasksTab, "Dynamic Masks");
    addToolsTab(backgroundsTab, "Backgrounds");
    addToolsTab(colorsTab, "Colors");
    toolDock->setWidget(toolsContainer);
    addDockWidget(Qt::LeftDockWidgetArea, toolDock);

    auto* inspectorDock = new QDockWidget("Tools", this);
    inspectorDock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    auto* inspectorContainer = new QWidget(inspectorDock);
    auto* inspectorContainerLayout = new QVBoxLayout(inspectorContainer);
    inspectorContainerLayout->setContentsMargins(4, 4, 4, 4);
    inspectorContainerLayout->setSpacing(4);
    auto* inspectorTabBar = new QWidget(inspectorContainer);
    auto* inspectorTabLayout = new FlowLayout(inspectorTabBar, 0, 6, 6);
    inspectorTabBar->setLayout(inspectorTabLayout);
    auto* inspectorTabs = new QStackedWidget(inspectorContainer);
    auto* inspectorButtonGroup = new QButtonGroup(inspectorContainer);
    inspectorButtonGroup->setExclusive(true);
    auto addInspectorTab = [inspectorTabs, inspectorTabLayout, inspectorButtonGroup](QWidget* page,
                                                                                    const QString& label) {
        if (!page) {
            return -1;
        }
        const int index = inspectorTabs->addWidget(page);
        auto* button = new QToolButton();
        button->setText(label);
        button->setCheckable(true);
        button->setAutoRaise(true);
        button->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
        inspectorTabLayout->addWidget(button);
        inspectorButtonGroup->addButton(button, index);
        QObject::connect(button, &QToolButton::clicked, inspectorTabs, [inspectorTabs, index]() {
            inspectorTabs->setCurrentIndex(index);
        });
        if (inspectorTabs->currentIndex() == index) {
            button->setChecked(true);
        }
        return index;
    };
    connect(inspectorTabs, &QStackedWidget::currentChanged, this, [inspectorButtonGroup](int index) {
        if (QAbstractButton* button = inspectorButtonGroup->button(index)) {
            button->setChecked(true);
        }
    });
    inspectorContainerLayout->addWidget(inspectorTabBar);
    inspectorContainerLayout->addWidget(inspectorTabs, 1);
    inspectorContainer->setLayout(inspectorContainerLayout);
    auto* inspectorWidget = new QWidget(inspectorTabs);
    auto* inspectorLayout = new QFormLayout(inspectorWidget);
    m_projectLabel = new QLabel("None", inspectorWidget);
    m_projectLabel->setWordWrap(true);
    m_projectLabel->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    m_projectLabel->installEventFilter(this);
    m_frameJump = new QSpinBox(inspectorWidget);
    m_frameJump->setEnabled(false);
    m_frameJump->setPrefix("Frame ");
    m_countsLabel = new QLabel("Frames: 0, Sprites: 0", inspectorWidget);
    m_selectionLabel = new QLabel("None", inspectorWidget);
    m_frameMetaLabel = new QLabel("-", inspectorWidget);
    m_spriteMetaLabel = new QLabel("-", inspectorWidget);
    m_spriteDynamicSetCombo = new QComboBox(inspectorWidget);
    m_spriteDynamicSetCombo->setEnabled(false);
    for (int i = 0; i < MAX_DYNA_SETS_PER_SPRITE; ++i) {
        m_spriteDynamicSetCombo->addItem(QString("Set %1").arg(i + 1), i);
    }
    m_spriteDetAreaCombo = new QComboBox(inspectorWidget);
    m_spriteDetAreaCombo->setEnabled(false);
    for (int i = 0; i < MAX_SPRITE_DETECT_AREAS; ++i) {
        m_spriteDetAreaCombo->addItem(QString("Area %1").arg(i + 1), i);
    }
    m_spriteDetAreaClearButton = new QPushButton("Clear area", inspectorWidget);
    m_spriteDetAreaClearButton->setEnabled(false);
    m_frameMaskAssign = new QComboBox(inspectorWidget);
    m_frameDynamicMaskAssign = new QComboBox(inspectorWidget);
    m_frameDynamicCopyButton = new QPushButton("Copy to frame...", inspectorWidget);
    m_frameBackgroundAssign = new QComboBox(inspectorWidget);
    m_backgroundAssignLabel = new QLabel("Background", inspectorWidget);
    m_shapeCompToggle = new QCheckBox("Shape comparison", inspectorWidget);
    m_triggerIdSpin = new QSpinBox(inspectorWidget);
    m_triggerIdSpin->setMinimum(-1);
    m_triggerIdSpin->setMaximum(kMonochromeTriggerId);
    m_triggerIdSpin->setSpecialValueText("None");
    m_triggerIdSpin->setKeyboardTracking(false);
    m_triggerIdSpin->setEnabled(false);
    m_triggerMonochromeCheck = new QCheckBox("Switch to Monochrome", inspectorWidget);
    m_triggerMonochromeCheck->setEnabled(false);
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
    inspectorLayout->addRow("Go to frame", m_frameJump);
    inspectorLayout->addRow("Counts", m_countsLabel);
    inspectorLayout->addRow("Selection", m_selectionLabel);
    inspectorLayout->addRow("Frame info", m_frameMetaLabel);
    inspectorLayout->addRow("Trigger ID", m_triggerIdSpin);
    inspectorLayout->addRow(m_triggerMonochromeCheck);
    inspectorLayout->addRow("Mask", m_frameMaskAssign);
    inspectorLayout->addRow("Dynamic mask", m_frameDynamicMaskAssign);
    inspectorLayout->addRow("Dynamic copy", m_frameDynamicCopyButton);
    inspectorLayout->addRow(m_backgroundAssignLabel, m_frameBackgroundAssign);
    inspectorLayout->addRow("Shape compare", m_shapeCompToggle);
    inspectorWidget->setLayout(inspectorLayout);
    auto* colorSetsWidget = new QWidget(inspectorTabs);
    auto* colorSetsLayout = new QVBoxLayout(colorSetsWidget);
    colorSetsLayout->addLayout(reducedLayout);
    colorSetsLayout->addSpacing(8);
    colorSetsLayout->addLayout(dynamicLayout);
    colorSetsLayout->addSpacing(8);
    colorSetsLayout->addLayout(rotationLayout);
    colorSetsLayout->addStretch(1);
    colorSetsWidget->setLayout(colorSetsLayout);

    auto* hdWidget = new QWidget(inspectorTabs);
    auto* hdLayout = new QFormLayout(hdWidget);
    hdLayout->addRow("HD source", m_hdSourceCombo);
    hdLayout->addRow("HD scale", m_hdScaleCombo);
    hdLayout->addRow("HD create", m_hdCreateButton);
    hdLayout->addRow("HD delete", m_hdDeleteButton);
    hdWidget->setLayout(hdLayout);
    auto* spriteInspectorWidget = new QWidget(inspectorTabs);
    auto* spriteInspectorLayout = new QFormLayout(spriteInspectorWidget);
    spriteInspectorLayout->addRow("Sprite info", m_spriteMetaLabel);
    spriteInspectorLayout->addRow("Dynamic Set", m_spriteDynamicSetCombo);
    spriteInspectorLayout->addRow("Detect area", m_spriteDetAreaCombo);
    spriteInspectorLayout->addRow("Detect clear", m_spriteDetAreaClearButton);
    spriteInspectorWidget->setLayout(spriteInspectorLayout);
    addInspectorTab(inspectorWidget, "Inspector");
    addInspectorTab(spriteInspectorWidget, "Sprite");
    addInspectorTab(colorSetsWidget, "Color Sets");
    spriteZonesTab->setParent(inspectorTabs);
    addInspectorTab(spriteZonesTab, "Sprite Zones");
    addInspectorTab(hdWidget, "HD");
    inspectorDock->setWidget(inspectorContainer);
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
    m_previewSelectedButton = new QToolButton(previewWidget);
    m_previewSelectedButton->setText("Selected");
    m_previewSelectedButton->setCheckable(true);
    m_previewSelectedButton->setToolTip("Show only selected frames");
    previewBar->addWidget(m_previewSelectedButton);
    m_previewHdButton = new QToolButton(previewWidget);
    m_previewHdButton->setText("HD");
    m_previewHdButton->setCheckable(true);
    m_previewHdButton->setToolTip("Show only frames with HD data");
    previewBar->addWidget(m_previewHdButton);
    m_previewMaskOverlayButton = new QToolButton(previewWidget);
    m_previewMaskOverlayButton->setText("Masks");
    m_previewMaskOverlayButton->setCheckable(true);
    m_previewMaskOverlayButton->setToolTip("Overlay masks on preview originals");
    m_previewMaskOverlayButton->setChecked(true);
    m_previewMaskOverlayEnabled = true;
    previewBar->addWidget(m_previewMaskOverlayButton);
    m_previewRotateButton = new QToolButton(previewWidget);
    m_previewRotateButton->setText("Rotate");
    m_previewRotateButton->setCheckable(true);
    m_previewRotateButton->setToolTip("Preview color rotations (requires rotation data)");
    previewBar->addWidget(m_previewRotateButton);
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
    m_framePreviewList->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_framePreviewList->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    m_framePreviewList->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_framePreviewList->setResizeMode(QListView::Adjust);
    m_framePreviewList->setUniformItemSizes(false);
    m_framePreviewList->setIconSize(QSize(kPreviewIconWidthHd, kPreviewIconHeightHd));
    m_framePreviewList->setGridSize(QSize());
    m_framePreviewList->setSpacing(1);
    m_framePreviewList->setItemDelegate(new FramePreviewDelegate(m_framePreviewList));
    m_framePreviewList->setAcceptDrops(true);
    m_framePreviewList->setDropIndicatorShown(true);
    m_framePreviewList->setDragDropMode(QAbstractItemView::DropOnly);
    m_framePreviewList->viewport()->installEventFilter(this);
    m_framePreviewList->installEventFilter(this);
    previewLayout->addWidget(m_framePreviewList);

    previewWidget->setLayout(previewLayout);
    previewDock->setWidget(previewWidget);
    addDockWidget(Qt::BottomDockWidgetArea, previewDock);

    m_coordLabel = new QLabel(this);
    m_coordLabel->setMinimumWidth(140);
    statusBar()->addPermanentWidget(m_coordLabel);

    if (m_spriteZoneSpritesList) {
        m_spriteZoneSpritesList->viewport()->installEventFilter(this);
        m_spriteZoneSpritesList->installEventFilter(this);
    }

    viewMenu->addAction(toolDock->toggleViewAction());
    viewMenu->addAction(inspectorDock->toggleViewAction());
    viewMenu->addAction(previewDock->toggleViewAction());

    auto* handbookAction = new QAction("&Handbook", this);
    auto* licensesAction = new QAction("&Dependencies && Licenses", this);
    auto* copyLogAction = new QAction("&Copy Log", this);
    auto* debugCheckAction = new QAction("&Run Debug Check", this);
    auto* aboutAction = new QAction("&About", this);
    helpMenu->addAction(handbookAction);
    helpMenu->addAction(licensesAction);
    helpMenu->addAction(copyLogAction);
    helpMenu->addAction(debugCheckAction);
    helpMenu->addAction(aboutAction);
    connect(handbookAction, &QAction::triggered, this, [this]() {
        QFile file(":/docs/handbook.md");
        if (!file.open(QIODevice::ReadOnly)) {
            QMessageBox::warning(this, "Handbook not found", "Embedded handbook is missing.");
            return;
        }
        auto* dialog = new QDialog(this);
        dialog->setAttribute(Qt::WA_DeleteOnClose);
        dialog->setWindowTitle("Handbook");
        dialog->resize(900, 700);
        auto* layout = new QVBoxLayout(dialog);
        auto* viewer = new QTextBrowser(dialog);
        viewer->setOpenExternalLinks(true);
        viewer->setMarkdown(QString::fromUtf8(file.readAll()));
        layout->addWidget(viewer);
        dialog->setLayout(layout);
        dialog->show();
    });
    connect(copyLogAction, &QAction::triggered, this, [this]() {
        if (m_logPath.isEmpty()) {
            QMessageBox::warning(this, "Copy Log", "No log file path configured.");
            return;
        }
        QFile file(m_logPath);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            QMessageBox::warning(this, "Copy Log", "Unable to open the log file.");
            return;
        }
        QGuiApplication::clipboard()->setText(QString::fromUtf8(file.readAll()));
        logLine("Log copied from Help menu");
    });
    connect(debugCheckAction, &QAction::triggered, this, [this]() {
        runDebugCheck();
    });
    connect(licensesAction, &QAction::triggered, this, [this]() {
        const QString appName = QCoreApplication::applicationName().isEmpty()
            ? QString("PPUC-Serum-Colorizer")
            : QCoreApplication::applicationName();
        auto* dialog = new QDialog(this);
        dialog->setAttribute(Qt::WA_DeleteOnClose);
        dialog->setWindowTitle("Dependencies & Licenses");
        dialog->resize(720, 520);
        auto* layout = new QVBoxLayout(dialog);
        auto* viewer = new QTextBrowser(dialog);
        viewer->setOpenExternalLinks(true);
        const QString text = QStringLiteral(
            "# Dependencies & Licenses\n"
            "\n"
            "This build of %1 includes the following third‑party components:\n"
            "\n"
            "- **Qt 6** (LGPL/GPL; dynamic linking)\n"
            "- **OpenCV** (BSD‑3‑Clause)\n"
            "- **libserum** (GPLv2+)\n"
            "\n"
            "License texts are shipped in the `licenses/` folder next to the application.\n");
        viewer->setMarkdown(text.arg(appName));
        layout->addWidget(viewer);
        dialog->setLayout(layout);
        dialog->show();
    });
    connect(aboutAction, &QAction::triggered, this, [this]() {
        const QString appName = QCoreApplication::applicationName().isEmpty()
            ? QString("PPUC-Serum-Colorizer")
            : QCoreApplication::applicationName();
        const QString appVersion = QCoreApplication::applicationVersion().isEmpty()
            ? QString("0.1.0")
            : QCoreApplication::applicationVersion();
        QMessageBox::information(this,
                                 QString("About %1").arg(appName),
                                 QString("%1 v%2 (Qt port).").arg(appName, appVersion));
    });

    connect(m_state, &ProjectState::projectPathChanged, this, [this](const QString& path) {
        updateWindowTitle();
        m_projectLabel->setText(path.isEmpty() ? "None" : path);
        updateProjectLabelHeight();
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
        updateFrameJumpRange();
    });
    connect(m_shapeCompToggle, &QCheckBox::toggled, this, [this](bool enabled) {
        const std::vector<int> targets = targetFrameIndices();
        for (int row : targets) {
            if (row < 0 || row >= static_cast<int>(m_frameShapeCompModes.size())) {
                continue;
            }
            m_frameShapeCompModes[static_cast<std::size_t>(row)] = enabled ? 1 : 0;
        }
    });
    connect(m_triggerMonochromeCheck, &QCheckBox::toggled, this, [this](bool enabled) {
        if (!m_triggerIdSpin) {
            return;
        }
        const int frameCount = m_frameStore ? m_frameStore->count() : 0;
        if (frameCount <= 0) {
            return;
        }
        if (m_frameTriggerIds.size() < static_cast<std::size_t>(frameCount)) {
            m_frameTriggerIds.resize(static_cast<std::size_t>(frameCount), 0xffffffffu);
        }
        const uint32_t triggerValue = enabled ? static_cast<uint32_t>(kMonochromeTriggerId) : 0xffffffffu;
        const std::vector<int> targets = targetFrameIndices();
        if (targets.empty()) {
            return;
        }
        for (int row : targets) {
            if (row < 0 || row >= static_cast<int>(m_frameTriggerIds.size())) {
                continue;
            }
            m_frameTriggerIds[static_cast<std::size_t>(row)] = triggerValue;
        }
        if (m_hasLegacyRoundTrip) {
            m_legacyRoundTrip.trigger_ids = m_frameTriggerIds;
        }
        QSignalBlocker blockSpin(m_triggerIdSpin);
        if (enabled) {
            m_triggerIdSpin->setValue(kMonochromeTriggerId);
            m_triggerIdSpin->setReadOnly(true);
        } else {
            m_triggerIdSpin->setReadOnly(false);
            if (m_triggerIdSpin->value() >= kMonochromeTriggerId) {
                m_triggerIdSpin->setValue(-1);
            }
        }
    });
    connect(m_triggerIdSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (!m_triggerMonochromeCheck || m_triggerMonochromeCheck->isChecked()) {
            return;
        }
        if (value >= kMonochromeTriggerId) {
            QSignalBlocker blockSpin(m_triggerIdSpin);
            m_triggerIdSpin->setValue(kMonochromeTriggerId - 1);
            value = kMonochromeTriggerId - 1;
        }
        const int frameCount = m_frameStore ? m_frameStore->count() : 0;
        if (frameCount <= 0) {
            return;
        }
        if (m_frameTriggerIds.size() < static_cast<std::size_t>(frameCount)) {
            m_frameTriggerIds.resize(static_cast<std::size_t>(frameCount), 0xffffffffu);
        }
        const uint32_t triggerValue = (value < 0) ? 0xffffffffu : static_cast<uint32_t>(value);
        const std::vector<int> targets = targetFrameIndices();
        if (targets.empty()) {
            return;
        }
        for (int row : targets) {
            if (row < 0 || row >= static_cast<int>(m_frameTriggerIds.size())) {
                continue;
            }
            m_frameTriggerIds[static_cast<std::size_t>(row)] = triggerValue;
        }
        if (m_hasLegacyRoundTrip) {
            m_legacyRoundTrip.trigger_ids = m_frameTriggerIds;
        }
    });
    connect(m_spriteDetAreaCombo, &QComboBox::currentIndexChanged, this, [this](int index) {
        if (index < 0 || index >= MAX_SPRITE_DETECT_AREAS) {
            return;
        }
        m_spriteDetAreaIndex = index;
        updateSpriteCanvasImage(m_spritesList ? m_spritesList->currentRow() : -1);
    });
    connect(m_spriteDetAreaClearButton, &QPushButton::clicked, this, [this]() {
        const int spriteIndex = m_spritesList ? m_spritesList->currentRow() : -1;
        if (spriteIndex < 0) {
            return;
        }
        ensureSpriteDataSize();
        const std::size_t base = static_cast<std::size_t>(spriteIndex) * MAX_SPRITE_DETECT_AREAS * 4 +
            static_cast<std::size_t>(m_spriteDetAreaIndex) * 4;
        if (base + 3 >= m_spriteDetAreas.size()) {
            return;
        }
        pushSpriteDetAreaUndoSnapshot(spriteIndex);
        m_spriteDetAreas[base] = 0xffff;
        m_spriteDetAreas[base + 1] = 0xffff;
        m_spriteDetAreas[base + 2] = 0xffff;
        m_spriteDetAreas[base + 3] = 0xffff;
        updateSpriteCanvasImage(spriteIndex);
    });
    connect(m_hdCreateButton, &QPushButton::clicked, this, [this]() {
        if (m_canvasTabs && m_canvasTabs->currentWidget() == m_backgroundsCanvas) {
            const int index = m_backgroundList ? m_backgroundList->currentRow() : -1;
            if (index < 0 || !m_backgroundStore) {
                logLine("HD background create: invalid selection");
                return;
            }
            logLine(QString("HD background create: index=%1").arg(index));
            if (hasHdBackground(index)) {
                statusBar()->showMessage("HD background already exists.", 2000);
                logLine("HD background create: already exists");
                return;
            }
            const cv::Mat* src = m_backgroundStore->at(index);
            if (!src || src->empty()) {
                statusBar()->showMessage("No background image to upscale.", 2000);
                logLine("HD background create: missing source");
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
                logLine("HD background create: resize failed");
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
            logLine("HD background create: success");
            return;
        }
        if (m_canvasTabs && m_canvasTabs->currentWidget() == m_spritesCanvas) {
            const int index = m_spritesList ? m_spritesList->currentRow() : -1;
            if (index < 0 || !m_spriteStore) {
                logLine("HD sprite create: invalid selection");
                return;
            }
            logLine(QString("HD sprite create: index=%1").arg(index));
            if (hasHdSprite(index)) {
                statusBar()->showMessage("HD sprite already exists.", 2000);
                logLine("HD sprite create: already exists");
                return;
            }
            const cv::Mat* src = m_spriteStore->at(index);
            if (!src || src->empty()) {
                statusBar()->showMessage("No sprite image to upscale.", 2000);
                logLine("HD sprite create: missing source");
                return;
            }
            if (m_spriteColoredX.size() <= static_cast<std::size_t>(index)) {
                m_spriteColoredX.resize(static_cast<std::size_t>(index + 1));
            }
            if (m_spriteExtraFlags.size() <= static_cast<std::size_t>(index)) {
                m_spriteExtraFlags.resize(static_cast<std::size_t>(index + 1), 0);
            }
            const int interpolation = m_hdScaleCombo ? m_hdScaleCombo->currentData().toInt() : cv::INTER_NEAREST;
            const QRect contentRect = spriteContentRect(index);
            const cv::Size targetSize(MAX_SPRITE_WIDTH, MAX_SPRITE_HEIGHT);
            const cv::Mat srcBgr = EnsureBgr(*src);
            cv::Mat hd = UpscaleSpriteToHd(srcBgr, contentRect, interpolation, false, 0);
            if (hd.empty()) {
                statusBar()->showMessage("HD sprite create failed.", 2000);
                logLine("HD sprite create: resize failed");
                return;
            }
            m_spriteColoredX[static_cast<std::size_t>(index)] = hd;
            m_spriteExtraFlags[static_cast<std::size_t>(index)] = 1;
            if (m_spriteDynamicMasksX.size() <= static_cast<std::size_t>(index)) {
                m_spriteDynamicMasksX.resize(static_cast<std::size_t>(index + 1));
            }
            if (index < static_cast<int>(m_spriteDynamicMasks.size())) {
                const cv::Mat& sdMask = m_spriteDynamicMasks[static_cast<std::size_t>(index)];
                cv::Mat& hdMask = m_spriteDynamicMasksX[static_cast<std::size_t>(index)];
                if (!sdMask.empty() && (hdMask.empty() || hdMask.size() != targetSize)) {
                    hdMask = UpscaleSpriteToHd(sdMask, contentRect, cv::INTER_NEAREST, true, 255);
                }
            }
            if (m_spriteMasksX.size() <= static_cast<std::size_t>(index)) {
                m_spriteMasksX.resize(static_cast<std::size_t>(index + 1));
            }
            if (index < static_cast<int>(m_spriteOriginals.size())) {
                const cv::Mat& sdOriginal = m_spriteOriginals[static_cast<std::size_t>(index)];
                cv::Mat& hdMask = m_spriteMasksX[static_cast<std::size_t>(index)];
                if (!sdOriginal.empty() && (hdMask.empty() || hdMask.size() != targetSize)) {
                    hdMask = UpscaleSpriteToHd(sdOriginal, contentRect, cv::INTER_NEAREST, true, 255);
                }
            }
            m_useHdSprite = true;
            if (m_spritesCanvas) {
                m_spritesCanvas->setHdButtonChecked(true);
                m_spritesCanvas->setHdButtonEnabled(true);
            }
            updateHdControlsForContext();
            refreshFrameSpriteLists();
            updateSpriteCanvasImage(index);
            statusBar()->showMessage("HD sprite created.", 2000);
            logLine("HD sprite create: success");
            return;
        }
        const int frameCount = m_frameStore ? m_frameStore->count() : 0;
        if (frameCount <= 0) {
            logLine("HD frame create: no frames");
            return;
        }
        const std::vector<int> targets = targetFrameIndices();
        if (targets.empty()) {
            logLine("HD frame create: no targets");
            return;
        }
        logLine(QString("HD frame create: targets=%1").arg(targets.size()));
        if (m_frameExtraFrames.size() < static_cast<std::size_t>(frameCount)) {
            m_frameExtraFrames.resize(static_cast<std::size_t>(frameCount));
        }
        if (m_frameExtraFlags.size() < static_cast<std::size_t>(frameCount)) {
            m_frameExtraFlags.resize(static_cast<std::size_t>(frameCount), 0);
        }
        const int sourceMode = m_hdSourceCombo ? m_hdSourceCombo->currentData().toInt() : 0;
        const int interpolation = m_hdScaleCombo ? m_hdScaleCombo->currentData().toInt() : cv::INTER_NEAREST;
        std::unordered_set<int> spritesToScale;
        if (m_spriteStore) {
            const int spriteCount = m_spriteStore->count();
            for (int index : targets) {
                if (index < 0 || index >= frameCount) {
                    continue;
                }
                const std::size_t baseSlot = static_cast<std::size_t>(index) * MAX_SPRITES_PER_FRAME;
                if (baseSlot + MAX_SPRITES_PER_FRAME > m_frameSpriteAssignments.size()) {
                    continue;
                }
                for (int slot = 0; slot < MAX_SPRITES_PER_FRAME; ++slot) {
                    const uint8_t spriteId = m_frameSpriteAssignments[baseSlot + static_cast<std::size_t>(slot)];
                    if (spriteId == 255 || spriteId >= spriteCount) {
                        continue;
                    }
                    const bool hasExtra = hasHdSprite(static_cast<int>(spriteId));
                    if (!hasExtra) {
                        spritesToScale.insert(static_cast<int>(spriteId));
                    }
                }
            }
        }
        if (!spritesToScale.empty()) {
            QMessageBox::StandardButton reply = QMessageBox::question(
                this,
                "Create HD Sprites",
                QString("The selected frames use %1 sprite(s) without HD images.\n"
                        "Create HD sprites now?")
                    .arg(spritesToScale.size()),
                QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel,
                QMessageBox::Yes);
            if (reply == QMessageBox::Cancel) {
                return;
            }
            if (reply == QMessageBox::Yes && m_spriteStore) {
                for (int spriteIndex : spritesToScale) {
                    if (spriteIndex < 0 || spriteIndex >= m_spriteStore->count()) {
                        continue;
                    }
                    if (hasHdSprite(spriteIndex)) {
                        continue;
                    }
                    const cv::Mat* src = m_spriteStore->at(spriteIndex);
                    if (!src || src->empty()) {
                        continue;
                    }
                    if (m_spriteColoredX.size() <= static_cast<std::size_t>(spriteIndex)) {
                        m_spriteColoredX.resize(static_cast<std::size_t>(spriteIndex + 1));
                    }
                    if (m_spriteExtraFlags.size() <= static_cast<std::size_t>(spriteIndex)) {
                        m_spriteExtraFlags.resize(static_cast<std::size_t>(spriteIndex + 1), 0);
                    }
                    const QRect contentRect = spriteContentRect(spriteIndex);
                    const cv::Size targetSize(MAX_SPRITE_WIDTH, MAX_SPRITE_HEIGHT);
                    const cv::Mat srcBgr = EnsureBgr(*src);
                    cv::Mat hd = UpscaleSpriteToHd(srcBgr, contentRect, interpolation, false, 0);
                    if (hd.empty()) {
                        continue;
                    }
                    m_spriteColoredX[static_cast<std::size_t>(spriteIndex)] = hd;
                    m_spriteExtraFlags[static_cast<std::size_t>(spriteIndex)] = 1;
                    if (m_spriteDynamicMasksX.size() <= static_cast<std::size_t>(spriteIndex)) {
                        m_spriteDynamicMasksX.resize(static_cast<std::size_t>(spriteIndex + 1));
                    }
                    if (spriteIndex < static_cast<int>(m_spriteDynamicMasks.size())) {
                        const cv::Mat& sdMask = m_spriteDynamicMasks[static_cast<std::size_t>(spriteIndex)];
                        cv::Mat& hdMask = m_spriteDynamicMasksX[static_cast<std::size_t>(spriteIndex)];
                        if (!sdMask.empty() && (hdMask.empty() || hdMask.size() != targetSize)) {
                            hdMask = UpscaleSpriteToHd(sdMask, contentRect, cv::INTER_NEAREST, true, 255);
                        }
                    }
                    if (m_spriteMasksX.size() <= static_cast<std::size_t>(spriteIndex)) {
                        m_spriteMasksX.resize(static_cast<std::size_t>(spriteIndex + 1));
                    }
                    if (spriteIndex < static_cast<int>(m_spriteOriginals.size())) {
                        const cv::Mat& sdOriginal = m_spriteOriginals[static_cast<std::size_t>(spriteIndex)];
                        cv::Mat& hdMask = m_spriteMasksX[static_cast<std::size_t>(spriteIndex)];
                        if (!sdOriginal.empty() && (hdMask.empty() || hdMask.size() != targetSize)) {
                            hdMask = UpscaleSpriteToHd(sdOriginal, contentRect, cv::INTER_NEAREST, true, 255);
                        }
                    }
                }
                refreshFrameSpriteLists();
            }
        }
        std::unordered_set<int> backgroundsToScale;
        if (m_backgroundStore) {
            const int bgCount = m_backgroundStore->count();
            for (int index : targets) {
                if (index < 0 || index >= frameCount) {
                    continue;
                }
                if (index >= static_cast<int>(m_frameBackgroundIds.size())) {
                    continue;
                }
                const uint16_t bgId = m_frameBackgroundIds[static_cast<std::size_t>(index)];
                if (bgId == 0xffff || bgId >= bgCount) {
                    continue;
                }
                if (!hasHdBackground(static_cast<int>(bgId))) {
                    backgroundsToScale.insert(static_cast<int>(bgId));
                }
            }
        }
        if (!backgroundsToScale.empty()) {
            QMessageBox::StandardButton reply = QMessageBox::question(
                this,
                "Create HD Backgrounds",
                QString("The selected frames use %1 background(s) without HD images.\n"
                        "Create HD backgrounds now?")
                    .arg(backgroundsToScale.size()),
                QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel,
                QMessageBox::Yes);
            if (reply == QMessageBox::Cancel) {
                return;
            }
            if (reply == QMessageBox::Yes && m_backgroundStore) {
                const int bgCount = m_backgroundStore->count();
                const cv::Size targetSize(kDefaultFrameWidth * 2, kDefaultFrameHeight * 2);
                for (int bgIndex : backgroundsToScale) {
                    if (bgIndex < 0 || bgIndex >= bgCount) {
                        continue;
                    }
                    if (hasHdBackground(bgIndex)) {
                        continue;
                    }
                    const cv::Mat* src = m_backgroundStore->at(bgIndex);
                    if (!src || src->empty()) {
                        continue;
                    }
                    if (m_backgroundFramesX.size() <= static_cast<std::size_t>(bgIndex)) {
                        m_backgroundFramesX.resize(static_cast<std::size_t>(bgIndex + 1));
                    }
                    if (m_backgroundExtraFlags.size() <= static_cast<std::size_t>(bgIndex)) {
                        m_backgroundExtraFlags.resize(static_cast<std::size_t>(bgIndex + 1), 0);
                    }
                    cv::Mat hd;
                    if (interpolation == -1) {
                        cv::Mat scaled = Scale2xBgr(EnsureBgr(*src));
                        if (scaled.size() == targetSize) {
                            hd = scaled;
                        } else {
                            cv::resize(scaled, hd, targetSize, 0.0, 0.0, cv::INTER_NEAREST);
                        }
                    } else {
                        cv::resize(EnsureBgr(*src), hd, targetSize, 0.0, 0.0, interpolation);
                        if (interpolation == cv::INTER_LINEAR || interpolation == cv::INTER_CUBIC) {
                            hd = AdjustBrightnessToSource(*src, hd);
                        }
                    }
                    if (hd.empty()) {
                        continue;
                    }
                    m_backgroundFramesX[static_cast<std::size_t>(bgIndex)] = hd;
                    m_backgroundExtraFlags[static_cast<std::size_t>(bgIndex)] = 1;
                }
                refreshBackgroundList();
            }
        }
        bool createdAny = false;
        for (int index : targets) {
            if (index < 0 || index >= frameCount) {
                continue;
            }
            if (hasHdFrame(index)) {
                continue;
            }
            cv::Size baseSize;
            if (const cv::Mat* baseFrame = m_frameStore->at(index)) {
                if (baseFrame && !baseFrame->empty()) {
                    baseSize = baseFrame->size();
                }
            }
            if (baseSize.width <= 0 || baseSize.height <= 0) {
                cv::Mat reference = buildOriginalPreviewForIndex(index);
                if (!reference.empty()) {
                    baseSize = reference.size();
                }
            }
            cv::Mat source;
            if (sourceMode == 1) {
                cv::Mat reference = buildOriginalPreviewForIndex(index);
                source = buildOriginalFrame(reference);
            } else {
                if (const cv::Mat* frame = m_frameStore->at(index)) {
                    source = EnsureBgr(*frame);
                }
            }
            if (source.empty()) {
                continue;
            }
            if (baseSize.width <= 0 || baseSize.height <= 0) {
                baseSize = source.size();
            }
            const cv::Size targetSize(baseSize.width * 2, baseSize.height * 2);
            cv::Mat resized;
            if (interpolation == -1) {
                cv::Mat scaled = Scale2xBgr(source);
                if (scaled.size() == targetSize) {
                    resized = scaled;
                } else {
                    cv::resize(scaled, resized, targetSize, 0.0, 0.0, cv::INTER_NEAREST);
                }
            } else {
                cv::resize(source, resized, targetSize, 0.0, 0.0, interpolation);
                if (interpolation == cv::INTER_LINEAR || interpolation == cv::INTER_CUBIC) {
                    resized = AdjustBrightnessToSource(source, resized);
                }
            }
            if (resized.empty()) {
                continue;
            }
            m_frameExtraFrames[static_cast<std::size_t>(index)] = resized;
            m_frameExtraFlags[static_cast<std::size_t>(index)] = 1;
            if (m_serumDataLoaded) {
                commitFrameToSerum(index, resized, true);
            }
            ensureBackgroundDataSize();
            if (index < static_cast<int>(m_frameBackgroundMasks.size()) &&
                index < static_cast<int>(m_frameBackgroundMasksX.size())) {
                const cv::Mat& sdMask = m_frameBackgroundMasks[static_cast<std::size_t>(index)];
                cv::Mat& hdMask = m_frameBackgroundMasksX[static_cast<std::size_t>(index)];
                if (!sdMask.empty() && (hdMask.empty() || hdMask.size() != resized.size())) {
                    cv::resize(sdMask, hdMask, resized.size(), 0.0, 0.0, cv::INTER_NEAREST);
                }
            }
            if (m_frameDynamicMaskMapsX.size() <= static_cast<std::size_t>(index)) {
                m_frameDynamicMaskMapsX.resize(static_cast<std::size_t>(index + 1));
            }
            if (index < static_cast<int>(m_frameDynamicMaskMaps.size())) {
                const cv::Mat& sdMap = m_frameDynamicMaskMaps[static_cast<std::size_t>(index)];
                cv::Mat& hdMap = m_frameDynamicMaskMapsX[static_cast<std::size_t>(index)];
                if (!sdMap.empty() && (hdMap.empty() || hdMap.size() != resized.size())) {
                    cv::resize(sdMap, hdMap, resized.size(), 0.0, 0.0, cv::INTER_NEAREST);
                }
            }
            if (index < static_cast<int>(m_frameHdUndoStacks.size())) {
                m_frameHdUndoStacks[static_cast<std::size_t>(index)] = UndoStack{};
            }
            updateFramePreviewAt(index);
            createdAny = true;
        }
        const int current = m_framesList ? m_framesList->currentRow() : -1;
        if (current >= 0 && hasHdFrame(current)) {
            setHdMode(true);
        }
        if (createdAny) {
            refreshFrameSpriteLists();
            statusBar()->showMessage("HD frame created.", 2000);
            logLine("HD frame create: success");
        } else {
            logLine("HD frame create: no frames created");
        }
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
        if (m_canvasTabs && m_canvasTabs->currentWidget() == m_spritesCanvas) {
            const int index = m_spritesList ? m_spritesList->currentRow() : -1;
            if (index < 0 || index >= static_cast<int>(m_spriteColoredX.size())) {
                return;
            }
            m_spriteColoredX[static_cast<std::size_t>(index)] = cv::Mat();
            if (index < static_cast<int>(m_spriteMasksX.size())) {
                m_spriteMasksX[static_cast<std::size_t>(index)] = cv::Mat();
            }
            if (index < static_cast<int>(m_spriteDynamicMasksX.size())) {
                m_spriteDynamicMasksX[static_cast<std::size_t>(index)] = cv::Mat();
            }
            if (index < static_cast<int>(m_spriteDynamicColorsX.size())) {
                m_spriteDynamicColorsX[static_cast<std::size_t>(index)].clear();
            }
            if (index < static_cast<int>(m_spriteExtraFlags.size())) {
                m_spriteExtraFlags[static_cast<std::size_t>(index)] = 0;
            }
            updateHdControlsForContext();
            refreshFrameSpriteLists();
            updateSpriteCanvasImage(index);
            statusBar()->showMessage("HD sprite deleted.", 2000);
            return;
        }
        const int frameCount = m_frameStore ? m_frameStore->count() : 0;
        if (frameCount <= 0) {
            return;
        }
        const std::vector<int> targets = targetFrameIndices();
        if (targets.empty()) {
            return;
        }
        bool deletedAny = false;
        for (int index : targets) {
            if (index < 0 || index >= static_cast<int>(m_frameExtraFrames.size())) {
                continue;
            }
            if (m_frameExtraFrames[static_cast<std::size_t>(index)].empty()) {
                continue;
            }
            m_frameExtraFrames[static_cast<std::size_t>(index)] = cv::Mat();
            if (index < static_cast<int>(m_frameExtraFlags.size())) {
                m_frameExtraFlags[static_cast<std::size_t>(index)] = 0;
            }
            if (m_serumDataLoaded) {
                const uint8_t extra = 0;
                m_serumData.isextraframe.setIndex(static_cast<uint32_t>(index), &extra, 1);
            }
            if (index < static_cast<int>(m_frameHdUndoStacks.size())) {
                m_frameHdUndoStacks[static_cast<std::size_t>(index)] = UndoStack{};
            }
            updateFramePreviewAt(index);
            deletedAny = true;
        }
        const int current = m_framesList ? m_framesList->currentRow() : -1;
        if (current >= 0 && !hasHdFrame(current)) {
            setHdMode(false);
        }
        if (deletedAny) {
            refreshFrameSpriteLists();
            statusBar()->showMessage("HD frame deleted.", 2000);
        }
    });
    connect(m_frameMaskAssign, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
        const int id = m_frameMaskAssign->currentData().toInt();
        const std::vector<int> targets = targetFrameIndices();
        for (int row : targets) {
            if (row < 0 || row >= static_cast<int>(m_frameCompMaskIds.size())) {
                continue;
            }
            m_frameCompMaskIds[static_cast<std::size_t>(row)] = id >= 0 ? static_cast<uint8_t>(id) : 255;
            updateFramePreviewAt(row);
        }
        updateMaskPreviewForFrame(m_framesList->currentRow());
        if (m_previewFilterEnabled && currentPreviewFilterKind() == PreviewFilterKind::Mask) {
            updatePreviewFilterState();
        }
        if (m_previewMaskOverlayEnabled && m_maskMode == MaskMode::Comparison) {
            refreshFramePreviews();
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
        if (m_previewMaskOverlayEnabled && m_maskMode == MaskMode::Dynamic) {
            refreshFramePreviews();
        }
    });
    connect(m_spriteDynamicSetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        if (!m_spriteDynamicSetCombo || index < 0) {
            return;
        }
        m_spriteDynamicSetIndex = m_spriteDynamicSetCombo->currentData().toInt();
        if (m_spriteDynamicMaskMode) {
            updateSpriteCanvasImage(m_spritesList ? m_spritesList->currentRow() : -1);
        }
    });
    connect(m_frameDynamicCopyButton, &QPushButton::clicked, this, [this]() {
        const int frameCount = m_frameStore ? m_frameStore->count() : 0;
        const int sourceIndex = m_framesList ? m_framesList->currentRow() : -1;
        if (frameCount <= 0 || sourceIndex < 0) {
            return;
        }
        bool ok = false;
        const int targetDisplay = QInputDialog::getInt(this,
                                                       "Copy Dynamic Masks",
                                                       "Copy to frame number (1-based):",
                                                       sourceIndex + 1,
                                                       1,
                                                       frameCount,
                                                       1,
                                                       &ok);
        if (!ok) {
            return;
        }
        const int targetIndex = targetDisplay - 1;
        if (targetIndex < 0 || targetIndex >= frameCount || targetIndex == sourceIndex) {
            return;
        }
        ensureMaskDataSize();
        if (sourceIndex < static_cast<int>(m_frameDynamicMaskMaps.size()) &&
            targetIndex < static_cast<int>(m_frameDynamicMaskMaps.size())) {
            const cv::Mat& sourceMap = m_frameDynamicMaskMaps[static_cast<std::size_t>(sourceIndex)];
            m_frameDynamicMaskMaps[static_cast<std::size_t>(targetIndex)] = sourceMap.clone();
        }
        if (sourceIndex < static_cast<int>(m_frameDynamicMaskMapsX.size()) &&
            targetIndex < static_cast<int>(m_frameDynamicMaskMapsX.size())) {
            const cv::Mat& sourceMapX = m_frameDynamicMaskMapsX[static_cast<std::size_t>(sourceIndex)];
            m_frameDynamicMaskMapsX[static_cast<std::size_t>(targetIndex)] = sourceMapX.clone();
        }
        if (sourceIndex < static_cast<int>(m_frameDynamicColors.size()) &&
            targetIndex < static_cast<int>(m_frameDynamicColors.size())) {
            const bool useHd = m_useHdFrame && hasHdFrame(sourceIndex);
            int stride = 0;
            const uint16_t* data = frameDynamicColorsData(sourceIndex, useHd, &stride);
            std::vector<uint16_t>* targetColors = ensureFrameDynamicColorsLocal(targetIndex, useHd);
            if (targetColors && data && stride > 0) {
                const std::size_t size =
                    static_cast<std::size_t>(MAX_DYNA_SETS_PER_FRAMEN) * static_cast<std::size_t>(stride);
                targetColors->assign(data, data + size);
            } else if (targetColors) {
                targetColors->clear();
            }
        }
        updateFramePreviewAt(targetIndex);
        if (m_framesList && m_framesList->currentRow() == targetIndex) {
            updateMaskPreviewForFrame(targetIndex);
            refreshDynamicPaletteButtons();
            updateDynamicMaskPreviewIcons();
        }
        statusBar()->showMessage(QString("Copied dynamic data to frame %1").arg(targetDisplay), 2000);
    });
    connect(m_frameBackgroundAssign, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
        const int id = m_frameBackgroundAssign->currentData().toInt();
        const std::vector<int> targets = targetFrameIndices();
        if (!targets.empty() && m_frameBackgroundIds.size() < static_cast<std::size_t>(m_frameStore->count())) {
            m_frameBackgroundIds.resize(static_cast<std::size_t>(m_frameStore->count()), 0xffff);
        }
        for (int row : targets) {
            if (row < 0 || row >= static_cast<int>(m_frameBackgroundIds.size())) {
                continue;
            }
            m_frameBackgroundIds[static_cast<std::size_t>(row)] = id >= 0 ? static_cast<uint16_t>(id) : 0xffff;
            updateFramePreviewAt(row);
        }
        updateFrameCanvasImage(m_framesList ? m_framesList->currentRow() : -1);
        updateFrameUsageHighlights(m_framesList ? m_framesList->currentRow() : -1);
        if (m_previewFilterEnabled && currentPreviewFilterKind() == PreviewFilterKind::Background) {
            updatePreviewFilterState();
        }
    });
    connect(m_maskList, &QListWidget::currentRowChanged, this, [this](int) {
        updateMaskPreviewForFrame(m_framesList->currentRow());
        if (m_previewFilterEnabled && currentPreviewFilterKind() == PreviewFilterKind::Mask) {
            updatePreviewFilterState();
        }
        if (m_previewMaskOverlayEnabled && m_maskMode == MaskMode::Comparison) {
            refreshFramePreviews();
        }
    });
    connect(m_dynamicMaskList, &QListWidget::currentRowChanged, this, [this](int) {
        if (m_frameDynamicMaskAssign) {
            QSignalBlocker blocker(m_frameDynamicMaskAssign);
            const int selected = m_dynamicMaskList ? m_dynamicMaskList->currentRow() : -1;
            m_frameDynamicMaskAssign->setCurrentIndex(selected >= 0 ? selected + 1 : 0);
        }
        syncDynamicSetSelection();
        refreshDynamicPaletteButtons();
        updateMaskPreviewForFrame(m_framesList->currentRow());
        if (m_previewFilterEnabled && currentPreviewFilterKind() == PreviewFilterKind::DynamicMask) {
            updatePreviewFilterState();
        }
        if (m_previewMaskOverlayEnabled && m_maskMode == MaskMode::Dynamic) {
            refreshFramePreviews();
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
        recordHistory(m_backgroundHistory, row);
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
        if (cv::Mat* mask = activeComparisonMask()) {
            const int frameIndex = m_framesList ? m_framesList->currentRow() : -1;
            pushMaskUndoSnapshot(MaskMode::Comparison, frameIndex);
            mask->setTo(cv::Scalar(0));
            updateMaskPreviewIcons();
            updatePreviewsForMaskId(m_maskList ? m_maskList->currentRow() : -1);
            updateMaskPreviewForFrame(m_framesList->currentRow());
        }
    });
    connect(m_dynamicMaskClearButton, &QPushButton::clicked, this, [this]() {
        const int setId = currentFrameDynamicMaskId();
        if (setId < 0) {
            return;
        }
        const std::vector<int> targets = targetFrameIndices();
        const uint8_t target = static_cast<uint8_t>(setId);
        for (int frameIndex : targets) {
            cv::Mat* map = activeDynamicMaskMap(frameIndex);
            if (!map || map->empty()) {
                continue;
            }
            pushMaskUndoSnapshot(MaskMode::Dynamic, frameIndex);
            for (int y = 0; y < map->rows; ++y) {
                uint8_t* row = map->ptr<uint8_t>(y);
                for (int x = 0; x < map->cols; ++x) {
                    if (row[x] == target) {
                        row[x] = 255;
                    }
                }
            }
            updateFramePreviewAt(frameIndex);
        }
        updateDynamicMaskPreviewIcons();
        updateMaskPreviewForFrame(m_framesList->currentRow());
    });
    connect(m_spriteZoneAddButton, &QToolButton::clicked, this, [this]() {
        const int frameIndex = m_framesList ? m_framesList->currentRow() : -1;
        if (frameIndex < 0) {
            return;
        }
        ensureMaskDataSize();
        const cv::Mat* frame = m_frameStore ? m_frameStore->at(frameIndex) : nullptr;
        const int width = frame ? frame->cols : kDefaultFrameWidth;
        const int height = frame ? frame->rows : kDefaultFrameHeight;
        int slotToUse = -1;
        for (int slot = 0; slot < MAX_SPRITES_PER_FRAME; ++slot) {
            const std::size_t slotIndex = static_cast<std::size_t>(frameIndex) * MAX_SPRITES_PER_FRAME +
                static_cast<std::size_t>(slot);
            const std::size_t bboxIndex = static_cast<std::size_t>(frameIndex) * MAX_SPRITES_PER_FRAME * 4 +
                static_cast<std::size_t>(slot) * 4;
            if (slotIndex >= m_frameSpriteAssignments.size() ||
                bboxIndex + 3 >= m_frameSpriteBBoxes.size()) {
                break;
            }
            if (m_frameSpriteAssignments[slotIndex] != 255) {
                continue;
            }
            const bool emptyBox = m_frameSpriteBBoxes[bboxIndex] == 0 &&
                m_frameSpriteBBoxes[bboxIndex + 1] == 0 &&
                m_frameSpriteBBoxes[bboxIndex + 2] == 0 &&
                m_frameSpriteBBoxes[bboxIndex + 3] == 0;
            if (!emptyBox) {
                continue;
            }
            slotToUse = slot;
            break;
        }
        if (slotToUse < 0) {
            statusBar()->showMessage("No empty sprite zone slots available.", 2000);
            return;
        }
        const std::size_t bboxIndex = static_cast<std::size_t>(frameIndex) * MAX_SPRITES_PER_FRAME * 4 +
            static_cast<std::size_t>(slotToUse) * 4;
        m_frameSpriteBBoxes[bboxIndex] = 0;
        m_frameSpriteBBoxes[bboxIndex + 1] = 0;
        m_frameSpriteBBoxes[bboxIndex + 2] = static_cast<uint16_t>(std::max(0, width - 1));
        m_frameSpriteBBoxes[bboxIndex + 3] = static_cast<uint16_t>(std::max(0, height - 1));
        const std::size_t slotIndex = static_cast<std::size_t>(frameIndex) * MAX_SPRITES_PER_FRAME +
            static_cast<std::size_t>(slotToUse);
        if (slotIndex < m_frameSpriteZoneFlags.size()) {
            m_frameSpriteZoneFlags[slotIndex] = 1;
        }
        m_selectedSpriteSlot = slotToUse;
        refreshSpriteZoneList();
        refreshSpriteZoneSpritesList();
        updateMaskPreviewForFrame(frameIndex);
    });
    connect(m_spriteZoneRemoveButton, &QToolButton::clicked, this, [this]() {
        const int frameIndex = m_framesList ? m_framesList->currentRow() : -1;
        if (frameIndex < 0 || m_selectedSpriteZoneIndex < 0 ||
            m_selectedSpriteZoneIndex >= static_cast<int>(m_spriteZones.size())) {
            return;
        }
            const SpriteZoneGroup& zone = m_spriteZones[static_cast<std::size_t>(m_selectedSpriteZoneIndex)];
            for (int slot : zone.slotIndices) {
                const std::size_t slotIndex = static_cast<std::size_t>(frameIndex) * MAX_SPRITES_PER_FRAME +
                    static_cast<std::size_t>(slot);
                const std::size_t bboxIndex = static_cast<std::size_t>(frameIndex) * MAX_SPRITES_PER_FRAME * 4 +
                    static_cast<std::size_t>(slot) * 4;
                if (slotIndex >= m_frameSpriteAssignments.size() ||
                    bboxIndex + 3 >= m_frameSpriteBBoxes.size()) {
                    continue;
                }
                m_frameSpriteAssignments[slotIndex] = 255;
                m_frameSpriteBBoxes[bboxIndex] = 0;
                m_frameSpriteBBoxes[bboxIndex + 1] = 0;
                m_frameSpriteBBoxes[bboxIndex + 2] = 0;
                m_frameSpriteBBoxes[bboxIndex + 3] = 0;
                if (slotIndex < m_frameSpriteZoneFlags.size()) {
                    m_frameSpriteZoneFlags[slotIndex] = 0;
                }
            }
        refreshSpriteZoneList();
        refreshSpriteZoneSpritesList();
        updateFramePreviewAt(frameIndex);
        updateMaskPreviewForFrame(frameIndex);
        updateFrameUsageHighlights(frameIndex);
    });
    connect(m_spriteZoneList, &QListWidget::currentRowChanged, this, [this](int row) {
        if (row < 0) {
            m_selectedSpriteZoneIndex = -1;
        } else {
            m_selectedSpriteZoneIndex = row;
            if (m_selectedSpriteZoneIndex >= 0 &&
                m_selectedSpriteZoneIndex < static_cast<int>(m_spriteZones.size())) {
                const SpriteZoneGroup& zone = m_spriteZones[static_cast<std::size_t>(m_selectedSpriteZoneIndex)];
                if (!zone.slotIndices.empty()) {
                    if (std::find(zone.slotIndices.begin(), zone.slotIndices.end(), m_selectedSpriteSlot) == zone.slotIndices.end()) {
                        m_selectedSpriteSlot = zone.slotIndices.front();
                    }
                }
            }
        }
        refreshSpriteZoneSpritesList();
        updateMaskPreviewForFrame(m_framesList ? m_framesList->currentRow() : -1);
    });
    connect(m_spriteZoneSpritesList, &QListWidget::currentRowChanged, this, [this](int row) {
        const bool hasSelection = row >= 0;
        if (m_spriteZoneSpriteUp) {
            m_spriteZoneSpriteUp->setEnabled(hasSelection);
        }
        if (m_spriteZoneSpriteDown) {
            m_spriteZoneSpriteDown->setEnabled(hasSelection);
        }
        if (m_spriteZoneSpriteRemove) {
            m_spriteZoneSpriteRemove->setEnabled(hasSelection);
        }
    });
    connect(m_spriteZoneSpriteUp, &QToolButton::clicked, this, [this]() {
        if (!m_spriteZoneSpritesList || m_selectedSpriteZoneIndex < 0 ||
            m_selectedSpriteZoneIndex >= static_cast<int>(m_spriteZones.size())) {
            return;
        }
        const int frameIndex = m_framesList ? m_framesList->currentRow() : -1;
        if (frameIndex < 0) {
            return;
        }
        QListWidgetItem* item = m_spriteZoneSpritesList->currentItem();
        if (!item) {
            return;
        }
        const int slot = item->data(kSpriteZoneSlotRole).toInt();
        const int spriteIndex = item->data(kSpriteZoneSpriteIndexRole).toInt();
        const SpriteZoneGroup& zone = m_spriteZones[static_cast<std::size_t>(m_selectedSpriteZoneIndex)];
        auto it = std::find(zone.slotIndices.begin(), zone.slotIndices.end(), slot);
        if (it == zone.slotIndices.end() || it == zone.slotIndices.begin()) {
            return;
        }
        const int prevSlot = *(it - 1);
        const std::size_t base = static_cast<std::size_t>(frameIndex) * MAX_SPRITES_PER_FRAME;
        std::swap(m_frameSpriteAssignments[base + static_cast<std::size_t>(prevSlot)],
                  m_frameSpriteAssignments[base + static_cast<std::size_t>(slot)]);
        m_spriteZonePreferredSlot = prevSlot;
        refreshSpriteZoneList();
        refreshSpriteZoneSpritesList();
        updateFramePreviewAt(frameIndex);
        updateMaskPreviewForFrame(frameIndex);
        updateFrameUsageHighlights(frameIndex);
        if (m_framesList && frameIndex >= 0) {
            const int previewRow = previewRowForFrame(frameIndex);
            QSignalBlocker frameBlocker(m_framesList);
            m_framesList->setCurrentRow(frameIndex);
            if (previewRow >= 0 && m_framePreviewList) {
                QSignalBlocker previewBlocker(m_framePreviewList);
                m_framePreviewList->setCurrentRow(previewRow);
            }
            showFrameAtIndex(frameIndex);
        }
    });
    connect(m_spriteZoneSpriteDown, &QToolButton::clicked, this, [this]() {
        if (!m_spriteZoneSpritesList || m_selectedSpriteZoneIndex < 0 ||
            m_selectedSpriteZoneIndex >= static_cast<int>(m_spriteZones.size())) {
            return;
        }
        const int frameIndex = m_framesList ? m_framesList->currentRow() : -1;
        if (frameIndex < 0) {
            return;
        }
        QListWidgetItem* item = m_spriteZoneSpritesList->currentItem();
        if (!item) {
            return;
        }
        const int slot = item->data(kSpriteZoneSlotRole).toInt();
        const int spriteIndex = item->data(kSpriteZoneSpriteIndexRole).toInt();
        const SpriteZoneGroup& zone = m_spriteZones[static_cast<std::size_t>(m_selectedSpriteZoneIndex)];
        auto it = std::find(zone.slotIndices.begin(), zone.slotIndices.end(), slot);
        if (it == zone.slotIndices.end() || (it + 1) == zone.slotIndices.end()) {
            return;
        }
        const int nextSlot = *(it + 1);
        const std::size_t base = static_cast<std::size_t>(frameIndex) * MAX_SPRITES_PER_FRAME;
        std::swap(m_frameSpriteAssignments[base + static_cast<std::size_t>(nextSlot)],
                  m_frameSpriteAssignments[base + static_cast<std::size_t>(slot)]);
        m_spriteZonePreferredSlot = nextSlot;
        refreshSpriteZoneList();
        refreshSpriteZoneSpritesList();
        updateFramePreviewAt(frameIndex);
        updateMaskPreviewForFrame(frameIndex);
        updateFrameUsageHighlights(frameIndex);
        if (m_framesList && frameIndex >= 0) {
            const int previewRow = previewRowForFrame(frameIndex);
            QSignalBlocker frameBlocker(m_framesList);
            m_framesList->setCurrentRow(frameIndex);
            if (previewRow >= 0 && m_framePreviewList) {
                QSignalBlocker previewBlocker(m_framePreviewList);
                m_framePreviewList->setCurrentRow(previewRow);
            }
            showFrameAtIndex(frameIndex);
        }
    });
    connect(m_spriteZoneSpriteRemove, &QToolButton::clicked, this, [this]() {
        if (!m_spriteZoneSpritesList) {
            return;
        }
        const int frameIndex = m_framesList ? m_framesList->currentRow() : -1;
        if (frameIndex < 0) {
            return;
        }
        QListWidgetItem* item = m_spriteZoneSpritesList->currentItem();
        if (!item) {
            return;
        }
        const int slot = item->data(kSpriteZoneSlotRole).toInt();
        if (slot < 0 || slot >= MAX_SPRITES_PER_FRAME) {
            return;
        }
        const std::size_t slotIndex = static_cast<std::size_t>(frameIndex) * MAX_SPRITES_PER_FRAME +
            static_cast<std::size_t>(slot);
        if (slotIndex >= m_frameSpriteAssignments.size()) {
            return;
        }
        m_frameSpriteAssignments[slotIndex] = 255;
        refreshSpriteZoneList();
        refreshSpriteZoneSpritesList();
        updateFramePreviewAt(frameIndex);
        updateMaskPreviewForFrame(frameIndex);
        updateFrameUsageHighlights(frameIndex);
        if (m_previewFilterEnabled && currentPreviewFilterKind() == PreviewFilterKind::Sprite) {
            updatePreviewFilterState();
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
        if (m_playbackActive) {
            jumpPlayback(frameIndex);
        }
        if (m_previewSelectedFrames.size() > 1 &&
            std::find(m_previewSelectedFrames.begin(),
                      m_previewSelectedFrames.end(),
                      frameIndex) != m_previewSelectedFrames.end()) {
            QSignalBlocker blocker(m_framePreviewList);
            for (int i = 0; i < m_framePreviewList->count(); ++i) {
                QListWidgetItem* restore = m_framePreviewList->item(i);
                if (!restore) {
                    continue;
                }
                const int restoreIndex = restore->data(kFrameIndexRole).toInt();
                const bool selected = std::find(m_previewSelectedFrames.begin(),
                                                m_previewSelectedFrames.end(),
                                                restoreIndex) != m_previewSelectedFrames.end();
                restore->setSelected(selected);
            }
        }
        QSignalBlocker blocker(m_framesList);
        m_framesList->setCurrentRow(frameIndex);
        showFrameAtIndex(frameIndex);
        if (m_framesList && frameIndex >= 0 && frameIndex < m_framesList->count()) {
            const QListWidgetItem* frameItem = m_framesList->item(frameIndex);
            const QString frameText = frameItem ? frameItem->text() : QString("Frame %1").arg(frameIndex);
            setInspectorSelection(QString("Frame: %1").arg(frameText));
            m_framesCanvas->setTitle(QString("Frame canvas - %1").arg(frameText));
            m_framesCanvas->setStatusText(QString("Selected %1").arg(frameText));
        }
        updateMetadataForFrame(frameIndex);
        recordHistory(m_frameHistory, frameIndex);
        updateUndoActions();
    });
    connect(m_framePreviewList, &QListWidget::itemPressed, this, [this](QListWidgetItem* item) {
        if (!item) {
            m_restorePreviewSelection = false;
            return;
        }
        const Qt::KeyboardModifiers mods = QApplication::keyboardModifiers();
        if (mods == Qt::NoModifier) {
            const std::vector<int> selected = selectedPreviewFrameIndices();
            if (selected.size() > 1 && item->isSelected()) {
                m_restorePreviewSelection = true;
                m_restorePreviewCurrent = item->data(kFrameIndexRole).toInt();
                m_restorePreviewSelectionIndices = selected;
                return;
            }
        }
        m_restorePreviewSelection = false;
    });
    connect(m_framePreviewList, &QListWidget::itemSelectionChanged, this, [this]() {
        if (m_restorePreviewSelection) {
            QSignalBlocker blocker(m_framePreviewList);
            for (int row = 0; row < m_framePreviewList->count(); ++row) {
                QListWidgetItem* item = m_framePreviewList->item(row);
                if (!item) {
                    continue;
                }
                const int frameIndex = item->data(kFrameIndexRole).toInt();
                const bool selected = std::find(m_restorePreviewSelectionIndices.begin(),
                                                m_restorePreviewSelectionIndices.end(),
                                                frameIndex) != m_restorePreviewSelectionIndices.end();
                item->setSelected(selected);
                if (selected && frameIndex == m_restorePreviewCurrent) {
                    m_framePreviewList->setCurrentRow(row);
                }
            }
            m_restorePreviewSelection = false;
        }
        const std::vector<int> newSelection = selectedPreviewFrameIndices();
        if (m_previewSelectionClearRequested) {
            m_previewSelectionClearRequested = false;
            m_previewSelectedFrames = newSelection;
            schedulePreviewSelectionUpdate();
            return;
        }
        const bool previewHasFocus = m_framePreviewList->hasFocus() || m_framePreviewList->viewport()->hasFocus();
        if (!previewHasFocus && m_previewSelectedFrames.size() > 1 && newSelection.size() <= 1) {
            const int candidate = newSelection.empty() ? -1 : newSelection.front();
            if (candidate >= 0 &&
                std::find(m_previewSelectedFrames.begin(),
                          m_previewSelectedFrames.end(),
                          candidate) != m_previewSelectedFrames.end()) {
                QSignalBlocker blocker(m_framePreviewList);
                for (int row = 0; row < m_framePreviewList->count(); ++row) {
                    QListWidgetItem* item = m_framePreviewList->item(row);
                    if (!item) {
                        continue;
                    }
                    const int frameIndex = item->data(kFrameIndexRole).toInt();
                    item->setSelected(std::find(m_previewSelectedFrames.begin(),
                                                m_previewSelectedFrames.end(),
                                                frameIndex) != m_previewSelectedFrames.end());
                }
            } else if (candidate < 0) {
                QSignalBlocker blocker(m_framePreviewList);
                for (int row = 0; row < m_framePreviewList->count(); ++row) {
                    QListWidgetItem* item = m_framePreviewList->item(row);
                    if (!item) {
                        continue;
                    }
                    const int frameIndex = item->data(kFrameIndexRole).toInt();
                    item->setSelected(std::find(m_previewSelectedFrames.begin(),
                                                m_previewSelectedFrames.end(),
                                                frameIndex) != m_previewSelectedFrames.end());
                }
            } else {
                m_previewSelectedFrames = newSelection;
            }
        } else {
            m_previewSelectedFrames = newSelection;
        }
        schedulePreviewSelectionUpdate();
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
            if (m_previewFilterEnabled && currentPreviewFilterKind() == PreviewFilterKind::DynamicMask) {
                updatePreviewFilterState();
            }
            recordHistory(m_frameHistory, m_framesList->currentRow());
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
            recordHistory(m_spriteHistory, m_spritesList->currentRow());
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
            recordHistory(m_imageHistory, m_imagesList->currentRow());
        } else {
            updateSelectionFromLists();
        }
    });

    connect(m_framesCanvas->canvas(), &GLCanvasWidget::imageClicked, this, [this](int x, int y, Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
        handleToolPress(true, x, y, button, modifiers);
    });
    connect(m_framesCanvas->canvas(), &GLCanvasWidget::imageDragged, this, [this](int x, int y, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers) {
        handleToolDrag(true, x, y, buttons, modifiers);
    });
    connect(m_framesCanvas->canvas(), &GLCanvasWidget::imageReleased, this, [this](int x, int y, Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
        handleToolRelease(true, x, y, button, modifiers);
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
        const int gap = FrameGapForWidth(layout.topWidth);
        if (y >= 0 && y < layout.topHeight && x >= layout.topX && x < layout.topX + layout.topWidth) {
            m_frameHoverArea = FrameHoverArea::Top;
        } else if (y >= layout.topHeight + gap &&
                   y < layout.topHeight + gap + layout.bottomHeight &&
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
            const int gap = FrameGapForWidth(layout.topWidth);
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
        const cv::Mat* sprite = (index >= 0) ? activeSpriteImage(index) : nullptr;
        if (!sprite || sprite->empty()) {
            m_coordLabel->setText(QString());
            return;
        }
        const QRect contentRect = spriteContentRect(index);
        const QRect displayRect = spriteDisplayRect(index, *sprite);
        const int baseWidth = displayRect.isValid() ? displayRect.width() : sprite->cols;
        const int baseHeight = displayRect.isValid() ? displayRect.height() : sprite->rows;
        cv::Mat originalRef;
        if (const cv::Mat* originalSource = spriteOriginalForDisplay(index)) {
            if (contentRect.isValid() && !contentRect.isEmpty()) {
                originalRef = (*originalSource)(
                    cv::Rect(contentRect.x(), contentRect.y(), contentRect.width(), contentRect.height()));
            } else {
                originalRef = *originalSource;
            }
        }
        cv::Mat original = originalRef.empty() ? cv::Mat() : buildOriginalFrame(originalRef);
        cv::Mat displayOriginal = BuildDisplayOriginal(original, cv::Size(baseWidth, baseHeight));
        FrameLayout layout = BuildFrameLayout(baseWidth, baseHeight,
                                              displayOriginal.cols, displayOriginal.rows);
        if (y >= 0 && y < layout.topHeight &&
            x >= layout.topX && x < layout.topX + layout.topWidth) {
            const int localX = x - layout.topX;
            const int localY = y;
            const int scaleX = (layout.topWidth > 0 && baseWidth > 0) ? layout.topWidth / baseWidth : 1;
            const int scaleY = (layout.topHeight > 0 && baseHeight > 0) ? layout.topHeight / baseHeight : 1;
            const int mappedX = (scaleX > 1) ? localX / scaleX : localX;
            const int mappedY = (scaleY > 1) ? localY / scaleY : localY;
            const int spriteX = std::clamp(mappedX, 0, std::max(0, baseWidth - 1)) + 1;
            const int spriteY = std::clamp(mappedY, 0, std::max(0, baseHeight - 1)) + 1;
            m_coordLabel->setText(QString("Sprite %1,%2").arg(spriteX).arg(spriteY));
            return;
        }
        const int gap = FrameGapForWidth(layout.topWidth);
        if (!original.empty() &&
            y >= layout.topHeight + gap &&
            y < layout.topHeight + gap + layout.bottomHeight &&
            x >= layout.bottomX && x < layout.bottomX + layout.bottomWidth) {
            const int localX = x - layout.bottomX;
            const int localY = y - (layout.topHeight + gap);
            const int scaleX = (layout.bottomWidth > 0 && original.cols > 0) ? layout.bottomWidth / original.cols : 1;
            const int scaleY = (layout.bottomHeight > 0 && original.rows > 0) ? layout.bottomHeight / original.rows : 1;
            const int mappedX = (scaleX > 1) ? localX / scaleX : localX;
            const int mappedY = (scaleY > 1) ? localY / scaleY : localY;
            const int refX = std::clamp(mappedX, 0, std::max(0, original.cols - 1)) + 1;
            const int refY = std::clamp(mappedY, 0, std::max(0, original.rows - 1)) + 1;
            m_coordLabel->setText(QString("Original %1,%2").arg(refX).arg(refY));
            return;
        }
        m_coordLabel->setText(QString());
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
            const std::vector<int> targets = targetFrameIndices();
            for (int row : targets) {
                if (row < 0 || row >= static_cast<int>(m_frameCompMaskIds.size())) {
                    continue;
                }
                m_frameCompMaskIds[static_cast<std::size_t>(row)] = static_cast<uint8_t>(index);
                updateFramePreviewAt(row);
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
            statusBar()->showMessage(QString("Selected dynamic mask %1").arg(index), 2000);
        }
        if (kind == "background") {
            if (index < 0 || index >= m_backgroundList->count()) {
                return;
            }
            const std::vector<int> targets = targetFrameIndices();
            if (!targets.empty() && m_frameBackgroundIds.size() < static_cast<std::size_t>(m_frameStore->count())) {
                m_frameBackgroundIds.resize(static_cast<std::size_t>(m_frameStore->count()), 0xffff);
            }
            for (int row : targets) {
                if (row < 0 || row >= static_cast<int>(m_frameBackgroundIds.size())) {
                    continue;
                }
                m_frameBackgroundIds[static_cast<std::size_t>(row)] = static_cast<uint16_t>(index);
                updateFramePreviewAt(row);
            }
            if (m_frameBackgroundAssign) {
                m_frameBackgroundAssign->setCurrentIndex(index + 1);
            }
            statusBar()->showMessage(QString("Assigned background %1").arg(index), 2000);
            updateFrameCanvasImage(m_framesList->currentRow());
            updateFrameUsageHighlights(m_framesList ? m_framesList->currentRow() : -1);
        }
        if (kind == "sprite") {
            if (index < 0 || index >= m_spritesList->count()) {
                return;
            }
            if (!m_spriteZoneMode) {
                statusBar()->showMessage("Enable Zones to assign sprites.", 2000);
                return;
            }
            if (m_selectedSpriteZoneIndex < 0 || m_selectedSpriteZoneIndex >= static_cast<int>(m_spriteZones.size())) {
                statusBar()->showMessage("Select a sprite zone to assign a sprite.", 2000);
                return;
            }
            const int frameIndex = m_framesList->currentRow();
            if (frameIndex < 0) {
                return;
            }
            SpriteZoneGroup zone = m_spriteZones[static_cast<std::size_t>(m_selectedSpriteZoneIndex)];
            int slotToUse = -1;
            for (int slot : zone.slotIndices) {
                const std::size_t slotIndex = static_cast<std::size_t>(frameIndex) * MAX_SPRITES_PER_FRAME +
                    static_cast<std::size_t>(slot);
                if (slotIndex < m_frameSpriteAssignments.size() &&
                    m_frameSpriteAssignments[slotIndex] == 255) {
                    slotToUse = slot;
                    break;
                }
            }
            if (slotToUse < 0) {
                for (int slot = 0; slot < MAX_SPRITES_PER_FRAME; ++slot) {
                    const std::size_t slotIndex = static_cast<std::size_t>(frameIndex) * MAX_SPRITES_PER_FRAME +
                        static_cast<std::size_t>(slot);
                    if (slotIndex < m_frameSpriteAssignments.size() &&
                        m_frameSpriteAssignments[slotIndex] == 255) {
                        slotToUse = slot;
                        break;
                    }
                }
            }
            if (slotToUse < 0) {
                statusBar()->showMessage("No empty sprite slots available for this frame.", 2000);
                return;
            }
            const std::vector<int> targets = targetFrameIndices();
            for (int row : targets) {
                const std::size_t slotIndex = static_cast<std::size_t>(row) * MAX_SPRITES_PER_FRAME +
                    static_cast<std::size_t>(slotToUse);
                if (slotIndex >= m_frameSpriteAssignments.size()) {
                    continue;
                }
                m_frameSpriteAssignments[slotIndex] = static_cast<uint8_t>(index);
                if (slotIndex < m_frameSpriteZoneFlags.size()) {
                    m_frameSpriteZoneFlags[slotIndex] = 1;
                }
                const std::size_t bboxIndex = static_cast<std::size_t>(row) * MAX_SPRITES_PER_FRAME * 4 +
                    static_cast<std::size_t>(slotToUse) * 4;
                if (bboxIndex + 3 < m_frameSpriteBBoxes.size()) {
                    m_frameSpriteBBoxes[bboxIndex] = static_cast<uint16_t>(zone.rect.x());
                    m_frameSpriteBBoxes[bboxIndex + 1] = static_cast<uint16_t>(zone.rect.y());
                    m_frameSpriteBBoxes[bboxIndex + 2] = static_cast<uint16_t>(zone.rect.x() + zone.rect.width() - 1);
                    m_frameSpriteBBoxes[bboxIndex + 3] = static_cast<uint16_t>(zone.rect.y() + zone.rect.height() - 1);
                }
                updateFramePreviewAt(row);
            }
            m_selectedSpriteSlot = slotToUse;
            m_spriteZonePreferredSlot = slotToUse;
            statusBar()->showMessage(QString("Assigned sprite %1 to zone %2")
                                         .arg(index)
                                         .arg(m_selectedSpriteZoneIndex + 1),
                                     2000);
            refreshSpriteZoneList();
            refreshSpriteZoneSpritesList();
            if (m_previewFilterEnabled && currentPreviewFilterKind() == PreviewFilterKind::Sprite) {
                updatePreviewFilterState();
            }
        }
        updateMaskPreviewForFrame(m_framesList->currentRow());
    });
    connect(m_spritesCanvas->canvas(), &GLCanvasWidget::imageClicked, this, [this](int x, int y, Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
        handleToolPress(false, x, y, button, modifiers);
    });
    connect(m_spritesCanvas->canvas(), &GLCanvasWidget::imageDragged, this, [this](int x, int y, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers) {
        handleToolDrag(false, x, y, buttons, modifiers);
    });
    connect(m_spritesCanvas->canvas(), &GLCanvasWidget::imageReleased, this, [this](int x, int y, Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
        handleToolRelease(false, x, y, button, modifiers);
    });
    connect(m_backgroundsCanvas->canvas(), &GLCanvasWidget::imageClicked, this, [this](int x, int y, Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
        handleBackgroundToolPress(x, y, button, modifiers);
    });
    connect(m_backgroundsCanvas->canvas(), &GLCanvasWidget::imageDragged, this, [this](int x, int y, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers) {
        handleBackgroundToolDrag(x, y, buttons, modifiers);
    });
    connect(m_backgroundsCanvas->canvas(), &GLCanvasWidget::imageReleased, this, [this](int x, int y, Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
        handleBackgroundToolRelease(x, y, button, modifiers);
    });

    connect(m_framesCanvas, &CanvasWidget::fitRequested, this, [this]() {
        m_framesCanvas->canvas()->requestFitOnResize(true);
    });
    connect(m_framesCanvas, &CanvasWidget::backRequested, this, [this]() {
        navigateHistory(m_frameHistory, m_framesList, false);
    });
    connect(m_framesCanvas, &CanvasWidget::forwardRequested, this, [this]() {
        navigateHistory(m_frameHistory, m_framesList, true);
    });
    connect(m_spritesCanvas, &CanvasWidget::fitRequested, this, [this]() {
        m_spritesCanvas->canvas()->requestFitOnResize(true);
    });
    connect(m_spritesCanvas, &CanvasWidget::backRequested, this, [this]() {
        navigateHistory(m_spriteHistory, m_spritesList, false);
    });
    connect(m_spritesCanvas, &CanvasWidget::forwardRequested, this, [this]() {
        navigateHistory(m_spriteHistory, m_spritesList, true);
    });
    connect(m_backgroundsCanvas, &CanvasWidget::fitRequested, this, [this]() {
        m_backgroundsCanvas->canvas()->requestFitOnResize(true);
    });
    connect(m_playbackCanvas, &CanvasWidget::fitRequested, this, [this]() {
        m_playbackCanvas->canvas()->requestFitOnResize(true);
    });
    connect(m_backgroundsCanvas, &CanvasWidget::backRequested, this, [this]() {
        navigateHistory(m_backgroundHistory, m_backgroundList, false);
    });
    connect(m_backgroundsCanvas, &CanvasWidget::forwardRequested, this, [this]() {
        navigateHistory(m_backgroundHistory, m_backgroundList, true);
    });
    connect(m_imagesCanvas, &CanvasWidget::backRequested, this, [this]() {
        navigateHistory(m_imageHistory, m_imagesList, false);
    });
    connect(m_imagesCanvas, &CanvasWidget::forwardRequested, this, [this]() {
        navigateHistory(m_imageHistory, m_imagesList, true);
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
    connect(m_playbackCanvas, &CanvasWidget::gridToggled, this, [this](bool enabled) {
        m_playbackCanvas->canvas()->setGridEnabled(enabled);
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
    connect(m_spritesCanvas, &CanvasWidget::maskToggled, this, [this](bool enabled) {
        setSpriteDetAreaMode(enabled);
    });
    connect(m_framesCanvas, &CanvasWidget::zoneToggled, this, [this](bool enabled) {
        setSpriteZoneMode(enabled);
    });
    connect(m_spritesCanvas, &CanvasWidget::dynamicToggled, this, [this](bool enabled) {
        m_spriteDynamicMaskMode = enabled;
        if (enabled && m_spriteDetAreaMode) {
            m_spriteDetAreaMode = false;
            m_spritesCanvas->setMaskButtonsChecked(false, true);
        }
        m_spritesCanvas->canvas()->clearPreviewImage();
        updateSpriteCanvasImage(m_spritesList ? m_spritesList->currentRow() : -1);
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
        refreshFramePreviews();
    });
    connect(m_framesCanvas, &CanvasWidget::originalToggled, this, [this](bool enabled) {
        m_showOriginalFrame = enabled;
        updateFrameCanvasImage(m_framesList->currentRow());
        updateMaskPreviewForFrame(m_framesList->currentRow());
    });
    connect(m_spritesCanvas, &CanvasWidget::originalToggled, this, [this](bool enabled) {
        m_showSpriteOriginal = enabled;
        updateSpriteCanvasImage(m_spritesList ? m_spritesList->currentRow() : -1);
    });
    connect(m_playbackCanvas, &CanvasWidget::originalToggled, this, [this](bool enabled) {
        m_playbackShowOriginal = enabled;
        if (m_playbackActive) {
            renderPlaybackFrame();
        }
    });
    connect(m_framesCanvas, &CanvasWidget::hdToggled, this, [this](bool enabled) {
        setHdMode(enabled);
    });
    connect(m_spritesCanvas, &CanvasWidget::hdToggled, this, [this](bool enabled) {
        m_useHdSprite = enabled;
        updateSpriteCanvasImage(m_spritesList ? m_spritesList->currentRow() : -1);
        updateHdControlsForContext();
    });
    connect(m_framesCanvas, &CanvasWidget::rotateToggled, this, [this](bool enabled) {
        setCanvasRotationEnabled(enabled);
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
        if (m_previewSelectedButton) {
            m_previewSelectedButton->setChecked(false);
        }
        m_previewSelectedOnly = false;
        updatePreviewFilterState();
    });
    connect(m_previewSelectedButton, &QToolButton::toggled, this, [this](bool enabled) {
        m_previewSelectedOnly = enabled;
        refreshFramePreviews();
    });
    connect(m_previewHdButton, &QToolButton::toggled, this, [this](bool enabled) {
        m_previewHdOnly = enabled;
        refreshFramePreviews();
    });
    connect(m_previewMaskOverlayButton, &QToolButton::toggled, this, [this](bool enabled) {
        m_previewMaskOverlayEnabled = enabled;
        refreshFramePreviews();
    });
    connect(m_previewRotateButton, &QToolButton::toggled, this, [this](bool enabled) {
        m_previewRotateEnabled = enabled;
        if (enabled) {
            m_previewRotationStates.clear();
            m_previewRotationStatesX.clear();
            m_previewRotationClock.restart();
            schedulePreviewRotationUpdate();
        } else if (m_previewRotationTimer) {
            m_previewRotationTimer->stop();
        }
        refreshFramePreviews();
    });
    connect(m_previewPlayButton, &QToolButton::clicked, this, [this]() {
        if (!m_playbackActive) {
            startPlayback();
        } else if (m_playbackPaused) {
            resumePlayback();
        } else {
            pausePlayback();
        }
    });
    connect(m_previewStopButton, &QToolButton::clicked, this, [this]() {
        stopPlayback();
    });
    connect(m_previewPauseButton, &QToolButton::clicked, this, [this]() {
        pausePlayback();
    });
    connect(m_previewPrevButton, &QToolButton::clicked, this, [this]() {
        stepPlayback(-1);
    });
    connect(m_previewNextButton, &QToolButton::clicked, this, [this]() {
        stepPlayback(1);
    });
    connect(m_previewRewindButton, &QToolButton::clicked, this, [this]() {
        stepPlayback(-10);
    });
    connect(m_previewFastForwardButton, &QToolButton::clicked, this, [this]() {
        stepPlayback(10);
    });
    connect(m_playbackOverrideDurationCheck, &QCheckBox::toggled, this, [this](bool enabled) {
        if (!m_playbackOverrideDurationSpin) {
            return;
        }
        m_playbackOverrideDurationSpin->setEnabled(enabled);
        if (m_playbackActive) {
            if (enabled) {
                m_playbackFrameDurationMs = m_playbackOverrideDurationSpin->value();
            } else if (m_playbackFrameIndex >= 0 &&
                       m_playbackFrameIndex < static_cast<int>(m_frameDurations.size()) &&
                       m_frameDurations[m_playbackFrameIndex] > 0) {
                m_playbackFrameDurationMs = static_cast<int>(m_frameDurations[m_playbackFrameIndex]);
            } else {
                m_playbackFrameDurationMs = 30;
            }
            m_playbackFrameClock.restart();
            schedulePlaybackTick();
        }
    });
    connect(m_playbackOverrideDurationSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (!m_playbackOverrideDurationCheck || !m_playbackOverrideDurationCheck->isChecked()) {
            return;
        }
        if (m_playbackActive) {
            m_playbackFrameDurationMs = value;
            m_playbackFrameClock.restart();
            schedulePlaybackTick();
        }
    });
    connect(m_previewRefreshButton, &QToolButton::clicked, this, [this]() {
        refreshAllPreviews();
    });
    m_rotationTimer = new QTimer(this);
    m_rotationTimer->setSingleShot(true);
    connect(m_rotationTimer, &QTimer::timeout, this, [this]() {
        updateCanvasRotationFrame();
    });
    m_previewRotationTimer = new QTimer(this);
    m_previewRotationTimer->setSingleShot(true);
    connect(m_previewRotationTimer, &QTimer::timeout, this, [this]() {
        refreshFramePreviewsForRotation();
        schedulePreviewRotationUpdate();
    });
    m_playbackTimer = new QTimer(this);
    m_playbackTimer->setSingleShot(true);
    connect(m_playbackTimer, &QTimer::timeout, this, [this]() {
        schedulePlaybackTick();
    });
    m_previewSelectionTimer = new QTimer(this);
    m_previewSelectionTimer->setSingleShot(true);
    connect(m_previewSelectionTimer, &QTimer::timeout, this, [this]() {
        updatePreviewSelectionStyles();
        if (m_previewSelectedOnly) {
            refreshFramePreviews();
        }
    });
    updatePlaybackButtons();
    m_paletteBlinkTimer = new QTimer(this);
    m_paletteBlinkTimer->setInterval(350);
    connect(m_paletteBlinkTimer, &QTimer::timeout, this, [this]() {
        if (!m_paletteBlinkActive) {
            return;
        }
        m_paletteBlinkOn = !m_paletteBlinkOn;
        updatePaletteBlinkPreview();
    });
    initPalette();
    refreshPaletteList();
    refreshReducedPaletteUI();
    refreshDynamicPaletteUI();
    refreshRotationEditor();
    setDrawColor(QColor(255, 255, 255), true);
    if (m_paletteAssignButton) {
        m_paletteAssignButton->setEnabled(false);
    }
    connect(m_paletteList, &QListWidget::currentRowChanged, this, [this](int row) {
        if (row >= 0 && row < m_paletteColors.size()) {
            if (m_reducedSetSlotActive || m_dynamicSetSlotActive || m_rotationSetSlotActive) {
                cancelPaletteSetSlot();
                return;
            }
            m_currentPaletteIndex = row;
            const bool reference = m_paletteSelectionIsReference;
            if (m_paletteList) {
                const QColor color = reference ? QColor(70, 150, 255) : QColor(255, 140, 0);
                m_paletteList->setProperty("selectionColor", color);
                m_paletteList->viewport()->update();
            }
            if (m_paletteSetSlotActive) {
                const QColor current(static_cast<int>(m_drawColor[2]),
                                     static_cast<int>(m_drawColor[1]),
                                     static_cast<int>(m_drawColor[0]));
                const std::vector<QColor> before(m_paletteColors.begin(), m_paletteColors.end());
                pushPaletteUndoSnapshot();
                m_paletteColors[row] = current;
                if (m_paletteSetIndex >= 0 && m_paletteSetIndex < m_fullPalettes.size()) {
                    m_fullPalettes[m_paletteSetIndex] = m_paletteColors;
                }
                applyPaletteColorChanges(before, std::vector<QColor>(m_paletteColors.begin(), m_paletteColors.end()));
                m_currentPaletteIndex = row;
                m_paletteSelectionIsReference = false;
                if (m_paletteList) {
                    m_paletteList->setProperty("selectionColor", QColor(255, 140, 0));
                    m_paletteList->viewport()->update();
                }
                refreshPaletteList();
                cancelPaletteSetSlot();
                if (m_paletteList) {
                    QSignalBlocker blocker(m_paletteList);
                    m_paletteList->setCurrentRow(row);
                    m_paletteList->viewport()->update();
                }
                if (m_paletteGradientButton) {
                    QTimer::singleShot(0, this, [this]() {
                        if (!m_paletteGradientButton) {
                            return;
                        }
                        const bool hasSelection = m_paletteList && m_paletteList->currentRow() >= 0;
                        const bool enable = hasSelection && !m_paletteSelectionIsReference;
                        m_paletteGradientButton->setEnabled(enable);
                    });
                }
                return;
            }
            if (m_paletteGradientActive && m_paletteGradientStartIndex >= 0 && m_paletteGradientStartIndex != row) {
                const std::vector<QColor> before(m_paletteColors.begin(), m_paletteColors.end());
                pushPaletteUndoSnapshot();
                applyPaletteGradient(m_paletteGradientStartIndex, row);
                cancelPaletteGradient();
                refreshPaletteList();
                applyPaletteColorChanges(before, std::vector<QColor>(m_paletteColors.begin(), m_paletteColors.end()));
            }
            if (!reference) {
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
            }
            setDrawColor(m_paletteColors[row], false);
            if (m_paletteAssignButton) {
                m_paletteAssignButton->setEnabled(true);
            }
            if (m_paletteGradientButton) {
                m_paletteGradientButton->setEnabled(!reference);
            }
        } else if (m_paletteGradientButton) {
            m_paletteGradientButton->setEnabled(false);
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
        const std::vector<QColor> before(m_paletteColors.begin(), m_paletteColors.end());
        pushPaletteUndoSnapshot();
        const cv::Vec3b quant = Rgb565ToBgr(BgrToRgb565(cv::Vec3b(picked.blue(), picked.green(), picked.red())));
        m_paletteColors[row] = QColor(quant[2], quant[1], quant[0]);
        if (m_paletteSetIndex >= 0 && m_paletteSetIndex < m_fullPalettes.size()) {
            m_fullPalettes[m_paletteSetIndex] = m_paletteColors;
        }
        refreshPaletteList();
        applyPaletteColorChanges(before, std::vector<QColor>(m_paletteColors.begin(), m_paletteColors.end()));
        m_paletteList->setCurrentRow(row);
    });
    connect(m_paletteList, &QListWidget::itemClicked, this, [this](QListWidgetItem*) {
        if (m_rotationSetSlotActive) {
            cancelPaletteSetSlot();
            return;
        }
        if (m_paletteGradientActive || m_paletteSetSlotActive) {
            return;
        }
        m_paletteSelectionIsReference = false;
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
        if (m_paletteList) {
            m_paletteList->setProperty("selectionColor", QColor(255, 140, 0));
            m_paletteList->viewport()->update();
        }
        if (m_paletteGradientButton) {
            const bool hasSelection = m_paletteList && m_paletteList->currentRow() >= 0;
            m_paletteGradientButton->setEnabled(hasSelection);
        }
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
        startPaletteSetSlot();
    });
    connect(m_paletteGradientButton, &QPushButton::clicked, this, [this]() {
        startPaletteGradient();
    });
    connect(m_paletteSetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        if (index < 0 || index >= m_fullPalettes.size()) {
            return;
        }
        if (m_paletteSetSlotActive || m_reducedSetSlotActive || m_dynamicSetSlotActive || m_rotationSetSlotActive) {
            cancelPaletteSetSlot();
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
        if (m_paletteSetSlotActive || m_reducedSetSlotActive || m_dynamicSetSlotActive || m_rotationSetSlotActive) {
            cancelPaletteSetSlot();
        }
        m_reducedPaletteIndex = index;
        refreshReducedPaletteButtons();
    });
    connect(m_dynamicSetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        if (index < 0 || index >= MAX_DYNA_SETS_PER_FRAMEN) {
            return;
        }
        if (m_paletteSetSlotActive || m_reducedSetSlotActive || m_dynamicSetSlotActive || m_rotationSetSlotActive) {
            cancelPaletteSetSlot();
        }
        m_dynamicSetIndex = index;
        setCurrentFrameDynamicMaskId(index);
        updateMaskPreviewForFrame(m_framesList ? m_framesList->currentRow() : -1);
        refreshDynamicPaletteButtons();
        updateDynamicMaskPreviewIcons();
    });
    connect(m_reducedAssignButton, &QPushButton::clicked, this, [this]() {
        startReducedSetSlot();
    });
    connect(m_dynamicAssignButton, &QPushButton::clicked, this, [this]() {
        startDynamicSetSlot();
    });
    connect(m_toolsTabs, &QStackedWidget::currentChanged, this, [this](int) {
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

    QSettings settings("PPUC", "PPUC-Serum-Colorizer");
    const QStringList recent = settings.value("recentFiles").toStringList();
    if (!recent.isEmpty()) {
        m_state->setRecentFiles(recent);
    }
    m_maxUndoDepth = std::clamp(settings.value("maxUndoDepth", kDefaultUndoDepth).toInt(), 1, 1000);
    m_maxHistoryDepth = std::clamp(settings.value("maxHistoryDepth", kDefaultHistoryDepth).toInt(), 1, 1000);
    m_frameCacheLimit = std::clamp(settings.value("frameCacheLimit", 16).toInt(), 1, 256);
    m_spriteCacheLimit = std::clamp(settings.value("spriteCacheLimit", 8).toInt(), 1, 256);
    m_backgroundCacheLimit = std::clamp(settings.value("backgroundCacheLimit", 4).toInt(), 1, 256);
    m_autosaveEnabled = settings.value("autosaveEnabled", true).toBool();
    m_autosaveIntervalMinutes = std::clamp(settings.value("autosaveIntervalMinutes", 1).toInt(), 1, 120);
    const bool wasCleanShutdown = settings.value("lastShutdownClean", true).toBool();
    m_loggingEnabled = settings.value("loggingEnabled", true).toBool();
    const QString defaultLogDir = QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation);
    const QString logDir = defaultLogDir.isEmpty()
        ? QDir::home().filePath(".ppuc-serum-colorizer")
        : defaultLogDir;
    QDir().mkpath(logDir);
    m_logPath = settings.value("logPath", QDir(logDir).filePath("ppuc-serum-colorizer.log")).toString();
    settings.setValue("lastShutdownClean", false);
    settings.sync();
    initLogging();
    if (!wasCleanShutdown) {
        QTimer::singleShot(0, this, [this]() { showCrashLogDialog(); });
    }
    if (!m_autosaveTimer) {
        m_autosaveTimer = new QTimer(this);
        connect(m_autosaveTimer, &QTimer::timeout, this, [this]() {
            autosaveProject(false);
        });
    }
    updateAutosaveTimer();
    resetNavigationHistory();
    m_uiReady = true;
}

void MainWindow::updateWindowTitle()
{
    const QString appName = QCoreApplication::applicationName().isEmpty()
        ? QString("PPUC-Serum-Colorizer")
        : QCoreApplication::applicationName();
    const QString appVersion = QCoreApplication::applicationVersion();
    if (m_state->projectPath().isEmpty()) {
        setWindowTitle(appVersion.isEmpty() ? appName : QString("%1 v%2").arg(appName, appVersion));
        return;
    }
    setWindowTitle(QString("%1 - %2").arg(appName, m_state->projectPath()));
}

void MainWindow::persistRecentFiles()
{
    QSettings settings("PPUC", "PPUC-Serum-Colorizer");
    settings.setValue("recentFiles", m_state->recentFiles());
    settings.sync();
}

void MainWindow::openProjectFile(const QString& filename)
{
    if (filename.isEmpty()) {
        return;
    }
    if (!m_uiReady) {
        QTimer::singleShot(150, this, [this, filename]() {
            openProjectFile(filename);
        });
        return;
    }
    logLine(QString("Open project: %1").arg(filename));
    disposeSerumRuntime();
    stopPlayback();
    QScopedValueRollback<bool> loadGuard(m_isLoadingProject, true);
    const QFileInfo info(filename);
    const QString suffix = info.suffix().toLower();
    if (suffix == "crom" || suffix == "crp" || suffix == "cromc") {
        if (suffix == "crom" && IsLikelyJsonFile(filename)) {
            QString error;
            m_imageStore->clear();
            m_frameStore->clear();
            m_spriteStore->clear();
            m_backgroundStore->clear();
            m_frameDurations.clear();
            m_spriteNames.clear();
            m_spriteColored.clear();
            m_spriteColoredX.clear();
            m_spriteOriginals.clear();
            m_spriteMasksX.clear();
            m_spriteDynamicMasks.clear();
            m_spriteDynamicMasksX.clear();
            m_spriteDynamicColors.clear();
            m_spriteDynamicColorsX.clear();
            m_spriteExtraFlags.clear();
            m_spriteShapeModes.clear();
            m_spriteDetAreas.clear();
            m_spriteDetDwords.clear();
            m_spriteDetDwordPos.clear();
            m_frameSpriteAssignments.clear();
            m_frameSpriteBBoxes.clear();
            m_spriteColFromFrame.clear();
            m_spriteRects.clear();
            m_spriteRectMirror.clear();
            m_sectionStarts.clear();
            m_sectionNames.clear();
            m_frameRefs.clear();
            m_frameDynamicColors.clear();
            m_compMasks.clear();
            m_frameCompMaskIds.clear();
            m_frameDynamicMaskMaps.clear();
            m_frameDynamicMaskMapsX.clear();
            m_frameShapeCompModes.clear();
            m_frameRotations.clear();
            m_frameRotationsX.clear();
            m_frameRotationsLocal.clear();
            m_frameRotationsLocalX.clear();
            m_frameExtraFrames.clear();
            m_frameExtraFlags.clear();
            m_backgroundFramesX.clear();
            m_backgroundExtraFlags.clear();
            m_frameBackgroundIds.clear();
            m_frameBackgroundMasks.clear();
            m_frameBackgroundMasksX.clear();
            m_useHdFrame = false;
            m_noColors = 64;
            m_serumData.Clear();
            m_serumDataLoaded = false;
            configureFrameStoreAdapter();
            updateMetadataForFrame(-1);
            updateMetadataForSprite(-1);
            populateBookmarks({}, {});
            m_hasLegacyRoundTrip = false;
            m_legacyRoundTrip = LegacyRoundTripData{};
            if (LoadProjectJson(*m_state, filename, &error)) {
                resetUndoStacks();
                resetNavigationHistory();
                m_projectDirty = false;
                statusBar()->showMessage(QString("Open: %1").arg(filename), 5000);
                persistRecentFiles();
                QTimer::singleShot(0, this, [this]() {
                    refreshAllPreviews();
                    refreshFrameSpriteLists();
                });
            } else {
                statusBar()->showMessage(QString("Open failed: %1").arg(error), 5000);
            }
            return;
        }
        QString cromPath;
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
        } else if (suffix == "cromc") {
            const QString base = info.completeBaseName();
            const QString dir = info.absolutePath();
            const QString candidate = dir + "/" + base + ".cRP";
            const QString candidateLower = dir + "/" + base + ".crp";
            if (QFileInfo::exists(candidate)) {
                rpPath = candidate;
            } else if (QFileInfo::exists(candidateLower)) {
                rpPath = candidateLower;
            }
        } else {
            const QString base = info.completeBaseName();
            const QString dir = info.absolutePath();
            const QString candidate = dir + "/" + base + ".cRP";
            if (QFileInfo::exists(candidate)) {
                rpPath = candidate;
            }
            cromPath = filename;
        }
        QString base;
        QString dir;
        if (suffix == "cromc") {
            base = info.completeBaseName();
            dir = info.absolutePath();
        } else if (!cromPath.isEmpty()) {
            base = QFileInfo(cromPath).completeBaseName();
            dir = QFileInfo(cromPath).absolutePath();
        } else {
            base = info.completeBaseName();
            dir = info.absolutePath();
        }
        QString cromcPath = dir + "/" + base + ".cROMc";
        const QString projectCromcPath = cromcPath;
        m_projectDir = dir;
        m_projectBaseName = base;
        m_serumRuntimeAltDir = dir;
        m_serumRuntimeRomName = base;
        const QString autoBase = base + ".autosave";
        const QString autoRp = QDir(dir).filePath(autoBase + ".cRP");
        const QString autoCromc = QDir(dir).filePath(autoBase + ".cROMc");
        if (autosaveFilesDiffer(rpPath, cromcPath, autoRp, autoCromc)) {
            QMessageBox box(this);
            box.setWindowTitle("Autosave Found");
            box.setText("An autosaved project differs from the project files.");
            box.setInformativeText("Do you want to use the autosaved data?");
            auto* useButton = box.addButton("Use Autosave", QMessageBox::AcceptRole);
            auto* discardButton = box.addButton("Discard Autosave", QMessageBox::DestructiveRole);
            auto* exportButton = box.addButton("Export Autosave...", QMessageBox::ActionRole);
            box.addButton(QMessageBox::Cancel);
            box.exec();
            if (box.clickedButton() == useButton) {
                rpPath = autoRp;
                cromcPath = autoCromc;
            } else if (box.clickedButton() == discardButton) {
                QFile::remove(autoRp);
                QFile::remove(autoCromc);
            } else if (box.clickedButton() == exportButton) {
                const QString destDir = QFileDialog::getExistingDirectory(
                    this, "Export Autosave", dir);
                if (!destDir.isEmpty()) {
                    QFile::copy(autoRp, QDir(destDir).filePath(autoBase + ".cRP"));
                    QFile::copy(autoCromc, QDir(destDir).filePath(autoBase + ".cROMc"));
                    QFile::remove(autoRp);
                    QFile::remove(autoCromc);
                }
            } else {
                return;
            }
        }
        LegacyProject legacy;
        std::string error;
        QString detail;
        if (!EnsureConcentrateExists(cromPath, cromcPath, &detail)) {
            statusBar()->showMessage(QString("Open failed: %1").arg(detail), 5000);
            return;
        }
        if (!LoadLegacyProjectFromConcentrate(cromcPath.toStdString(),
                                              rpPath.toStdString(),
                                              legacy,
                                              &error,
                                              true)) {
            statusBar()->showMessage(QString("Open failed: %1").arg(QString::fromStdString(error)), 5000);
            return;
        }
        if (!rpPath.isEmpty() && !legacy.frames.empty() && !legacy.frame_refs.empty()) {
            QString refError;
            const int refWidth = legacy.frame_width > 0 ? legacy.frame_width : kDefaultFrameWidth;
            const int refHeight = legacy.frame_height > 0 ? legacy.frame_height : kDefaultFrameHeight;
            if (!LoadReferenceFramesFromRp(rpPath,
                                           refWidth,
                                           refHeight,
                                           static_cast<int>(legacy.frames.size()),
                                           legacy.frame_refs,
                                           &refError)) {
                statusBar()->showMessage(QString("Open warning: %1").arg(refError), 5000);
            }
        }
        m_hasLegacyRoundTrip = true;
        m_legacyRoundTrip = LegacyRoundTripData{};
        m_legacyRoundTrip.name = legacy.name;
        m_legacyRoundTrip.frame_width_x = legacy.frame_width_x;
        m_legacyRoundTrip.frame_height_x = legacy.frame_height_x;
        m_legacyRoundTrip.hash_codes = legacy.hash_codes;
        m_legacyRoundTrip.active_frames = legacy.active_frames;
        m_legacyRoundTrip.trigger_ids = legacy.trigger_ids;
        m_legacyRoundTrip.dynashadow_dir = legacy.dynashadow_dir;
        m_legacyRoundTrip.dynashadow_col = legacy.dynashadow_col;
        m_legacyRoundTrip.dynashadow_dir_x = legacy.dynashadow_dir_x;
        m_legacyRoundTrip.dynashadow_col_x = legacy.dynashadow_col_x;
        m_legacyRoundTrip.active_col_sets = legacy.active_col_sets;
        m_legacyRoundTrip.mask_names = legacy.mask_names;
        m_legacyRoundTrip.frame_comp_mask_ids = legacy.frame_comp_mask_ids;
        m_legacyRoundTrip.frame_shape_comp_modes = legacy.frame_shape_comp_modes;
        m_legacyRoundTrip.draw_col_mode = legacy.draw_col_mode;
        m_legacyRoundTrip.draw_mode = legacy.draw_mode;
        m_legacyRoundTrip.mask_sel_mode = legacy.mask_sel_mode;
        m_legacyRoundTrip.fill_mode = legacy.fill_mode;
        m_legacyRoundTrip.edit_colors = legacy.edit_colors;
        m_legacyRoundTrip.n_image_pos_saves = legacy.n_image_pos_saves;
        m_legacyRoundTrip.image_pos_names = legacy.image_pos_names;
        m_legacyRoundTrip.image_pos_data = legacy.image_pos_data;
        m_legacyRoundTrip.is_imported = legacy.is_imported;
        m_legacyRoundTrip.time_elapsed = legacy.time_elapsed;
        m_legacyRoundTrip.is_pup_pack = legacy.is_pup_pack;
        m_legacyRoundTrip.pup_pack = legacy.pup_pack;
        m_legacyRoundTrip.preview_reduced_palette = legacy.preview_reduced_palette;

        m_imageStore->clear();
        m_frameStore->clear();
        m_spriteStore->clear();
        m_backgroundStore->clear();
        m_frameDurations.clear();
        m_frameTriggerIds.clear();
        m_spriteNames.clear();
        m_spriteColored.clear();
        m_spriteColoredX.clear();
        m_spriteOriginals.clear();
        m_spriteMasksX.clear();
        m_spriteDynamicMasks.clear();
        m_spriteDynamicMasksX.clear();
        m_spriteDynamicColors.clear();
        m_spriteDynamicColorsX.clear();
        m_spriteExtraFlags.clear();
        m_spriteShapeModes.clear();
        m_spriteDetAreas.clear();
        m_spriteDetDwords.clear();
        m_spriteDetDwordPos.clear();
        m_frameSpriteAssignments.clear();
        m_frameSpriteBBoxes.clear();
        m_spriteColFromFrame.clear();
        m_spriteRects.clear();
        m_spriteRectMirror.clear();
        m_sectionStarts.clear();
        m_sectionNames.clear();
        m_frameRefs.clear();
        m_frameDynamicColors.clear();
        m_compMasks.clear();
        m_frameCompMaskIds.clear();
        m_frameDynamicMaskMaps.clear();
        m_frameDynamicMaskMapsX.clear();
        m_frameShapeCompModes.clear();
        m_frameRotations.clear();
        m_frameRotationsX.clear();
        m_frameRotationsLocal.clear();
        m_frameRotationsLocalX.clear();
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
        if (!setupSerumData(cromcPath, legacy, &error)) {
            statusBar()->showMessage(QString("Open failed: %1").arg(QString::fromStdString(error)), 5000);
            return;
        }
        configureFrameStoreAdapter();
        configureSpriteStoreAdapter();
        configureBackgroundStoreAdapter();
        if (!m_serumDataLoaded) {
            for (const auto& frame : legacy.frames) {
                m_frameStore->add(frame);
            }
        } else if (m_frameStore) {
            m_frameStore->setCount(static_cast<int>(legacy.frames.size()));
        }
        if (!m_serumDataLoaded) {
            for (const auto& sprite : legacy.sprites) {
                m_spriteStore->add(sprite);
            }
        }
        m_frameDurations = legacy.frame_durations;
        m_frameTriggerIds = legacy.trigger_ids;
        if (m_frameTriggerIds.size() < legacy.frames.size()) {
            m_frameTriggerIds.resize(legacy.frames.size(), 0xffffffffu);
        } else if (m_frameTriggerIds.size() > legacy.frames.size()) {
            m_frameTriggerIds.resize(legacy.frames.size());
        }
        if (m_hasLegacyRoundTrip) {
            m_legacyRoundTrip.trigger_ids = m_frameTriggerIds;
        }
        m_spriteNames = legacy.sprite_labels;
        if (!m_serumDataLoaded) {
            m_spriteColored = legacy.sprite_colored;
        } else {
            m_spriteColored.assign(legacy.sprites.size(), cv::Mat());
        }
        m_spriteColoredX = legacy.sprite_colored_x;
        if (!m_serumDataLoaded) {
            m_spriteOriginals = legacy.sprite_originals;
        } else {
            m_spriteOriginals.assign(legacy.sprites.size(), cv::Mat());
            const std::size_t serumCount = m_serumData.spriteoriginal.elementCount();
            const std::size_t legacyCount = m_serumData.spritedescriptionso.elementCount();
            const std::size_t serumScanCount = (serumCount > 0) ? serumCount : legacyCount;
            const std::size_t legacyScanCount = (legacyCount > 0) ? legacyCount : serumCount;
            for (std::size_t i = 0; i < legacy.sprite_originals.size(); ++i) {
                if (legacy.sprite_originals[i].empty()) {
                    continue;
                }
                const uint8_t* serum = serumSpriteOriginalData(static_cast<uint32_t>(i));
                const bool serumHasOpaque = SpriteHasNonTransparentPixel(serum, serumScanCount);
                const bool legacyHasOpaque =
                    SpriteHasNonTransparentPixel(legacy.sprite_originals[i].data, legacyScanCount);
                if (!serumHasOpaque && legacyHasOpaque) {
                    m_spriteOriginals[i] = legacy.sprite_originals[i];
                }
            }
        }
        m_spriteMasksX = legacy.sprite_masks_x;
        if (!m_serumDataLoaded) {
            m_spriteDynamicMasks = legacy.sprite_dynamic_masks;
        } else {
            m_spriteDynamicMasks.assign(legacy.sprites.size(), cv::Mat());
        }
        m_spriteDynamicMasksX = legacy.sprite_dynamic_masks_x;
        if (!m_serumDataLoaded) {
            m_spriteDynamicColors = legacy.sprite_dynamic_colors;
        } else {
            m_spriteDynamicColors.assign(legacy.sprites.size(), std::vector<uint16_t>());
        }
        m_spriteDynamicColorsX = legacy.sprite_dynamic_colors_x;
        m_spriteExtraFlags = legacy.sprite_extra_flags;
        m_spriteShapeModes = legacy.sprite_shape_modes;
        m_spriteDetAreas = legacy.sprite_det_areas;
        m_spriteDetDwords = legacy.sprite_det_dwords;
        m_spriteDetDwordPos = legacy.sprite_det_dword_pos;
        m_frameSpriteAssignments = legacy.frame_sprites;
        m_frameSpriteBBoxes = legacy.frame_sprite_bboxes;
        m_spriteColFromFrame = legacy.sprite_col_from_frame;
        m_spriteRects = legacy.sprite_rects;
        m_spriteRectMirror = legacy.sprite_rect_mirror;
        m_sectionStarts = legacy.section_firsts;
        m_sectionNames = legacy.section_names;
        m_frameRefs = legacy.frame_refs;
        m_frameCompMaskIds = legacy.frame_comp_mask_ids;
        m_frameShapeCompModes = legacy.frame_shape_comp_modes;
        m_frameExtraFlags = legacy.frame_extra_flags;
        m_backgroundExtraFlags = legacy.background_extra_flags;
        m_frameBackgroundIds = legacy.background_ids;
        if (m_serumDataLoaded) {
            const std::size_t frameCount = legacy.frames.size();
            const std::size_t totalSlots = frameCount * MAX_SPRITES_PER_FRAME;
            const std::size_t totalBBoxes = totalSlots * 4;
            m_frameSpriteAssignments.assign(totalSlots, 255);
            m_frameSpriteBBoxes.assign(totalBBoxes, 0);
            for (std::size_t i = 0; i < frameCount; ++i) {
                const uint32_t idx = static_cast<uint32_t>(i);
                const std::size_t base = i * MAX_SPRITES_PER_FRAME;
                const std::size_t bboxBase = base * 4;
                const uint8_t* spriteData = m_serumData.framesprites.hasData(idx)
                    ? m_serumData.framesprites[idx]
                    : nullptr;
                if (spriteData) {
                    std::memcpy(m_frameSpriteAssignments.data() + base,
                                spriteData,
                                MAX_SPRITES_PER_FRAME);
                }
                const uint16_t* bboxData = m_serumData.framespriteBB.hasData(idx)
                    ? m_serumData.framespriteBB[idx]
                    : nullptr;
                if (bboxData) {
                    std::memcpy(m_frameSpriteBBoxes.data() + bboxBase,
                                bboxData,
                                MAX_SPRITES_PER_FRAME * 4 * sizeof(uint16_t));
                }
            }
            if (m_frameSpriteZoneFlags.size() != totalSlots) {
                m_frameSpriteZoneFlags.assign(totalSlots, 0);
            }
            for (std::size_t i = 0; i < totalSlots; ++i) {
                if (m_frameSpriteAssignments[i] != 255) {
                    m_frameSpriteZoneFlags[i] = 1;
                }
            }
            const std::size_t spriteCount = legacy.sprites.size();
            m_spriteColoredX.assign(spriteCount, cv::Mat());
            m_spriteMasksX.assign(spriteCount, cv::Mat());
            m_spriteDynamicMasksX.assign(spriteCount, cv::Mat());
            m_spriteDynamicColorsX.assign(spriteCount, std::vector<uint16_t>());
        }
        if (!m_serumDataLoaded) {
            m_frameDynamicColors = legacy.frame_dynamic_colors;
            m_compMasks = legacy.comp_masks;
            m_frameDynamicMaskMaps = legacy.frame_dynamic_mask_maps;
            m_frameDynamicMaskMapsX = legacy.frame_dynamic_mask_maps_x;
            m_frameRotations = legacy.frame_rotations;
            m_frameRotationsX = legacy.frame_rotations_x;
            m_frameExtraFrames = legacy.frames_x;
            m_frameBackgroundMasks = legacy.background_masks;
            m_frameBackgroundMasksX = legacy.background_masks_x;
            m_backgroundFramesX = legacy.background_frames_x;
        } else {
            const std::size_t frameCount = legacy.frames.size();
            m_frameDynamicColors.assign(frameCount, std::vector<uint16_t>());
            m_compMasks.assign(MAX_MASKS, cv::Mat());
            m_frameDynamicMaskMaps.assign(frameCount, cv::Mat());
            m_frameDynamicMaskMapsX.assign(frameCount, cv::Mat());
            m_frameExtraFrames.assign(frameCount, cv::Mat());
            m_frameBackgroundMasks = legacy.background_masks;
            m_frameBackgroundMasksX = legacy.background_masks_x;
            m_backgroundFramesX.assign(static_cast<std::size_t>(m_backgroundStore->count()), cv::Mat());
        }
        if (!m_serumDataLoaded) {
            for (const auto& bg : legacy.background_frames) {
                m_backgroundStore->add(bg);
            }
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
        m_state->openProject(projectCromcPath);
            m_state->setFramesAndSprites(frames, sprites);
        }
        resetUndoStacks();
        m_projectDirty = false;
        updateWindowTitle();
        m_projectLabel->setText(cromcPath);
        updateProjectLabelHeight();
        refreshRecentMenu();
        refreshImageList();
        refreshCounts();
        refreshFrameSpriteLists();
        resetNavigationHistory();
        const bool wantSelectFirst = !legacy.frames.empty();
        populateBookmarks(legacy.section_firsts, legacy.section_names);
        statusBar()->showMessage(QString("Open legacy: %1").arg(cromcPath), 5000);
        m_state->setRecentFiles(m_state->recentFiles());
        persistRecentFiles();
        QTimer::singleShot(0, this, [this]() {
            refreshAllPreviews();
            refreshFrameSpriteLists();
            if (m_framesCanvas) {
                m_framesCanvas->canvas()->requestFitOnResize(true);
            }
        });
        if (wantSelectFirst && m_framesList && m_framesList->count() > 0) {
            QTimer::singleShot(0, this, [this]() {
                if (m_framesList && m_framesList->count() > 0) {
                    m_framesList->setCurrentRow(0);
                }
            });
        }
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
    m_frameCompMaskIds.clear();
    m_frameDynamicMaskMaps.clear();
    m_frameDynamicMaskMapsX.clear();
    m_frameShapeCompModes.clear();
    m_frameRotations.clear();
    m_frameRotationsX.clear();
    m_frameRotationsLocal.clear();
    m_frameRotationsLocalX.clear();
    m_frameExtraFrames.clear();
    m_serumData.Clear();
    m_serumDataLoaded = false;
    configureFrameStoreAdapter();
    configureSpriteStoreAdapter();
    configureBackgroundStoreAdapter();
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
    m_hasLegacyRoundTrip = false;
    m_legacyRoundTrip = LegacyRoundTripData{};
    if (LoadProjectJson(*m_state, filename, &error)) {
        statusBar()->showMessage(QString("Open: %1").arg(filename), 5000);
        persistRecentFiles();
        QTimer::singleShot(0, this, [this]() {
            refreshAllPreviews();
            refreshFrameSpriteLists();
        });
    } else {
        statusBar()->showMessage(QString("Open failed: %1").arg(error), 5000);
    }
}

bool MainWindow::saveProjectToPath(const QString& filename)
{
    if (filename.isEmpty()) {
        return false;
    }
    logLine(QString("Save project: %1").arg(filename));
    QFileInfo info(filename);
    QString suffix = info.suffix().toLower();
    QString target = filename;
    if (suffix.isEmpty()) {
        target = filename + ".cRP";
        suffix = "crp";
    }
    if (suffix == "crom" || suffix == "crp" || suffix == "cromc") {
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
    QFileInfo info(filename);
    const QString suffix = info.suffix().toLower();
    const QString dir = info.absolutePath();
    const QString base = info.completeBaseName();
    QString rpPath = dir + "/" + base + ".cRP";
    if (suffix == "crp") {
        rpPath = filename;
    }
    const QString cromcPath = dir + "/" + base + ".cROMc";
    return saveLegacyProjectToPaths(rpPath, cromcPath, base, true, true, false);
}

bool MainWindow::saveLegacyProjectToPaths(const QString& rpPath,
                                          const QString& cromcPath,
                                          const QString& baseName,
                                          bool updateState,
                                          bool showErrors,
                                          bool preferSerumMasks)
{
    const int frameCount = m_frameStore ? m_frameStore->count() : 0;
    if (frameCount <= 0) {
        if (showErrors) {
            statusBar()->showMessage(frameCount < 0
                                         ? "Save failed: frames not ready"
                                         : "Save failed: no frames",
                                     5000);
        }
        return false;
    }
    if (m_frameStore) {
        m_frameStore->flush();
    }
    ensureMaskDataSize();
    if (m_serumDataLoaded) {
        std::size_t missingRefs = 0;
        for (const auto& ref : m_frameRefs) {
            if (ref.empty()) {
                ++missingRefs;
            }
        }
        if (missingRefs > 0) {
            const QString message = QString("Save aborted: missing %1 original frame(s) from .cRP. "
                                            "Open the project with its .cRP to preserve originals.")
                                        .arg(missingRefs);
            if (showErrors) {
                statusBar()->showMessage(message, 7000);
            }
            logLine(message);
            return false;
        }
    }
    LegacyProject project = buildLegacyProject(baseName, preferSerumMasks);
    std::string error;
    if (!SaveLegacyProjectRp(rpPath.toStdString(), project, &error)) {
        if (showErrors) {
            statusBar()->showMessage(QString("Save failed: %1").arg(QString::fromStdString(error)), 5000);
        }
        return false;
    }
    std::string cromcError;
    if (!SaveConcentrateProjectWithSeed(cromcPath.toStdString(),
                                        project,
                                        m_serumDataLoaded ? &m_serumData : nullptr,
                                        &cromcError)) {
        if (showErrors) {
            statusBar()->showMessage(QString("Saved, but .cROMc failed: %1").arg(QString::fromStdString(cromcError)),
                                     5000);
        }
    }
    if (updateState) {
        m_state->saveProject(cromcPath);
        m_projectDirty = false;
    }
    return true;
}

bool MainWindow::autosaveProject(bool showProgress)
{
    if (!m_autosaveEnabled || m_projectDir.isEmpty() || m_projectBaseName.isEmpty()) {
        logLine("Autosave skipped: disabled or missing project path");
        return false;
    }
    if (m_frameHasStart || m_spriteHasStart) {
        logLine("Autosave skipped: drawing in progress");
        return false;
    }
    if (!m_projectDirty) {
        logLine("Autosave skipped: project clean");
        return false;
    }
    if (m_isLoadingProject) {
        logLine("Autosave skipped: loading project");
        return false;
    }
    if (!m_frameStore || m_frameStore->count() <= 0) {
        logLine("Autosave skipped: no frames");
        return false;
    }
    const QString autoBase = m_projectBaseName + ".autosave";
    const QString autoRp = QDir(m_projectDir).filePath(autoBase + ".cRP");
    const QString autoCromc = QDir(m_projectDir).filePath(autoBase + ".cROMc");
    std::unique_ptr<QProgressDialog> progress;
    if (showProgress) {
        progress.reset(new QProgressDialog("Autosaving...", QString(), 0, 0, this));
        progress->setWindowModality(Qt::ApplicationModal);
        progress->setCancelButton(nullptr);
        progress->setMinimumDuration(0);
        progress->show();
        QApplication::processEvents();
    }
    const bool ok = saveLegacyProjectToPaths(autoRp, autoCromc, m_projectBaseName, false, false, false);
    if (progress) {
        progress->close();
    }
    if (!QFileInfo::exists(autoCromc)) {
        logLine(QString("Autosave failed: missing %1").arg(autoCromc));
        return false;
    }
    if (ok) {
        logLine(QString("Autosave: %1").arg(autoCromc));
    }
    return ok;
}

void MainWindow::updateAutosaveTimer()
{
    if (!m_autosaveTimer) {
        return;
    }
    if (!m_autosaveEnabled) {
        m_autosaveTimer->stop();
        return;
    }
    const int intervalMs = std::max(1, m_autosaveIntervalMinutes) * 60 * 1000;
    m_autosaveTimer->start(intervalMs);
}

void MainWindow::setSerumRuntimePathsForPlayback(bool useAutosave)
{
    if (m_projectDir.isEmpty() || m_projectBaseName.isEmpty()) {
        return;
    }
    if (useAutosave) {
        m_serumRuntimeAltDir = m_projectDir;
        m_serumRuntimeRomName = m_projectBaseName + ".autosave";
    } else {
        m_serumRuntimeAltDir = m_projectDir;
        m_serumRuntimeRomName = m_projectBaseName;
    }
}

bool MainWindow::autosaveFilesDiffer(const QString& rpPath,
                                     const QString& cromcPath,
                                     const QString& autoRp,
                                     const QString& autoCromc) const
{
    const QFileInfo autoRpInfo(autoRp);
    const QFileInfo autoCromcInfo(autoCromc);
    if (!autoRpInfo.exists() || !autoCromcInfo.exists()) {
        return false;
    }
    const QFileInfo rpInfo(rpPath);
    const QFileInfo cromcInfo(cromcPath);
    if (!rpInfo.exists() || !cromcInfo.exists()) {
        return true;
    }
    if (rpInfo.size() != autoRpInfo.size() || cromcInfo.size() != autoCromcInfo.size()) {
        return true;
    }
    if (autoRpInfo.lastModified() > rpInfo.lastModified() ||
        autoCromcInfo.lastModified() > cromcInfo.lastModified()) {
        return true;
    }
    return false;
}

LegacyProject MainWindow::buildLegacyProject(const QString& baseName, bool preferSerumMasks) const
{
    LegacyProject project;
    project.name = baseName.toStdString();
    if (m_hasLegacyRoundTrip) {
        if (!m_legacyRoundTrip.name.empty()) {
            project.name = m_legacyRoundTrip.name;
        }
        project.frame_width_x = m_legacyRoundTrip.frame_width_x;
        project.frame_height_x = m_legacyRoundTrip.frame_height_x;
        project.hash_codes = m_legacyRoundTrip.hash_codes;
        project.active_frames = m_legacyRoundTrip.active_frames;
        project.trigger_ids = m_legacyRoundTrip.trigger_ids;
        project.dynashadow_dir = m_legacyRoundTrip.dynashadow_dir;
        project.dynashadow_col = m_legacyRoundTrip.dynashadow_col;
        project.dynashadow_dir_x = m_legacyRoundTrip.dynashadow_dir_x;
        project.dynashadow_col_x = m_legacyRoundTrip.dynashadow_col_x;
        project.active_col_sets = m_legacyRoundTrip.active_col_sets;
        project.mask_names = m_legacyRoundTrip.mask_names;
        project.draw_col_mode = m_legacyRoundTrip.draw_col_mode;
        project.draw_mode = m_legacyRoundTrip.draw_mode;
        project.mask_sel_mode = m_legacyRoundTrip.mask_sel_mode;
        project.fill_mode = m_legacyRoundTrip.fill_mode;
        project.edit_colors = m_legacyRoundTrip.edit_colors;
        project.n_image_pos_saves = m_legacyRoundTrip.n_image_pos_saves;
        project.image_pos_names = m_legacyRoundTrip.image_pos_names;
        project.image_pos_data = m_legacyRoundTrip.image_pos_data;
        project.is_imported = m_legacyRoundTrip.is_imported;
        project.time_elapsed = m_legacyRoundTrip.time_elapsed;
        project.is_pup_pack = m_legacyRoundTrip.is_pup_pack;
        project.pup_pack = m_legacyRoundTrip.pup_pack;
        project.preview_reduced_palette = m_legacyRoundTrip.preview_reduced_palette;
    }

    const int frameCount = m_frameStore->count();
    if (!m_frameTriggerIds.empty()) {
        project.trigger_ids = m_frameTriggerIds;
    }
    if (m_serumDataLoaded && frameCount > 0) {
        project.hash_codes.assign(static_cast<std::size_t>(frameCount), 0);
        project.active_frames.assign(static_cast<std::size_t>(frameCount), 0);
        for (int i = 0; i < frameCount; ++i) {
            const uint32_t index = static_cast<uint32_t>(i);
            bool haveHash = false;
            bool haveActive = false;
            if (m_serumData.hashcodes.hasData(index)) {
                project.hash_codes[static_cast<std::size_t>(i)] = *m_serumData.hashcodes[index];
                haveHash = true;
            }
            if (m_serumData.activeframes.hasData(index)) {
                project.active_frames[static_cast<std::size_t>(i)] = *m_serumData.activeframes[index];
                haveActive = true;
            }
            if (m_hasLegacyRoundTrip) {
                if (!haveHash && i < static_cast<int>(m_legacyRoundTrip.hash_codes.size())) {
                    project.hash_codes[static_cast<std::size_t>(i)] = m_legacyRoundTrip.hash_codes[static_cast<std::size_t>(i)];
                }
                if (!haveActive && i < static_cast<int>(m_legacyRoundTrip.active_frames.size())) {
                    project.active_frames[static_cast<std::size_t>(i)] = m_legacyRoundTrip.active_frames[static_cast<std::size_t>(i)];
                }
            }
        }
    } else if (m_hasLegacyRoundTrip) {
        project.hash_codes.resize(static_cast<std::size_t>(frameCount), 0);
        project.active_frames.resize(static_cast<std::size_t>(frameCount), 0);
    }
    project.trigger_ids.resize(static_cast<std::size_t>(frameCount), 0xffffffffu);
    project.frames.resize(static_cast<std::size_t>(frameCount));
    cv::Size baseSize(kDefaultFrameWidth, kDefaultFrameHeight);
    for (int i = 0; i < frameCount; ++i) {
        const cv::Mat frame = m_frameStore->loadCopy(i);
        if (!frame.empty()) {
            baseSize = frame.size();
            break;
        }
    }
    for (int i = 0; i < frameCount; ++i) {
        const cv::Mat frame = m_frameStore->loadCopy(i);
        if (!frame.empty()) {
            project.frames[static_cast<std::size_t>(i)] = frame;
            continue;
        }
        project.frames[static_cast<std::size_t>(i)] =
            cv::Mat(baseSize, CV_8UC3, cv::Scalar(0, 0, 0));
    }

    const int spriteCount = m_spriteStore->count();
    project.sprites.resize(static_cast<std::size_t>(spriteCount));
    for (int i = 0; i < spriteCount; ++i) {
        const cv::Mat sprite = m_spriteStore->loadCopy(i);
        if (!sprite.empty()) {
            project.sprites[static_cast<std::size_t>(i)] = sprite;
        }
    }
    if (!m_spriteColored.empty()) {
        project.sprite_colored = m_spriteColored;
    } else {
        project.sprite_colored = project.sprites;
    }
    project.sprite_colored_x = m_spriteColoredX;
    project.sprite_originals = m_spriteOriginals;
    project.sprite_masks_x = m_spriteMasksX;
    project.sprite_dynamic_masks = m_spriteDynamicMasks;
    project.sprite_dynamic_masks_x = m_spriteDynamicMasksX;
    project.sprite_dynamic_colors = m_spriteDynamicColors;
    project.sprite_dynamic_colors_x = m_spriteDynamicColorsX;
    if (m_serumDataLoaded) {
        const std::size_t spriteCountSize = static_cast<std::size_t>(spriteCount);
        if (project.sprite_originals.size() < spriteCountSize) {
            project.sprite_originals.resize(spriteCountSize);
        }
        if (project.sprite_dynamic_masks.size() < spriteCountSize) {
            project.sprite_dynamic_masks.resize(spriteCountSize);
        }
        if (project.sprite_dynamic_colors.size() < spriteCountSize) {
            project.sprite_dynamic_colors.resize(spriteCountSize);
        }
        if (project.sprite_colored_x.size() < spriteCountSize) {
            project.sprite_colored_x.resize(spriteCountSize);
        }
        if (project.sprite_masks_x.size() < spriteCountSize) {
            project.sprite_masks_x.resize(spriteCountSize);
        }
        if (project.sprite_dynamic_masks_x.size() < spriteCountSize) {
            project.sprite_dynamic_masks_x.resize(spriteCountSize);
        }
        if (project.sprite_dynamic_colors_x.size() < spriteCountSize) {
            project.sprite_dynamic_colors_x.resize(spriteCountSize);
        }
        const int stride = serumDynamicStride();
        const std::size_t colorsPerSprite =
            static_cast<std::size_t>(MAX_DYNA_SETS_PER_SPRITE) * static_cast<std::size_t>(stride);
        for (int i = 0; i < spriteCount; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i);
            if (project.sprite_originals[idx].empty() &&
                i < static_cast<int>(m_serumData.nsprites) &&
                m_serumData.spriteoriginal.hasData(static_cast<uint32_t>(i)) &&
                m_serumData.fwidth > 0 && m_serumData.fheight > 0) {
                const uint8_t* data = m_serumData.spriteoriginal[static_cast<uint32_t>(i)];
                if (data) {
                    cv::Mat original(static_cast<int>(m_serumData.fheight),
                                     static_cast<int>(m_serumData.fwidth),
                                     CV_8UC1,
                                     cv::Scalar(255));
                    std::memcpy(original.data,
                                data,
                                static_cast<std::size_t>(m_serumData.fwidth) * m_serumData.fheight);
                    project.sprite_originals[idx] = original;
                }
            }
            if (project.sprite_dynamic_masks[idx].empty() &&
                i < static_cast<int>(m_serumData.nsprites) &&
                m_serumData.dynaspritemasks.hasData(static_cast<uint32_t>(i)) &&
                m_serumData.fwidth > 0 && m_serumData.fheight > 0) {
                const uint8_t* data = m_serumData.dynaspritemasks[static_cast<uint32_t>(i)];
                if (data) {
                    cv::Mat mask(static_cast<int>(m_serumData.fheight),
                                 static_cast<int>(m_serumData.fwidth),
                                 CV_8UC1,
                                 cv::Scalar(255));
                    std::memcpy(mask.data,
                                data,
                                static_cast<std::size_t>(m_serumData.fwidth) * m_serumData.fheight);
                    project.sprite_dynamic_masks[idx] = mask;
                }
            }
            if (project.sprite_dynamic_colors[idx].empty() &&
                i < static_cast<int>(m_serumData.nsprites) &&
                m_serumData.dynasprite4cols.hasData(static_cast<uint32_t>(i)) &&
                colorsPerSprite > 0) {
                const uint16_t* data = m_serumData.dynasprite4cols[static_cast<uint32_t>(i)];
                if (data) {
                    project.sprite_dynamic_colors[idx].assign(data, data + colorsPerSprite);
                }
            }
            if (project.sprite_colored_x[idx].empty() &&
                i < static_cast<int>(m_serumData.nsprites) &&
                m_serumData.spritecolored_extra.hasData(static_cast<uint32_t>(i))) {
                const uint16_t* data = m_serumData.spritecolored_extra[static_cast<uint32_t>(i)];
                if (data) {
                    project.sprite_colored_x[idx] = ConvertRgb565ToBgrMat(data,
                                                                         MAX_SPRITE_WIDTH,
                                                                         MAX_SPRITE_HEIGHT);
                }
            }
            if (project.sprite_masks_x[idx].empty() &&
                i < static_cast<int>(m_serumData.nsprites) &&
                m_serumData.spritemask_extra.hasData(static_cast<uint32_t>(i))) {
                const uint8_t* data = m_serumData.spritemask_extra[static_cast<uint32_t>(i)];
                if (data) {
                    cv::Mat mask(MAX_SPRITE_HEIGHT, MAX_SPRITE_WIDTH, CV_8UC1, cv::Scalar(255));
                    std::memcpy(mask.data, data,
                                static_cast<std::size_t>(MAX_SPRITE_WIDTH) * MAX_SPRITE_HEIGHT);
                    project.sprite_masks_x[idx] = mask;
                }
            }
            if (project.sprite_dynamic_masks_x[idx].empty() &&
                i < static_cast<int>(m_serumData.nsprites) &&
                m_serumData.dynaspritemasks_extra.hasData(static_cast<uint32_t>(i))) {
                const uint8_t* data = m_serumData.dynaspritemasks_extra[static_cast<uint32_t>(i)];
                if (data) {
                    cv::Mat mask(MAX_SPRITE_HEIGHT, MAX_SPRITE_WIDTH, CV_8UC1, cv::Scalar(255));
                    std::memcpy(mask.data, data,
                                static_cast<std::size_t>(MAX_SPRITE_WIDTH) * MAX_SPRITE_HEIGHT);
                    project.sprite_dynamic_masks_x[idx] = mask;
                }
            }
            if (project.sprite_dynamic_colors_x[idx].empty() &&
                i < static_cast<int>(m_serumData.nsprites) &&
                m_serumData.dynasprite4cols_extra.hasData(static_cast<uint32_t>(i)) &&
                colorsPerSprite > 0) {
                const uint16_t* data = m_serumData.dynasprite4cols_extra[static_cast<uint32_t>(i)];
                if (data) {
                    project.sprite_dynamic_colors_x[idx].assign(data, data + colorsPerSprite);
                }
            }
        }
    }
    project.sprite_extra_flags = m_spriteExtraFlags;
    project.sprite_shape_modes = m_spriteShapeModes;
    project.sprite_det_areas = m_spriteDetAreas;
    project.sprite_det_dwords = m_spriteDetDwords;
    project.sprite_det_dword_pos = m_spriteDetDwordPos;
    project.frame_sprites = m_frameSpriteAssignments;
    project.frame_sprite_bboxes = m_frameSpriteBBoxes;
    project.sprite_col_from_frame = m_spriteColFromFrame;
    project.sprite_rects = m_spriteRects;
    project.sprite_rect_mirror = m_spriteRectMirror;
    if (m_serumDataLoaded && frameCount > 0) {
        const std::size_t totalSlots = static_cast<std::size_t>(frameCount) * MAX_SPRITES_PER_FRAME;
        const std::size_t totalBBoxes = totalSlots * 4;
        if (project.frame_sprites.size() != totalSlots) {
            project.frame_sprites.assign(totalSlots, 255);
        }
        if (project.frame_sprite_bboxes.size() != totalBBoxes) {
            project.frame_sprite_bboxes.assign(totalBBoxes, 0);
        }
        auto allDefaultSprites = [&]() {
            for (uint8_t v : project.frame_sprites) {
                if (v != 255) {
                    return false;
                }
            }
            return true;
        };
        const bool needSprites = allDefaultSprites();
        int spriteFramesCopied = 0;
        int spriteBBoxesCopied = 0;
        int serumFramesWithSprites = 0;
        for (int i = 0; i < frameCount; ++i) {
            const uint32_t idx = static_cast<uint32_t>(i);
            const std::size_t base = static_cast<std::size_t>(i) * MAX_SPRITES_PER_FRAME;
            const std::size_t bboxBase = base * 4;
            const uint8_t* spriteData = m_serumData.framesprites[idx];
            bool hasSprite = false;
            for (int slot = 0; slot < MAX_SPRITES_PER_FRAME; ++slot) {
                if (spriteData && spriteData[slot] != 255) {
                    hasSprite = true;
                    break;
                }
            }
            if (hasSprite) {
                ++serumFramesWithSprites;
            }
            if (needSprites && hasSprite && spriteData) {
                std::memcpy(project.frame_sprites.data() + base, spriteData, MAX_SPRITES_PER_FRAME);
                ++spriteFramesCopied;
            }
            const uint16_t* bboxData = m_serumData.framespriteBB[idx];
            if (bboxData && (hasSprite || m_serumData.framespriteBB.hasData(idx))) {
                std::memcpy(project.frame_sprite_bboxes.data() + bboxBase,
                            bboxData,
                            MAX_SPRITES_PER_FRAME * 4 * sizeof(uint16_t));
                ++spriteBBoxesCopied;
            }
        }
        const_cast<MainWindow*>(this)->logLine(
            QString("Save: sprites from serum frames=%1 bboxes=%2 serumFramesWithSprites=%3")
                .arg(spriteFramesCopied)
                .arg(spriteBBoxesCopied)
                .arg(serumFramesWithSprites));
    }

    project.frame_durations.assign(static_cast<std::size_t>(frameCount), 30);
    for (int i = 0; i < frameCount && i < static_cast<int>(m_frameDurations.size()); ++i) {
        project.frame_durations[i] = m_frameDurations[i];
    }

    project.section_firsts = m_sectionStarts;
    project.section_names = m_sectionNames;
    project.frame_refs = m_frameRefs;
    project.no_colors = m_noColors > 0 ? m_noColors : 64;
    if (!preferSerumMasks) {
        project.comp_masks = m_compMasks;
        project.frame_comp_mask_ids = m_frameCompMaskIds;
        project.frame_shape_comp_modes = m_frameShapeCompModes;
        project.frame_dynamic_mask_maps = m_frameDynamicMaskMaps;
        project.frame_dynamic_mask_maps_x = m_frameDynamicMaskMapsX;
    } else {
        project.comp_masks.clear();
        project.frame_comp_mask_ids.clear();
        project.frame_shape_comp_modes.clear();
        project.frame_dynamic_mask_maps.clear();
        project.frame_dynamic_mask_maps_x.clear();
    }
    if (!m_serumDataLoaded) {
        project.frame_dynamic_colors = m_frameDynamicColors;
        project.frame_rotations = m_frameRotations;
        project.frame_rotations_x = m_frameRotationsX;
    } else {
        project.frame_dynamic_colors.assign(static_cast<std::size_t>(frameCount), {});
        const int stride = serumDynamicStride();
        const std::size_t size =
            static_cast<std::size_t>(MAX_DYNA_SETS_PER_FRAMEN) * static_cast<std::size_t>(stride);
        int dynColorsCopied = 0;
        for (int i = 0; i < frameCount; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i);
            bool useLocal = false;
            if (idx < m_frameDynamicColors.size() && !m_frameDynamicColors[idx].empty()) {
                const auto& local = m_frameDynamicColors[idx];
                for (uint16_t v : local) {
                    if (v != 0) {
                        useLocal = true;
                        break;
                    }
                }
            }
            if (useLocal) {
                project.frame_dynamic_colors[idx] = m_frameDynamicColors[idx];
                continue;
            }
            const uint16_t* data = frameDynamicColorsData(i, false, nullptr);
            if (data) {
                project.frame_dynamic_colors[idx].assign(data, data + size);
                ++dynColorsCopied;
            } else {
                project.frame_dynamic_colors[idx].assign(size, 0);
            }
        }
        const_cast<MainWindow*>(this)->logLine(
            QString("Save: dyn colors from serum=%1").arg(dynColorsCopied));
        const std::size_t blockSize = rotationBlockSize();
        project.frame_rotations.assign(static_cast<std::size_t>(frameCount) * blockSize, 0);
        project.frame_rotations_x.assign(static_cast<std::size_t>(frameCount) * blockSize, 0);
        for (int i = 0; i < frameCount; ++i) {
            const std::size_t offset = static_cast<std::size_t>(i) * blockSize;
            if (const uint16_t* data = rotationBlockForRead(i, false)) {
                std::copy_n(data, blockSize, project.frame_rotations.begin() + offset);
            }
            if (const uint16_t* dataHd = rotationBlockForRead(i, true)) {
                std::copy_n(dataHd, blockSize, project.frame_rotations_x.begin() + offset);
            }
        }
    }
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
    if (m_serumDataLoaded && frameCount > 0) {
        const int width = m_serumData.fwidth > 0 ? static_cast<int>(m_serumData.fwidth) : kDefaultFrameWidth;
        const int height = m_serumData.fheight > 0 ? static_cast<int>(m_serumData.fheight) : kDefaultFrameHeight;
        const int widthX = m_serumData.fwidth_extra > 0 ? static_cast<int>(m_serumData.fwidth_extra) : width;
        const int heightX = m_serumData.fheight_extra > 0 ? static_cast<int>(m_serumData.fheight_extra) : height;
        const std::size_t pixels = static_cast<std::size_t>(width) * height;
        const std::size_t pixelsX = static_cast<std::size_t>(widthX) * heightX;

        if (project.comp_masks.size() != MAX_MASKS) {
            project.comp_masks.resize(MAX_MASKS);
        }
        for (int i = 0; i < MAX_MASKS; ++i) {
            if (!m_serumData.compmasks.hasData(static_cast<uint32_t>(i))) {
                continue;
            }
            const uint8_t* data = m_serumData.compmasks[static_cast<uint32_t>(i)];
            if (!data || pixels == 0) {
                continue;
            }
            const std::size_t idx = static_cast<std::size_t>(i);
            const cv::Mat& existing = project.comp_masks[idx];
            bool shouldOverride = existing.empty();
            if (!shouldOverride && existing.type() == CV_8UC1) {
                if (cv::countNonZero(existing) == 0) {
                    cv::Mat serumMask(height, width, CV_8UC1, cv::Scalar(0));
                    std::memcpy(serumMask.data, data, pixels);
                    if (cv::countNonZero(serumMask) > 0) {
                        project.comp_masks[idx] = serumMask;
                        continue;
                    }
                }
                continue;
            }
            if (shouldOverride) {
                cv::Mat mask(height, width, CV_8UC1, cv::Scalar(0));
                std::memcpy(mask.data, data, pixels);
                project.comp_masks[idx] = mask;
            }
        }
        if (project.frame_comp_mask_ids.empty()) {
            project.frame_comp_mask_ids.assign(static_cast<std::size_t>(frameCount), 255);
            for (int i = 0; i < frameCount; ++i) {
                bool set = false;
                if (m_serumData.compmaskID.hasData(static_cast<uint32_t>(i))) {
                    project.frame_comp_mask_ids[static_cast<std::size_t>(i)] =
                        *m_serumData.compmaskID[static_cast<uint32_t>(i)];
                    set = true;
                }
                if (!set && m_hasLegacyRoundTrip &&
                    i < static_cast<int>(m_legacyRoundTrip.frame_comp_mask_ids.size())) {
                    project.frame_comp_mask_ids[static_cast<std::size_t>(i)] =
                        m_legacyRoundTrip.frame_comp_mask_ids[static_cast<std::size_t>(i)];
                }
            }
        }
        if (project.frame_shape_comp_modes.empty()) {
            project.frame_shape_comp_modes.assign(static_cast<std::size_t>(frameCount), 0);
            for (int i = 0; i < frameCount; ++i) {
                bool set = false;
                if (m_serumData.shapecompmode.hasData(static_cast<uint32_t>(i))) {
                    project.frame_shape_comp_modes[static_cast<std::size_t>(i)] =
                        *m_serumData.shapecompmode[static_cast<uint32_t>(i)];
                    set = true;
                }
                if (!set && m_hasLegacyRoundTrip &&
                    i < static_cast<int>(m_legacyRoundTrip.frame_shape_comp_modes.size())) {
                    project.frame_shape_comp_modes[static_cast<std::size_t>(i)] =
                        m_legacyRoundTrip.frame_shape_comp_modes[static_cast<std::size_t>(i)];
                }
            }
        }
        if (!project.hash_codes.empty() &&
            project.frame_comp_mask_ids.size() == static_cast<std::size_t>(frameCount) &&
            project.frame_shape_comp_modes.size() == static_cast<std::size_t>(frameCount) &&
            project.comp_masks.size() >= MAX_MASKS &&
            project.frame_refs.size() == static_cast<std::size_t>(frameCount)) {
            int hashComputed = 0;
            for (int i = 0; i < frameCount; ++i) {
                if (project.hash_codes[static_cast<std::size_t>(i)] != 0) {
                    continue;
                }
                const cv::Mat& ref = project.frame_refs[static_cast<std::size_t>(i)];
                if (ref.empty() || ref.type() != CV_8UC1) {
                    continue;
                }
                const uint8_t maskId =
                    project.frame_comp_mask_ids[static_cast<std::size_t>(i)];
                const uint8_t shape =
                    project.frame_shape_comp_modes[static_cast<std::size_t>(i)];
                const uint8_t* mask = nullptr;
                if (maskId < project.comp_masks.size()) {
                    const cv::Mat& maskMat = project.comp_masks[maskId];
                    if (!maskMat.empty() && maskMat.type() == CV_8UC1) {
                        mask = maskMat.data;
                    }
                }
                project.hash_codes[static_cast<std::size_t>(i)] =
                    Serum_CalcFrameHash(ref.data,
                                        mask,
                                        static_cast<uint32_t>(ref.total()),
                                        shape);
                if (project.hash_codes[static_cast<std::size_t>(i)] != 0) {
                    ++hashComputed;
                }
            }
            int hashZero = 0;
            for (const uint32_t h : project.hash_codes) {
                if (h == 0) {
                    ++hashZero;
                }
            }
            const_cast<MainWindow*>(this)->logLine(
                QString("Save: hashes computed=%1 zeros=%2")
                    .arg(hashComputed)
                    .arg(hashZero));
        }
        if (project.frame_dynamic_mask_maps.size() != static_cast<std::size_t>(frameCount)) {
            project.frame_dynamic_mask_maps.resize(static_cast<std::size_t>(frameCount));
        }
        int dynMaskCopied = 0;
        for (int i = 0; i < frameCount; ++i) {
            cv::Mat& map = project.frame_dynamic_mask_maps[static_cast<std::size_t>(i)];
            if (!m_serumData.dynamasks.hasData(static_cast<uint32_t>(i))) {
                continue;
            }
            const uint8_t* data = m_serumData.dynamasks[static_cast<uint32_t>(i)];
            if (data && pixels > 0) {
                bool shouldOverride = map.empty();
                if (!shouldOverride && map.type() == CV_8UC1) {
                    if (cv::countNonZero(map != 255) == 0) {
                        shouldOverride = true;
                    }
                }
                if (shouldOverride) {
                    map = cv::Mat(height, width, CV_8UC1, cv::Scalar(255));
                    std::memcpy(map.data, data, pixels);
                    ++dynMaskCopied;
                }
            }
        }
        const_cast<MainWindow*>(this)->logLine(
            QString("Save: dyn masks sd copied=%1").arg(dynMaskCopied));
        if (project.frame_dynamic_mask_maps_x.size() != static_cast<std::size_t>(frameCount)) {
            project.frame_dynamic_mask_maps_x.resize(static_cast<std::size_t>(frameCount));
        }
        int dynMaskXCopied = 0;
        for (int i = 0; i < frameCount; ++i) {
            if (!project.frame_extra_flags.empty() &&
                project.frame_extra_flags[static_cast<std::size_t>(i)] == 0) {
                continue;
            }
            cv::Mat& mapX = project.frame_dynamic_mask_maps_x[static_cast<std::size_t>(i)];
            if (!m_serumData.dynamasks_extra.hasData(static_cast<uint32_t>(i))) {
                continue;
            }
            const uint8_t* data = m_serumData.dynamasks_extra[static_cast<uint32_t>(i)];
            if (data && pixelsX > 0) {
                bool shouldOverride = mapX.empty();
                if (!shouldOverride && mapX.type() == CV_8UC1) {
                    if (cv::countNonZero(mapX != 255) == 0) {
                        shouldOverride = true;
                    }
                }
                if (shouldOverride) {
                    mapX = cv::Mat(heightX, widthX, CV_8UC1, cv::Scalar(255));
                    std::memcpy(mapX.data, data, pixelsX);
                    ++dynMaskXCopied;
                }
            }
        }
        const_cast<MainWindow*>(this)->logLine(
            QString("Save: dyn masks hd copied=%1").arg(dynMaskXCopied));
        if (project.background_ids.empty()) {
            project.background_ids.assign(static_cast<std::size_t>(frameCount), 0xffff);
            for (int i = 0; i < frameCount; ++i) {
                if (m_serumData.backgroundIDs.hasData(static_cast<uint32_t>(i))) {
                    project.background_ids[static_cast<std::size_t>(i)] =
                        *m_serumData.backgroundIDs[static_cast<uint32_t>(i)];
                }
            }
        }
        if (project.background_masks.size() != static_cast<std::size_t>(frameCount)) {
            project.background_masks.resize(static_cast<std::size_t>(frameCount));
        }
        for (int i = 0; i < frameCount; ++i) {
            cv::Mat& mask = project.background_masks[static_cast<std::size_t>(i)];
            if (!mask.empty()) {
                continue;
            }
            if (!m_serumData.backgroundmask.hasData(static_cast<uint32_t>(i))) {
                continue;
            }
            const uint8_t* data = m_serumData.backgroundmask[static_cast<uint32_t>(i)];
            if (data && pixels > 0) {
                mask = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
                std::memcpy(mask.data, data, pixels);
            }
        }
        if (project.background_masks_x.size() != static_cast<std::size_t>(frameCount)) {
            project.background_masks_x.resize(static_cast<std::size_t>(frameCount));
        }
        for (int i = 0; i < frameCount; ++i) {
            if (!project.frame_extra_flags.empty() &&
                project.frame_extra_flags[static_cast<std::size_t>(i)] == 0) {
                continue;
            }
            cv::Mat& maskX = project.background_masks_x[static_cast<std::size_t>(i)];
            if (!maskX.empty()) {
                continue;
            }
            if (!m_serumData.backgroundmask_extra.hasData(static_cast<uint32_t>(i))) {
                continue;
            }
            const uint8_t* data = m_serumData.backgroundmask_extra[static_cast<uint32_t>(i)];
            if (data && pixelsX > 0) {
                maskX = cv::Mat(heightX, widthX, CV_8UC1, cv::Scalar(0));
                std::memcpy(maskX.data, data, pixelsX);
            }
        }
    }
    const int backgroundCount = m_backgroundStore ? m_backgroundStore->count() : 0;
    project.background_frames.resize(static_cast<std::size_t>(backgroundCount));
    if (m_backgroundStore) {
        for (int i = 0; i < backgroundCount; ++i) {
            const cv::Mat bg = m_backgroundStore->loadCopy(i);
            if (!bg.empty()) {
                project.background_frames[static_cast<std::size_t>(i)] = bg;
                continue;
            }
            project.background_frames[static_cast<std::size_t>(i)] =
                cv::Mat(baseSize, CV_8UC3, cv::Scalar(0, 0, 0));
        }
    }
    project.background_frames_x = m_backgroundFramesX;
    if (m_serumDataLoaded) {
        const std::size_t bgCountSize = static_cast<std::size_t>(backgroundCount);
        if (project.background_frames_x.size() < bgCountSize) {
            project.background_frames_x.resize(bgCountSize);
        }
        for (int i = 0; i < backgroundCount; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i);
            if (!project.background_frames_x[idx].empty()) {
                continue;
            }
            if (i < static_cast<int>(m_serumData.nbackgrounds) &&
                m_serumData.backgroundframes_v2_extra.hasData(static_cast<uint32_t>(i))) {
                const uint16_t* data = m_serumData.backgroundframes_v2_extra[static_cast<uint32_t>(i)];
                if (data && m_serumData.fwidth_extra > 0 && m_serumData.fheight_extra > 0) {
                    project.background_frames_x[idx] = ConvertRgb565ToBgrMat(
                        data,
                        static_cast<int>(m_serumData.fwidth_extra),
                        static_cast<int>(m_serumData.fheight_extra));
                }
            }
        }
    }
    project.background_extra_flags = m_backgroundExtraFlags;
    project.background_ids = m_frameBackgroundIds;
    project.background_masks = m_frameBackgroundMasks;
    project.background_masks_x = m_frameBackgroundMasksX;
    project.active_reduced_palette = static_cast<uint8_t>(std::max(0, std::min(m_reducedPaletteIndex, kReducedPaletteCount - 1)));
    if (!m_hasLegacyRoundTrip) {
        project.preview_reduced_palette = project.active_reduced_palette;
    } else if (project.preview_reduced_palette >= kReducedPaletteCount) {
        project.preview_reduced_palette = project.active_reduced_palette;
    }
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
        if (QListWidgetItem* item = m_framePreviewList->item(row)) {
            if (!item->isSelected()) {
                item->setSelected(true);
            }
        }
    } else {
        m_framePreviewList->setCurrentRow(-1);
    }
    if (m_previewSelectedFrames.size() > 1) {
        for (int i = 0; i < m_framePreviewList->count(); ++i) {
            QListWidgetItem* item = m_framePreviewList->item(i);
            if (!item) {
                continue;
            }
            const int frameIndex = item->data(kFrameIndexRole).toInt();
            const bool selected = std::find(m_previewSelectedFrames.begin(),
                                            m_previewSelectedFrames.end(),
                                            frameIndex) != m_previewSelectedFrames.end();
            item->setSelected(selected);
        }
    }
    schedulePreviewSelectionUpdate();
    const std::vector<int> currentSelection = selectedPreviewFrameIndices();
    if (!currentSelection.empty()) {
        m_previewSelectedFrames = currentSelection;
    }
}

void MainWindow::updatePreviewSelectionStyles()
{
    if (m_isLoadingProject) {
        return;
    }
    if (!m_framePreviewList) {
        return;
    }
    const int currentRow = m_framePreviewList->currentRow();
    for (int row = 0; row < m_framePreviewList->count(); ++row) {
        QListWidgetItem* item = m_framePreviewList->item(row);
        if (!item) {
            continue;
        }
        const bool selected = item->isSelected();
        const bool secondary = selected && row != currentRow;
        item->setData(kPreviewSecondarySelectedRole, secondary);
    }
    m_framePreviewList->viewport()->update();
}

void MainWindow::schedulePreviewSelectionUpdate()
{
    if (m_isLoadingProject) {
        return;
    }
    if (!m_previewSelectionTimer) {
        updatePreviewSelectionStyles();
        if (m_previewSelectedOnly) {
            refreshFramePreviews();
        }
        return;
    }
    if (!m_previewSelectionTimer->isActive()) {
        m_previewSelectionTimer->start(0);
    }
}

void MainWindow::clearPreviewSelection()
{
    if (!m_framePreviewList) {
        return;
    }
    QSignalBlocker blocker(m_framePreviewList);
    if (m_framePreviewList->selectionModel()) {
        QSignalBlocker selectionBlocker(m_framePreviewList->selectionModel());
        m_framePreviewList->clearSelection();
    } else {
        m_framePreviewList->clearSelection();
    }
    m_framePreviewList->setCurrentRow(-1);
    m_previewSelectedFrames.clear();
    m_restorePreviewSelection = false;
    m_restorePreviewCurrent = -1;
    m_restorePreviewSelectionIndices.clear();
    m_previewSelectionClearRequested = true;
    schedulePreviewSelectionUpdate();
}

std::vector<int> MainWindow::selectedPreviewFrameIndices() const
{
    std::vector<int> indices;
    if (!m_framePreviewList) {
        return indices;
    }
    const auto items = m_framePreviewList->selectedItems();
    indices.reserve(static_cast<std::size_t>(items.size()));
    for (const QListWidgetItem* item : items) {
        if (!item) {
            continue;
        }
        const QVariant frameData = item->data(kFrameIndexRole);
        if (!frameData.isValid()) {
            continue;
        }
        indices.push_back(frameData.toInt());
    }
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    return indices;
}

void MainWindow::updatePlaybackPreviewHighlight(int frameIndex)
{
    if (!m_framePreviewList) {
        return;
    }
    QSignalBlocker blocker(m_framePreviewList);
    QListWidgetItem* target = nullptr;
    for (int row = 0; row < m_framePreviewList->count(); ++row) {
        QListWidgetItem* item = m_framePreviewList->item(row);
        if (!item) {
            continue;
        }
        const int itemFrame = item->data(kFrameIndexRole).toInt();
        const bool active = (frameIndex >= 0 && itemFrame == frameIndex);
        item->setData(kPreviewPlaybackRole, active);
        if (active) {
            target = item;
        }
    }
    m_framePreviewList->viewport()->update();
    if (target) {
        m_framePreviewList->scrollToItem(target, QAbstractItemView::PositionAtCenter);
    }
}

std::vector<int> MainWindow::targetFrameIndices() const
{
    std::vector<int> indices = selectedPreviewFrameIndices();
    if (indices.empty() && !m_previewSelectedFrames.empty()) {
        indices = m_previewSelectedFrames;
    }
    if (!indices.empty()) {
        return indices;
    }
    const int current = m_framesList ? m_framesList->currentRow() : -1;
    if (current >= 0) {
        indices.push_back(current);
    }
    return indices;
}

std::vector<int> MainWindow::playbackFrameIndices() const
{
    std::vector<int> indices = selectedPreviewFrameIndices();
    if (indices.empty() && !m_previewSelectedFrames.empty()) {
        indices = m_previewSelectedFrames;
    }
    if (indices.size() >= 2) {
        return indices;
    }
    int count = static_cast<int>(m_frameDurations.size());
    if (count <= 0 && m_frameStore) {
        count = m_frameStore->count();
    }
    if (count <= 0) {
        return {};
    }
    indices.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        indices.push_back(i);
    }
    return indices;
}

void MainWindow::startPlayback()
{
    stopPlayback();
    bool useAutosave = false;
    std::unique_ptr<QProgressDialog> progress;
    const bool shouldAutosave = m_autosaveEnabled && !m_isLoadingProject && m_projectDirty;
    if (!shouldAutosave) {
        logLine(QString("Playback start: autosave skipped (enabled=%1 dirty=%2 loading=%3)")
                    .arg(m_autosaveEnabled ? "true" : "false")
                    .arg(m_projectDirty ? "true" : "false")
                    .arg(m_isLoadingProject ? "true" : "false"));
    }
    if (shouldAutosave) {
        progress.reset(new QProgressDialog("Preparing playback...", QString(), 0, 0, this));
        progress->setWindowModality(Qt::ApplicationModal);
        progress->setCancelButton(nullptr);
        progress->setMinimumDuration(0);
        progress->show();
        QApplication::processEvents();
        useAutosave = autosaveProject(false);
        if (!useAutosave) {
            if (progress) {
                progress->close();
            }
            logLine("Playback failed: autosave did not complete.");
            return;
        }
    }
    setSerumRuntimePathsForPlayback(useAutosave);
    disposeSerumRuntime();
    m_playbackUsesSerumRuntime = ensureSerumRuntime();
    if (progress) {
        progress->close();
    }
    const std::vector<int> selected = selectedPreviewFrameIndices();
    const std::vector<int> fallbackSelected = selected.empty() ? m_previewSelectedFrames : selected;
    const bool multiSelect = fallbackSelected.size() >= 2;
    if (multiSelect) {
        m_playbackFrames = fallbackSelected;
    } else {
        m_playbackFrames = playbackFrameIndices();
    }
    if (m_playbackFrames.empty()) {
        return;
    }
    int currentPreview = -1;
    if (m_framePreviewList) {
        if (QListWidgetItem* currentItem = m_framePreviewList->currentItem()) {
            currentPreview = currentItem->data(kFrameIndexRole).toInt();
        }
    }
    const int current = m_framesList ? m_framesList->currentRow() : -1;
    const int selectedSingle = (selected.size() == 1) ? selected.front() : -1;
    int startIndex = m_playbackFrames.front();
    if (selectedSingle >= 0) {
        startIndex = selectedSingle;
    } else if (!fallbackSelected.empty() && fallbackSelected.size() == 1) {
        startIndex = fallbackSelected.front();
    } else if (currentPreview >= 0) {
        const auto it = std::find(m_playbackFrames.begin(), m_playbackFrames.end(), currentPreview);
        if (it != m_playbackFrames.end()) {
            startIndex = currentPreview;
        }
    } else if (current >= 0) {
        const auto it = std::find(m_playbackFrames.begin(), m_playbackFrames.end(), current);
        if (it != m_playbackFrames.end()) {
            startIndex = current;
        }
    }
    const auto startIt = std::find(m_playbackFrames.begin(), m_playbackFrames.end(), startIndex);
    m_playbackPos = startIt != m_playbackFrames.end()
        ? static_cast<int>(std::distance(m_playbackFrames.begin(), startIt))
        : 0;
    m_playbackActive = true;
    m_playbackPaused = false;
    if (m_canvasTabs && m_playbackTab) {
        m_canvasTabs->setCurrentWidget(m_playbackTab);
    }
    logLine(QString("Playback start: frames=%1 start=%2 autosave=%3 serum=%4")
                .arg(m_playbackFrames.size())
                .arg(startIndex)
                .arg(useAutosave ? "true" : "false")
                .arg(m_playbackUsesSerumRuntime ? "true" : "false"));
    renderPlaybackFrame();
    updatePlaybackButtons();
}

void MainWindow::pausePlayback()
{
    if (!m_playbackActive || m_playbackPaused) {
        return;
    }
    m_playbackPaused = true;
    if (m_playbackTimer) {
        m_playbackTimer->stop();
    }
    updatePlaybackButtons();
}

void MainWindow::resumePlayback()
{
    if (!m_playbackActive || !m_playbackPaused) {
        return;
    }
    m_playbackPaused = false;
    schedulePlaybackTick();
    updatePlaybackButtons();
}

void MainWindow::stopPlayback()
{
    if (m_playbackTimer) {
        m_playbackTimer->stop();
    }
    m_playbackActive = false;
    m_playbackPaused = false;
    if (m_playbackUsesSerumRuntime) {
        Serum_Scene_Reset();
    }
    m_playbackFrames.clear();
    m_playbackPos = -1;
    m_playbackFrameIndex = -1;
    m_playbackRotationData = nullptr;
    m_playbackBase565.clear();
    m_playbackRotationMask.clear();
    m_playbackStaticFrame.release();
    m_playbackOriginalPreview.release();
    m_playbackOriginalBuffer.clear();
    m_playbackUsesSerumRuntime = false;
    updatePlaybackPreviewHighlight(-1);
    updatePlaybackButtons();
}

void MainWindow::stepPlayback(int delta)
{
    if (delta == 0) {
        return;
    }
    if (m_playbackFrames.empty()) {
        m_playbackFrames = playbackFrameIndices();
    }
    if (m_playbackFrames.empty()) {
        return;
    }
    int pos = m_playbackPos;
    if (pos < 0) {
        const int current = m_framesList ? m_framesList->currentRow() : -1;
        const auto it = std::find(m_playbackFrames.begin(), m_playbackFrames.end(), current);
        pos = it != m_playbackFrames.end()
            ? static_cast<int>(std::distance(m_playbackFrames.begin(), it))
            : 0;
    }
    pos += delta;
    if (pos < 0) {
        pos = 0;
    } else if (pos >= static_cast<int>(m_playbackFrames.size())) {
        pos = static_cast<int>(m_playbackFrames.size()) - 1;
    }
    m_playbackPos = pos;
    m_playbackActive = true;
    m_playbackPaused = true;
    if (m_playbackTimer) {
        m_playbackTimer->stop();
    }
    if (m_canvasTabs && m_playbackTab) {
        m_canvasTabs->setCurrentWidget(m_playbackTab);
    }
    renderPlaybackFrame();
    updatePlaybackButtons();
}

void MainWindow::jumpPlayback(int index)
{
    if (index < 0) {
        return;
    }
    if (m_playbackFrames.empty()) {
        m_playbackFrames = playbackFrameIndices();
    }
    if (m_playbackFrames.empty()) {
        return;
    }
    auto it = std::find(m_playbackFrames.begin(), m_playbackFrames.end(), index);
    if (it == m_playbackFrames.end()) {
        m_playbackFrames = playbackFrameIndices();
        it = std::find(m_playbackFrames.begin(), m_playbackFrames.end(), index);
    }
    if (it == m_playbackFrames.end()) {
        return;
    }
    m_playbackPos = static_cast<int>(std::distance(m_playbackFrames.begin(), it));
    if (m_canvasTabs && m_playbackTab) {
        m_canvasTabs->setCurrentWidget(m_playbackTab);
    }
    renderPlaybackFrame();
    if (m_playbackActive && !m_playbackPaused) {
        schedulePlaybackTick();
    }
}

void MainWindow::updatePlaybackIdleFrame(int index)
{
    if (!m_playbackCanvas || m_playbackActive) {
        return;
    }
    if (index < 0) {
        m_playbackCanvas->setTitle("Playback");
        m_playbackCanvas->setImage(cv::Mat());
        m_playbackCanvas->canvas()->setGridSegments(0, 0, 0);
        m_playbackCanvas->canvas()->setGridScales(1, 1);
        m_playbackCanvas->canvas()->setGridRegions(QRect(), QRect());
        return;
    }
    const bool useHd = m_useHdFrame && hasHdFrame(index);
    const cv::Mat composed = renderFrameWithSerum(index, useHd);
    if (composed.empty()) {
        m_playbackCanvas->setImage(cv::Mat());
        m_playbackCanvas->canvas()->setGridSegments(0, 0, 0);
        m_playbackCanvas->canvas()->setGridScales(1, 1);
        m_playbackCanvas->canvas()->setGridRegions(QRect(), QRect());
        return;
    }
    cv::Mat reference = buildOriginalPreviewForIndex(index);
    cv::Mat original;
    if (m_playbackShowOriginal && !reference.empty()) {
        original = buildOriginalFrame(reference);
    }
    const QColor gap = m_playbackCanvas
        ? m_playbackCanvas->palette().color(QPalette::Window)
        : QApplication::palette().color(QPalette::Window);
    const cv::Scalar gapColor(gap.blue(), gap.green(), gap.red());
    const cv::Mat combined = (!original.empty())
        ? buildCombinedFrame(composed, original, gapColor)
        : EnsureBgr(composed);
    m_playbackCanvas->setTitle(QString("Playback (frame %1)").arg(index + 1));
    m_playbackCanvas->setImage(combined);
    if (!original.empty()) {
        const int gapPixels = FrameGapForWidth(composed.cols);
        m_playbackCanvas->canvas()->setGridSegments(composed.rows, gapPixels, original.rows);
        m_playbackCanvas->canvas()->setGridScales(1, 1);
        FrameLayout layout = BuildFrameLayout(composed, original);
        const QRect topRegion(layout.topX, 0, layout.topWidth, layout.topHeight);
        const QRect bottomRegion(layout.bottomX,
                                 layout.topHeight + gapPixels,
                                 layout.bottomWidth,
                                 layout.bottomHeight);
        m_playbackCanvas->canvas()->setGridRegions(topRegion, bottomRegion);
    } else {
        m_playbackCanvas->canvas()->setGridSegments(0, 0, 0);
        m_playbackCanvas->canvas()->setGridScales(1, 1);
        m_playbackCanvas->canvas()->setGridRegions(QRect(), QRect());
    }
}

void MainWindow::renderPlaybackFrame()
{
    if (!m_playbackActive || m_playbackFrames.empty() || !m_playbackCanvas) {
        return;
    }
    if (m_playbackPos < 0 || m_playbackPos >= static_cast<int>(m_playbackFrames.size())) {
        return;
    }
    const int frameIndex = m_playbackFrames[static_cast<std::size_t>(m_playbackPos)];
    m_playbackFrameIndex = frameIndex;
    updatePlaybackPreviewHighlight(frameIndex);
    m_playbackFrameDurationMs = 30;
    if (frameIndex >= 0 && frameIndex < static_cast<int>(m_frameDurations.size())) {
        const int duration = static_cast<int>(m_frameDurations[frameIndex]);
        if (duration > 0) {
            m_playbackFrameDurationMs = duration;
        }
    }
    if (m_playbackOverrideDurationCheck && m_playbackOverrideDurationCheck->isChecked() &&
        m_playbackOverrideDurationSpin) {
        m_playbackFrameDurationMs = m_playbackOverrideDurationSpin->value();
    }
    m_playbackUseHd = m_useHdFrame && hasHdFrame(frameIndex);
    m_playbackRotationData = rotationBlockForRead(frameIndex, m_playbackUseHd);
    m_playbackBase565.clear();
    m_playbackRotationMask.clear();
    m_playbackStaticFrame.release();
    m_playbackOriginalPreview.release();
    m_playbackOriginalBuffer.clear();
    m_playbackWidth = 0;
    m_playbackHeight = 0;
    m_playbackFrameClock.restart();

    cv::Mat reference = buildOriginalPreviewForIndex(frameIndex);
    const bool hasReference = frameIndex >= 0 &&
        frameIndex < static_cast<int>(m_frameRefs.size()) &&
        !m_frameRefs[static_cast<std::size_t>(frameIndex)].empty();
    if (m_playbackUsesSerumRuntime && !hasReference) {
        const QString message = QString("Playback failed: missing original frame %1 for libserum runtime.")
                                    .arg(frameIndex);
        logLine(message);
        statusBar()->showMessage(message, 7000);
        stopPlayback();
        return;
    }
    if (m_playbackUsesSerumRuntime && m_serumRuntimeFrame) {
        cv::Mat referenceForSerum = reference;
        if (referenceForSerum.type() != CV_8UC1 && !referenceForSerum.empty()) {
            cv::Mat gray;
            cv::cvtColor(EnsureBgr(referenceForSerum), gray, cv::COLOR_BGR2GRAY);
            referenceForSerum = gray;
        }
        const int targetW = m_serumData.fwidth > 0 ? static_cast<int>(m_serumData.fwidth) : referenceForSerum.cols;
        const int targetH = m_serumData.fheight > 0 ? static_cast<int>(m_serumData.fheight) : referenceForSerum.rows;
        if (!referenceForSerum.empty() &&
            (referenceForSerum.cols != targetW || referenceForSerum.rows != targetH)) {
            cv::resize(referenceForSerum, referenceForSerum, cv::Size(targetW, targetH), 0.0, 0.0, cv::INTER_NEAREST);
        }
        if (!referenceForSerum.empty()) {
            m_playbackOriginalBuffer.assign(referenceForSerum.data,
                                            referenceForSerum.data + referenceForSerum.total());
            const uint32_t colorizeResult = Serum_Colorize(m_playbackOriginalBuffer.data());
            if (m_serumRuntimeFrame->frameID == IDENTIFY_NO_FRAME) {
                const QString message = QString("Playback failed: Serum_Colorize could not identify frame %1 (result=%2).")
                                            .arg(frameIndex)
                                            .arg(colorizeResult);
                logLine(message);
                statusBar()->showMessage(message, 7000);
                stopPlayback();
                return;
            }
            const bool useHdRuntime = m_playbackUseHd &&
                m_serumRuntimeFrame->frame64 && m_serumRuntimeFrame->width64 > 0;
            const uint16_t* framePtr = useHdRuntime ? m_serumRuntimeFrame->frame64 : m_serumRuntimeFrame->frame32;
            const int width = useHdRuntime
                ? static_cast<int>(m_serumRuntimeFrame->width64)
                : static_cast<int>(m_serumRuntimeFrame->width32);
            const int height = useHdRuntime
                ? (m_serumData.fheight_extra > 0 ? static_cast<int>(m_serumData.fheight_extra) : 64)
                : (m_serumData.fheight > 0 ? static_cast<int>(m_serumData.fheight) : 32);
            if (framePtr && width > 0 && height > 0) {
                m_playbackStaticFrame = ConvertRgb565ToBgrMat(framePtr, width, height);
            } else {
                const QString message = QString("Playback failed: libserum returned invalid frame buffer for %1.")
                                            .arg(frameIndex);
                logLine(message);
                statusBar()->showMessage(message, 7000);
                stopPlayback();
                return;
            }
        } else {
            const QString message = QString("Playback failed: missing reference buffer for %1.")
                                        .arg(frameIndex);
            logLine(message);
            statusBar()->showMessage(message, 7000);
            stopPlayback();
            return;
        }
    }
    if (!m_playbackUsesSerumRuntime) {
        const QString message = QString("Playback failed: libserum runtime unavailable.");
        logLine(message);
        statusBar()->showMessage(message, 7000);
        stopPlayback();
        return;
    }
    if (m_playbackShowOriginal && !reference.empty()) {
        m_playbackOriginalPreview = buildOriginalFrame(reference);
    }
    const QColor gap = m_playbackCanvas
        ? m_playbackCanvas->palette().color(QPalette::Window)
        : QApplication::palette().color(QPalette::Window);
    const cv::Scalar gapColor(gap.blue(), gap.green(), gap.red());
    const cv::Mat combined = (!m_playbackOriginalPreview.empty())
        ? buildCombinedFrame(m_playbackStaticFrame, m_playbackOriginalPreview, gapColor)
        : m_playbackStaticFrame;
    m_playbackCanvas->setTitle(QString("Playback (frame %1)").arg(frameIndex + 1));
    m_playbackCanvas->setImage(combined);
    if (!m_playbackOriginalPreview.empty()) {
        const int gapPixels = FrameGapForWidth(m_playbackStaticFrame.cols);
        m_playbackCanvas->canvas()->setGridSegments(m_playbackStaticFrame.rows,
                                                    gapPixels,
                                                    m_playbackOriginalPreview.rows);
        m_playbackCanvas->canvas()->setGridScales(1, 1);
        FrameLayout layout = BuildFrameLayout(m_playbackStaticFrame, m_playbackOriginalPreview);
        const QRect topRegion(layout.topX, 0, layout.topWidth, layout.topHeight);
        const QRect bottomRegion(layout.bottomX,
                                 layout.topHeight + gapPixels,
                                 layout.bottomWidth,
                                 layout.bottomHeight);
        m_playbackCanvas->canvas()->setGridRegions(topRegion, bottomRegion);
    } else {
        m_playbackCanvas->canvas()->setGridSegments(0, 0, 0);
        m_playbackCanvas->canvas()->setGridScales(1, 1);
        m_playbackCanvas->canvas()->setGridRegions(QRect(), QRect());
    }
    schedulePlaybackTick();
}

void MainWindow::schedulePlaybackTick()
{
    if (!m_playbackActive || m_playbackPaused || !m_playbackTimer) {
        return;
    }
    const int elapsed = static_cast<int>(m_playbackFrameClock.elapsed());
    const int remaining = m_playbackFrameDurationMs - elapsed;
    if (remaining <= 0) {
        advancePlaybackFrame();
        return;
    }
    if (m_playbackUsesSerumRuntime && m_serumRuntimeFrame) {
        const uint32_t rotationResult = Serum_Rotate();
        const uint32_t nextDelay = rotationResult & 0xffff;
        if (rotationResult & (FLAG_RETURNED_V2_ROTATED32 | FLAG_RETURNED_V2_ROTATED64 | FLAG_RETURNED_V2_SCENE)) {
            const bool useHdRuntime = m_playbackUseHd &&
                m_serumRuntimeFrame->frame64 && m_serumRuntimeFrame->width64 > 0;
            const uint16_t* framePtr = useHdRuntime ? m_serumRuntimeFrame->frame64 : m_serumRuntimeFrame->frame32;
            const int width = useHdRuntime
                ? static_cast<int>(m_serumRuntimeFrame->width64)
                : static_cast<int>(m_serumRuntimeFrame->width32);
            const int height = useHdRuntime
                ? (m_serumData.fheight_extra > 0 ? static_cast<int>(m_serumData.fheight_extra) : 64)
                : (m_serumData.fheight > 0 ? static_cast<int>(m_serumData.fheight) : 32);
            if (framePtr && width > 0 && height > 0) {
                const cv::Mat rotatedMat = ConvertRgb565ToBgrMat(framePtr, width, height);
                if (m_playbackCanvas) {
                    const QColor gap = m_playbackCanvas
                        ? m_playbackCanvas->palette().color(QPalette::Window)
                        : QApplication::palette().color(QPalette::Window);
                    const cv::Scalar gapColor(gap.blue(), gap.green(), gap.red());
                    const cv::Mat combined = (!m_playbackOriginalPreview.empty())
                        ? buildCombinedFrame(rotatedMat, m_playbackOriginalPreview, gapColor)
                        : rotatedMat;
                    m_playbackCanvas->setImage(combined);
                }
            }
        }
        const int delay = (nextDelay > 0)
            ? std::min<int>(remaining, static_cast<int>(std::max<uint32_t>(nextDelay, 16)))
            : remaining;
        m_playbackTimer->start(delay);
        return;
    }
    if (!m_playbackUsesSerumRuntime) {
        m_playbackTimer->stop();
        return;
    }
    m_playbackTimer->start(remaining);
}

void MainWindow::advancePlaybackFrame()
{
    if (!m_playbackActive || m_playbackFrames.empty()) {
        stopPlayback();
        return;
    }
    m_playbackPos += 1;
    if (m_playbackPos >= static_cast<int>(m_playbackFrames.size())) {
        m_playbackPos = 0;
    }
    renderPlaybackFrame();
}

void MainWindow::updatePlaybackButtons()
{
    const bool hasFrames = m_frameStore && m_frameStore->count() > 0;
    if (m_previewPlayButton) {
        m_previewPlayButton->setEnabled(hasFrames);
    }
    if (m_previewStopButton) {
        m_previewStopButton->setEnabled(m_playbackActive);
    }
    if (m_previewPauseButton) {
        m_previewPauseButton->setEnabled(m_playbackActive && !m_playbackPaused);
    }
    if (m_previewPrevButton) {
        m_previewPrevButton->setEnabled(hasFrames);
    }
    if (m_previewNextButton) {
        m_previewNextButton->setEnabled(hasFrames);
    }
    if (m_previewRewindButton) {
        m_previewRewindButton->setEnabled(hasFrames);
    }
    if (m_previewFastForwardButton) {
        m_previewFastForwardButton->setEnabled(hasFrames);
    }
    if (m_playbackOverrideDurationCheck) {
        m_playbackOverrideDurationCheck->setEnabled(hasFrames);
    }
    if (m_playbackOverrideDurationSpin) {
        const bool enabled = hasFrames && m_playbackOverrideDurationCheck &&
            m_playbackOverrideDurationCheck->isChecked();
        m_playbackOverrideDurationSpin->setEnabled(enabled);
    }
}

void MainWindow::updatePreviewsForMaskId(int maskId)
{
    if (maskId < 0) {
        return;
    }
    const int count = m_frameStore ? m_frameStore->count() : 0;
    for (int i = 0; i < count && i < static_cast<int>(m_frameCompMaskIds.size()); ++i) {
        if (m_frameCompMaskIds[static_cast<std::size_t>(i)] == static_cast<uint8_t>(maskId)) {
            updateFramePreviewAt(i);
        }
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
            if (!m_framesList || m_framesList->currentRow() < 0) {
                statusBar()->showMessage("Select a frame to filter dynamic masks.", 2000);
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
            if (!m_spritesList || m_spritesList->currentRow() < 0) {
                statusBar()->showMessage("Select a sprite to filter frames.", 2000);
            }
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
    std::unordered_set<int> selectedFrames;
    if (m_previewSelectedOnly) {
        selectedFrames.reserve(m_previewSelectedFrames.size());
        for (int index : m_previewSelectedFrames) {
            selectedFrames.insert(index);
        }
    }
    if (m_previewSelectedOnly && selectedFrames.empty()) {
        return indices;
    }
    if (!m_previewFilterEnabled || m_previewFilterKind == PreviewFilterKind::None) {
        indices.reserve(static_cast<std::size_t>(count));
        for (int i = 0; i < count; ++i) {
            indices.push_back(i);
        }
        if (m_previewSelectedOnly && !selectedFrames.empty()) {
            std::vector<int> filtered;
            filtered.reserve(indices.size());
            for (int index : indices) {
                if (selectedFrames.count(index) > 0) {
                    filtered.push_back(index);
                }
            }
            indices.swap(filtered);
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
            const int referenceIndex = m_framesList ? m_framesList->currentRow() : -1;
            if (referenceIndex < 0 || referenceIndex >= static_cast<int>(m_frameDynamicMaskMaps.size())) {
                if (!(m_serumDataLoaded && referenceIndex >= 0 &&
                      referenceIndex < static_cast<int>(m_serumData.nframes))) {
                    break;
                }
            }
            const auto signatureForData = [](const uint8_t* data, int width, int height) -> uint32_t {
                if (!data || width <= 0 || height <= 0) {
                    return 0;
                }
                uint32_t signature = 0;
                for (int y = 0; y < height; ++y) {
                    const uint8_t* row = data + static_cast<std::size_t>(y) * width;
                    for (int x = 0; x < width; ++x) {
                        const uint8_t value = row[x];
                        if (value < MAX_DYNA_SETS_PER_FRAMEN) {
                            signature |= (1u << value);
                        }
                    }
                }
                return signature;
            };
            const uint32_t referenceSignature = [&]() -> uint32_t {
                if (m_serumDataLoaded && referenceIndex >= 0 &&
                    referenceIndex < static_cast<int>(m_serumData.nframes) &&
                    m_serumData.dynamasks.hasData(static_cast<uint32_t>(referenceIndex))) {
                    return signatureForData(m_serumData.dynamasks[static_cast<uint32_t>(referenceIndex)],
                                            static_cast<int>(m_serumData.fwidth),
                                            static_cast<int>(m_serumData.fheight));
                }
                if (referenceIndex >= 0 && referenceIndex < static_cast<int>(m_frameDynamicMaskMaps.size())) {
                    return signatureForData(m_frameDynamicMaskMaps[static_cast<std::size_t>(referenceIndex)].data,
                                            m_frameDynamicMaskMaps[static_cast<std::size_t>(referenceIndex)].cols,
                                            m_frameDynamicMaskMaps[static_cast<std::size_t>(referenceIndex)].rows);
                }
                return 0;
            }();
            for (int i = 0; i < count; ++i) {
                uint32_t signature = 0;
                if (m_serumDataLoaded && i >= 0 && i < static_cast<int>(m_serumData.nframes) &&
                    m_serumData.dynamasks.hasData(static_cast<uint32_t>(i))) {
                    signature = signatureForData(m_serumData.dynamasks[static_cast<uint32_t>(i)],
                                                 static_cast<int>(m_serumData.fwidth),
                                                 static_cast<int>(m_serumData.fheight));
                } else if (i >= 0 && i < static_cast<int>(m_frameDynamicMaskMaps.size())) {
                    signature = signatureForData(m_frameDynamicMaskMaps[static_cast<std::size_t>(i)].data,
                                                 m_frameDynamicMaskMaps[static_cast<std::size_t>(i)].cols,
                                                 m_frameDynamicMaskMaps[static_cast<std::size_t>(i)].rows);
                }
                if (signature == referenceSignature) {
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
        case PreviewFilterKind::Sprite: {
            const int selected = m_spritesList ? m_spritesList->currentRow() : -1;
            if (selected < 0 || m_frameSpriteAssignments.empty()) {
                break;
            }
            const int frameSlots = std::max(1, MAX_SPRITES_PER_FRAME);
            for (int i = 0; i < count; ++i) {
                const std::size_t base = static_cast<std::size_t>(i) * frameSlots;
                bool usesSprite = false;
                for (int slot = 0; slot < frameSlots; ++slot) {
                    const std::size_t idx = base + static_cast<std::size_t>(slot);
                    if (idx >= m_frameSpriteAssignments.size()) {
                        break;
                    }
                    if (m_frameSpriteAssignments[idx] == static_cast<uint8_t>(selected)) {
                        usesSprite = true;
                        break;
                    }
                }
                if (usesSprite) {
                    indices.push_back(i);
                }
            }
            break;
        }
        case PreviewFilterKind::None:
            break;
    }
    if (m_previewSelectedOnly && !selectedFrames.empty()) {
        std::vector<int> filtered;
        filtered.reserve(indices.size());
        for (int index : indices) {
            if (selectedFrames.count(index) > 0) {
                filtered.push_back(index);
            }
        }
        indices.swap(filtered);
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
    if (m_isLoadingProject) {
        return;
    }
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
    if (m_isLoadingProject) {
        return;
    }
    int scrollValue = 0;
    if (m_framePreviewList && m_framePreviewList->horizontalScrollBar()) {
        scrollValue = m_framePreviewList->horizontalScrollBar()->value();
    }
    const std::vector<int> selectedBefore = selectedPreviewFrameIndices();
    if (!selectedBefore.empty()) {
        m_previewSelectedFrames = selectedBefore;
    }
    const std::vector<int>& restoreSelection =
        selectedBefore.empty() ? m_previewSelectedFrames : selectedBefore;
    std::unordered_set<int> selectedLookup(restoreSelection.begin(), restoreSelection.end());
    QSignalBlocker blocker(m_framePreviewList);
    m_framePreviewList->clear();

    if (!m_framesList || m_framesList->count() == 0) {
        auto* item = new QListWidgetItem("No frames");
        item->setFlags(item->flags() & ~Qt::ItemIsEnabled);
        m_framePreviewList->addItem(item);
        updatePlaybackButtons();
        return;
    }

    const int count = m_frameStore->count();
    const QListWidgetItem* firstItem = (m_framesList && m_framesList->count() > 0)
        ? m_framesList->item(0)
        : nullptr;
    if (count <= 0 ||
        (count == 1 && firstItem && firstItem->text().startsWith("No frames"))) {
        auto* item = new QListWidgetItem("No frames");
        item->setFlags(item->flags() & ~Qt::ItemIsEnabled);
        m_framePreviewList->addItem(item);
        updatePlaybackButtons();
        return;
    }

    const std::vector<int> indices = buildPreviewFrameIndices();
    if (indices.empty()) {
        auto* item = new QListWidgetItem(m_previewFilterEnabled ? "No frames match filter" : "No frames");
        item->setFlags(item->flags() & ~Qt::ItemIsEnabled);
        m_framePreviewList->addItem(item);
        updatePlaybackButtons();
        return;
    }

    for (int i : indices) {
        cv::Mat reference = buildOriginalPreviewForIndex(i);
        cv::Mat hdFrame;
        if (i >= 0 && i < static_cast<int>(m_frameExtraFrames.size())) {
            hdFrame = m_frameExtraFrames[static_cast<std::size_t>(i)];
        }
        cv::Mat composed = renderFrameWithSerum(i, false);
        if (composed.empty()) {
            continue;
        }
        cv::Mat hdComposed;
        if (!hdFrame.empty()) {
            hdComposed = renderFrameWithSerum(i, true);
        }
        cv::Mat previewMat = buildPreviewFrame(i, composed, reference, hdComposed);
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
        if (!selectedLookup.empty() && selectedLookup.count(i) > 0) {
            item->setSelected(true);
        }
        if (m_playbackActive && i == m_playbackFrameIndex) {
            item->setData(kPreviewPlaybackRole, true);
        }
        m_framePreviewList->addItem(item);
    }

    refreshFramePreviewSelection();
    schedulePreviewSelectionUpdate();
    m_framePreviewList->doItemsLayout();
    if (m_framePreviewList && m_framePreviewList->horizontalScrollBar()) {
        m_framePreviewList->horizontalScrollBar()->setValue(scrollValue);
    }
    if (m_previewRotateEnabled && m_previewRotationTimer && !m_previewRotationTimer->isActive()) {
        schedulePreviewRotationUpdate();
    }
    if (m_playbackActive && m_playbackFrameIndex >= 0) {
        updatePlaybackPreviewHighlight(m_playbackFrameIndex);
    }
    updatePlaybackButtons();
}

void MainWindow::refreshFramePreviewsForRotation()
{
    if (m_isLoadingProject || !m_framePreviewList) {
        return;
    }
    const std::vector<int> indices = buildPreviewFrameIndices();
    if (indices.empty()) {
        return;
    }
    QSignalBlocker blocker(m_framePreviewList);
    QSignalBlocker selectionBlocker(m_framePreviewList->selectionModel());
    for (int i : indices) {
        const int row = previewRowForFrame(i);
        if (row < 0 || row >= m_framePreviewList->count()) {
            continue;
        }
        QListWidgetItem* item = m_framePreviewList->item(row);
        if (!item) {
            continue;
        }
        cv::Mat reference = buildOriginalPreviewForIndex(i);
        cv::Mat hdFrame;
        if (i >= 0 && i < static_cast<int>(m_frameExtraFrames.size())) {
            hdFrame = m_frameExtraFrames[static_cast<std::size_t>(i)];
        }
        cv::Mat composed = renderFrameWithSerum(i, false);
        if (composed.empty()) {
            continue;
        }
        cv::Mat hdComposed;
        if (!hdFrame.empty()) {
            hdComposed = renderFrameWithSerum(i, true);
        }
        cv::Mat previewMat = buildPreviewFrame(i, composed, reference, hdComposed);
        cv::Mat rgb;
        cv::cvtColor(previewMat, rgb, cv::COLOR_BGR2RGB);
        QImage previewImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
        const QSize iconSize = hasHdFrame(i)
            ? QSize(kPreviewIconWidthHd, kPreviewIconHeightHd)
            : QSize(kPreviewIconWidth, kPreviewIconHeight);
        QPixmap pixmap = QPixmap::fromImage(previewImage.copy());
        pixmap = pixmap.scaled(iconSize, Qt::KeepAspectRatio, Qt::FastTransformation);
        item->setIcon(QIcon(pixmap));
        item->setData(kPreviewIconSizeRole, iconSize);
        item->setSizeHint(PreviewItemSizeForIcon(iconSize, m_framePreviewList->font()));
        if (m_playbackActive && i == m_playbackFrameIndex) {
            item->setData(kPreviewPlaybackRole, true);
        } else {
            item->setData(kPreviewPlaybackRole, false);
        }
    }
    m_framePreviewList->doItemsLayout();
    if (m_previewRotateEnabled && m_previewRotationTimer) {
        schedulePreviewRotationUpdate();
    }
}

void MainWindow::updateFramePreviewAt(int index)
{
    if (m_isLoadingProject) {
        return;
    }
    const int row = previewRowForFrame(index);
    if (row < 0 || row >= m_framePreviewList->count()) {
        return;
    }
    const cv::Mat* image = m_frameStore->at(index);
    if (!image || image->empty()) {
        return;
    }
    QSignalBlocker blocker(m_framePreviewList);
    QSignalBlocker selectionBlocker(m_framePreviewList->selectionModel());
    QListWidgetItem* item = m_framePreviewList->item(row);
    if (!item) {
        return;
    }
    cv::Mat reference = buildOriginalPreviewForIndex(index);
    cv::Mat hdFrame;
    if (index >= 0 && index < static_cast<int>(m_frameExtraFrames.size())) {
        hdFrame = m_frameExtraFrames[static_cast<std::size_t>(index)];
    }
    cv::Mat composed = renderFrameWithSerum(index, false);
    cv::Mat hdComposed;
    if (!hdFrame.empty()) {
        hdComposed = renderFrameWithSerum(index, true);
    }
    cv::Mat previewMat = buildPreviewFrame(index, composed, reference, hdComposed);
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
    if (m_playbackActive && index == m_playbackFrameIndex) {
        item->setData(kPreviewPlaybackRole, true);
    } else {
        item->setData(kPreviewPlaybackRole, false);
    }
    m_framePreviewList->doItemsLayout();
}

cv::Mat MainWindow::buildPreviewFrame(int index,
                                      const cv::Mat& colorized,
                                      const cv::Mat& reference,
                                      const cv::Mat& hdFrame) const
{
    cv::Mat color = EnsureBgr(colorized);
    cv::Mat hd = EnsureBgr(hdFrame);
    if (m_previewRotateEnabled) {
        color = applyRotationPreview(color, index, false);
        if (!hd.empty()) {
            hd = applyRotationPreview(hd, index, true);
        }
    }
    cv::Mat original = buildOriginalFrame(reference);
    if (m_previewMaskOverlayEnabled && !original.empty()) {
        MaskMode overlayMode = m_maskMode;
        if (overlayMode == MaskMode::None) {
            const PreviewFilterKind filterKind = currentPreviewFilterKind();
            if (filterKind == PreviewFilterKind::DynamicMask) {
                overlayMode = MaskMode::Dynamic;
            } else {
                overlayMode = MaskMode::Comparison;
            }
        }
        if (overlayMode == MaskMode::Comparison) {
            uint8_t maskId = 255;
            const uint32_t frameId = static_cast<uint32_t>(index);
            if (m_serumDataLoaded && index >= 0 && index < static_cast<int>(m_serumData.nframes) &&
                m_serumData.compmaskID.hasData(frameId)) {
                const uint8_t* maskPtr = m_serumData.compmaskID[frameId];
                if (maskPtr) {
                    maskId = maskPtr[0];
                }
            } else if (index >= 0 && index < static_cast<int>(m_frameCompMaskIds.size())) {
                maskId = m_frameCompMaskIds[static_cast<std::size_t>(index)];
            }
            if (maskId != 255) {
                cv::Mat mask;
                if (m_serumDataLoaded && m_serumData.compmasks.hasData(maskId)) {
                    mask = cv::Mat(static_cast<int>(m_serumData.fheight),
                                   static_cast<int>(m_serumData.fwidth),
                                   CV_8UC1,
                                   const_cast<uint8_t*>(m_serumData.compmasks[maskId]));
                } else if (maskId < m_compMasks.size()) {
                    mask = m_compMasks[static_cast<std::size_t>(maskId)];
                }
                if (MaskHasContent(mask)) {
                    original = buildMaskPreview(original, mask, cv::Vec3b(200, 0, 200));
                }
            }
        } else if (overlayMode == MaskMode::Dynamic) {
            const int setId = currentFrameDynamicMaskId();
            cv::Mat map;
            if (m_serumDataLoaded && index >= 0 && index < static_cast<int>(m_serumData.nframes) &&
                m_serumData.dynamasks.hasData(static_cast<uint32_t>(index))) {
                map = cv::Mat(static_cast<int>(m_serumData.fheight),
                              static_cast<int>(m_serumData.fwidth),
                              CV_8UC1,
                              const_cast<uint8_t*>(m_serumData.dynamasks[static_cast<uint32_t>(index)]));
            } else if (index >= 0 && index < static_cast<int>(m_frameDynamicMaskMaps.size())) {
                map = m_frameDynamicMaskMaps[static_cast<std::size_t>(index)];
            }
            if (setId >= 0 && !map.empty()) {
                cv::Mat mask = buildDynamicMaskFromMap(map, setId);
                if (MaskHasContent(mask)) {
                    original = buildMaskPreview(original, mask, cv::Vec3b(0, 200, 255));
                }
            }
        }
    }
    if (color.empty() && hd.empty() && original.empty()) {
        return cv::Mat();
    }
    const QColor gap = m_framePreviewList
        ? m_framePreviewList->palette().color(QPalette::Window)
        : QApplication::palette().color(QPalette::Window);
    const int maxWidth = std::max({color.cols, original.cols, hd.cols});
    const int gapPixels = FrameGapForWidth(maxWidth);
    return BuildStackedFrames({color, original, hd}, cv::Scalar(gap.blue(), gap.green(), gap.red()), gapPixels);
}

cv::Mat MainWindow::applyRotationPreview(const cv::Mat& colorized,
                                         int frameIndex,
                                         bool useHd) const
{
    if (colorized.empty() || frameIndex < 0) {
        return EnsureBgr(colorized);
    }
    const uint16_t* rotationsData = rotationBlockForRead(frameIndex, useHd);
    if (!rotationsData) {
        return EnsureBgr(colorized);
    }

    const uint32_t elapsed = m_previewRotationClock.isValid()
        ? static_cast<uint32_t>(m_previewRotationClock.elapsed())
        : 0;
    auto& stateMap = useHd ? m_previewRotationStatesX : m_previewRotationStates;
    auto [it, inserted] = stateMap.emplace(frameIndex, SerumEditorRotationState{});
    SerumEditorRotationState& state = it->second;
    if (inserted) {
        SerumEditor_InitRotationState(rotationsData, &state, elapsed);
    }
    std::vector<uint16_t> base565;
    std::vector<uint16_t> rotationsInFrame;
    int width = 0;
    int height = 0;
    if (!renderFrameWithSerumRaw(frameIndex, useHd, cv::Mat(), base565, width,
                                 height, &rotationsInFrame, nullptr) ||
        base565.empty()) {
        return EnsureBgr(colorized);
    }
    std::vector<uint16_t> rotated(base565.size(), 0);
    SerumEditor_ApplyRotationsMasked(rotationsData, base565.data(), rotated.data(),
                                     rotationsInFrame.data(),
                                     static_cast<uint32_t>(width),
                                     static_cast<uint32_t>(height),
                                     &state, elapsed);
    return ConvertRgb565ToBgrMat(rotated.data(), width, height);
}

void MainWindow::setCanvasRotationEnabled(bool enabled)
{
    m_canvasRotateEnabled = enabled;
    if (m_framesCanvas) {
        m_framesCanvas->setRotateChecked(enabled);
    }
    if (!enabled) {
        if (m_rotationTimer) {
            m_rotationTimer->stop();
        }
        m_rotationFrameIndex = -1;
        updateFrameCanvasImage(m_framesList ? m_framesList->currentRow() : -1);
        return;
    }
    m_rotationClock.restart();
    m_rotationFrameIndex = -1;
    updateCanvasRotationFrame();
}

void MainWindow::updateCanvasRotationFrame()
{
    if (!m_canvasRotateEnabled || !m_framesCanvas) {
        return;
    }
    const int frameIndex = m_framesList ? m_framesList->currentRow() : -1;
    if (frameIndex < 0) {
        return;
    }
    const bool useHd = m_useHdFrame && hasHdFrame(frameIndex);
    const uint16_t* rotationsData = rotationBlockForRead(frameIndex, useHd);
    if (!rotationsData) {
        setCanvasRotationEnabled(false);
        return;
    }
    if (frameIndex != m_rotationFrameIndex || useHd != m_rotationUseHd) {
        SerumEditor_InitRotationState(rotationsData, &m_rotationState,
                                      static_cast<uint32_t>(m_rotationClock.elapsed()));
        m_rotationFrameIndex = frameIndex;
        m_rotationUseHd = useHd;
    }

    std::vector<uint16_t> base565;
    std::vector<uint16_t> rotationsInFrame;
    int width = 0;
    int height = 0;
    if (!renderFrameWithSerumRaw(frameIndex, useHd, cv::Mat(), base565, width,
                                 height, &rotationsInFrame, nullptr) ||
        base565.empty()) {
        const cv::Mat composed = renderFrameWithSerum(frameIndex, useHd);
        setFrameCanvasFromComposed(frameIndex, composed);
        return;
    }
    std::vector<uint16_t> rotated(base565.size(), 0);
    const uint32_t nowMs = static_cast<uint32_t>(m_rotationClock.elapsed());
    const uint32_t nextDelay = SerumEditor_ApplyRotationsMasked(
        rotationsData, base565.data(), rotated.data(),
        rotationsInFrame.data(), static_cast<uint32_t>(width),
        static_cast<uint32_t>(height), &m_rotationState, nowMs);
    const cv::Mat rotatedMat = ConvertRgb565ToBgrMat(rotated.data(), width, height);
    setFrameCanvasFromComposed(frameIndex, rotatedMat);

    if (m_rotationTimer) {
        if (nextDelay == 0) {
            m_rotationTimer->stop();
        } else {
            m_rotationTimer->start(static_cast<int>(std::max<uint32_t>(nextDelay, 16)));
        }
    }
}

void MainWindow::resetCanvasRotationState()
{
    m_rotationFrameIndex = -1;
    m_rotationUseHd = false;
    if (m_rotationTimer) {
        m_rotationTimer->stop();
    }
    m_previewRotationStates.clear();
    m_previewRotationStatesX.clear();
    if (m_canvasRotateEnabled) {
        m_rotationClock.restart();
        updateCanvasRotationFrame();
    }
    if (m_previewRotateEnabled) {
        m_previewRotationClock.restart();
        schedulePreviewRotationUpdate();
        refreshFramePreviews();
    }
}

void MainWindow::schedulePreviewRotationUpdate()
{
    if (m_isLoadingProject) {
        return;
    }
    if (!m_previewRotateEnabled || !m_previewRotationTimer) {
        return;
    }
    const std::vector<int> indices = buildPreviewFrameIndices();
    if (indices.empty()) {
        m_previewRotationTimer->stop();
        return;
    }
    const auto minDelayForData = [](const uint16_t* data) -> uint32_t {
        if (!data) {
            return 0;
        }
        uint32_t best = 0;
        for (int rot = 0; rot < MAX_COLOR_ROTATIONN; ++rot) {
            const std::size_t base = static_cast<std::size_t>(rot) * MAX_LENGTH_COLOR_ROTATION;
            const uint16_t length = data[base];
            const uint16_t delay = data[base + 1];
            if (length == 0 || delay == 0) {
                continue;
            }
            if (best == 0 || delay < best) {
                best = delay;
            }
        }
        return best;
    };
    uint32_t nextDelay = 0;
    for (int frameIndex : indices) {
        if (frameIndex < 0) {
            continue;
        }
        uint32_t delay = 0;
        delay = minDelayForData(rotationBlockForRead(frameIndex, false));
        const uint32_t hdDelay = minDelayForData(rotationBlockForRead(frameIndex, true));
        if (hdDelay > 0) {
            delay = (delay == 0) ? hdDelay : std::min(delay, hdDelay);
        }
        if (delay > 0) {
            if (nextDelay == 0 || delay < nextDelay) {
                nextDelay = delay;
            }
        }
    }
    if (nextDelay == 0) {
        m_previewRotationTimer->stop();
        return;
    }
    m_previewRotationTimer->start(static_cast<int>(std::max<uint32_t>(nextDelay, 16)));
}

void MainWindow::updateFrameUsageHighlights(int frameIndex)
{
    if (!m_spritesList && !m_dynamicMaskList && !m_backgroundList && !m_spriteZoneList) {
        return;
    }
    if (frameIndex < 0) {
        if (m_spritesList) {
            for (int i = 0; i < m_spritesList->count(); ++i) {
                if (auto* item = m_spritesList->item(i)) {
                    item->setData(kPreviewUsageRole, false);
                }
            }
            m_spritesList->viewport()->update();
        }
        if (m_dynamicMaskList) {
            for (int i = 0; i < m_dynamicMaskList->count(); ++i) {
                if (auto* item = m_dynamicMaskList->item(i)) {
                    item->setData(kPreviewUsageRole, false);
                }
            }
            m_dynamicMaskList->viewport()->update();
        }
        if (m_backgroundList) {
            for (int i = 0; i < m_backgroundList->count(); ++i) {
                if (auto* item = m_backgroundList->item(i)) {
                    item->setData(kPreviewUsageRole, false);
                }
            }
            m_backgroundList->viewport()->update();
        }
        if (m_spriteZoneList) {
            for (int i = 0; i < m_spriteZoneList->count(); ++i) {
                if (auto* item = m_spriteZoneList->item(i)) {
                    item->setData(kPreviewUsageRole, false);
                }
            }
            m_spriteZoneList->viewport()->update();
        }
        return;
    }

    if (m_spritesList) {
        std::unordered_set<int> usedSprites;
        const std::size_t base = static_cast<std::size_t>(frameIndex) * MAX_SPRITES_PER_FRAME;
        if (base + MAX_SPRITES_PER_FRAME <= m_frameSpriteAssignments.size()) {
            for (int slot = 0; slot < MAX_SPRITES_PER_FRAME; ++slot) {
                const uint8_t spriteId = m_frameSpriteAssignments[base + static_cast<std::size_t>(slot)];
                if (spriteId != 255) {
                    usedSprites.insert(static_cast<int>(spriteId));
                }
            }
        }
        for (int i = 0; i < m_spritesList->count(); ++i) {
            auto* item = m_spritesList->item(i);
            if (!item) {
                continue;
            }
            const int spriteId = item->data(Qt::UserRole).toInt();
            item->setData(kPreviewUsageRole, usedSprites.count(spriteId) > 0);
        }
        m_spritesList->viewport()->update();
    }

    if (m_dynamicMaskList) {
        std::array<bool, MAX_DYNA_SETS_PER_FRAMEN> used{};
        used.fill(false);
        auto markUsed = [&used](const cv::Mat& map) {
            for (int y = 0; y < map.rows; ++y) {
                const uint8_t* row = map.ptr<uint8_t>(y);
                for (int x = 0; x < map.cols; ++x) {
                    const uint8_t value = row[x];
                    if (value != 255 && value < MAX_DYNA_SETS_PER_FRAMEN) {
                        used[value] = true;
                    }
                }
            }
        };
        if (frameIndex < static_cast<int>(m_frameDynamicMaskMaps.size())) {
            const cv::Mat& sdMap = m_frameDynamicMaskMaps[static_cast<std::size_t>(frameIndex)];
            if (!sdMap.empty()) {
                markUsed(sdMap);
            }
        }
        if (frameIndex < static_cast<int>(m_frameDynamicMaskMapsX.size())) {
            const cv::Mat& hdMap = m_frameDynamicMaskMapsX[static_cast<std::size_t>(frameIndex)];
            if (!hdMap.empty()) {
                markUsed(hdMap);
            }
        }
        for (int i = 0; i < m_dynamicMaskList->count() && i < MAX_DYNA_SETS_PER_FRAMEN; ++i) {
            auto* item = m_dynamicMaskList->item(i);
            if (!item) {
                continue;
            }
            item->setData(kPreviewUsageRole, used[static_cast<std::size_t>(i)]);
        }
        m_dynamicMaskList->viewport()->update();
    }

    if (m_backgroundList) {
        uint16_t bgId = 0xffff;
        if (frameIndex >= 0 && frameIndex < static_cast<int>(m_frameBackgroundIds.size())) {
            bgId = m_frameBackgroundIds[static_cast<std::size_t>(frameIndex)];
        }
        for (int i = 0; i < m_backgroundList->count(); ++i) {
            auto* item = m_backgroundList->item(i);
            if (!item) {
                continue;
            }
            const int itemId = item->data(Qt::UserRole).toInt();
            item->setData(kPreviewUsageRole, bgId != 0xffff && itemId == static_cast<int>(bgId));
        }
        m_backgroundList->viewport()->update();
    }

    if (m_spriteZoneList) {
        const std::size_t zoneCount = m_spriteZones.size();
        for (int i = 0; i < m_spriteZoneList->count(); ++i) {
            auto* item = m_spriteZoneList->item(i);
            if (!item) {
                continue;
            }
            bool used = false;
            if (i >= 0 && static_cast<std::size_t>(i) < zoneCount) {
                used = !m_spriteZones[static_cast<std::size_t>(i)].sprites.empty();
            }
            item->setData(kPreviewUsageRole, used);
        }
        m_spriteZoneList->viewport()->update();
    }
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
    const int gap = FrameGapForWidth(top.cols);
    return BuildStackedFrames({top, bottom}, gapColor, gap);
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

cv::Mat MainWindow::buildSpriteCoverageMask(int index, bool useHd) const
{
    if (index < 0 || !m_spriteStore) {
        return cv::Mat();
    }
    const bool useSerumData = m_serumDataLoaded &&
        index >= 0 && index < static_cast<int>(m_serumData.nframes);
    const cv::Mat* frame = nullptr;
    if (useHd && index >= 0 && index < static_cast<int>(m_frameExtraFrames.size()) &&
        !m_frameExtraFrames[static_cast<std::size_t>(index)].empty()) {
        frame = &m_frameExtraFrames[static_cast<std::size_t>(index)];
    } else if (m_frameStore) {
        frame = m_frameStore->at(index);
    }
    if (!frame || frame->empty()) {
        return cv::Mat();
    }
    std::size_t spriteCount = m_spriteOriginals.size();
    if (useSerumData) {
        spriteCount = std::max(spriteCount, static_cast<std::size_t>(m_serumData.nsprites));
    }
    if (spriteCount == 0) {
        return cv::Mat();
    }
    const std::size_t baseSlot = static_cast<std::size_t>(index) * MAX_SPRITES_PER_FRAME;
    if (baseSlot + MAX_SPRITES_PER_FRAME > m_frameSpriteAssignments.size()) {
        return cv::Mat();
    }
    const std::size_t spriteBbBase = static_cast<std::size_t>(index) * MAX_SPRITES_PER_FRAME * 4;
    if (spriteBbBase + MAX_SPRITES_PER_FRAME * 4 > m_frameSpriteBBoxes.size()) {
        return cv::Mat();
    }

    cv::Mat reference = buildOriginalPreviewForIndex(index);
    if (reference.type() != CV_8UC1) {
        cv::Mat gray;
        cv::cvtColor(EnsureBgr(reference), gray, cv::COLOR_BGR2GRAY);
        reference = gray;
    }
    if (reference.empty()) {
        return cv::Mat();
    }

    std::vector<SerumEditorSpriteView> spriteViews(spriteCount);
    for (int slot = 0; slot < MAX_SPRITES_PER_FRAME; ++slot) {
        const uint8_t spriteId = m_frameSpriteAssignments[baseSlot + static_cast<std::size_t>(slot)];
        if (spriteId == 255 || spriteId >= spriteCount) {
            continue;
        }
        if (useHd && !hasHdSprite(static_cast<int>(spriteId))) {
            continue;
        }
        SerumEditorSpriteView& view = spriteViews[spriteId];
        if (view.original) {
            continue;
        }
        const std::size_t spriteIndex = static_cast<std::size_t>(spriteId);
        const cv::Mat* spriteOriginal = const_cast<MainWindow*>(this)->ensureSpriteOriginalLocal(
            static_cast<int>(spriteId));
        if (!spriteOriginal || spriteOriginal->empty()) {
            continue;
        }
        view.original = spriteOriginal->data;
        if (spriteIndex < m_spriteMasksX.size()) {
            view.mask_extra = m_spriteMasksX[spriteIndex].data;
        }
        if (spriteIndex < m_spriteShapeModes.size()) {
            view.shape_mode = m_spriteShapeModes[spriteIndex];
        }
        const std::size_t detAreaOffset = spriteIndex * MAX_SPRITE_DETECT_AREAS * 4;
        if (detAreaOffset + MAX_SPRITE_DETECT_AREAS * 4 <= m_spriteDetAreas.size()) {
            view.det_areas = m_spriteDetAreas.data() + detAreaOffset;
        }
        const std::size_t detDwordOffset = spriteIndex * MAX_SPRITE_DETECT_AREAS;
        if (detDwordOffset + MAX_SPRITE_DETECT_AREAS <= m_spriteDetDwords.size()) {
            view.det_dwords = m_spriteDetDwords.data() + detDwordOffset;
        }
        if (detDwordOffset + MAX_SPRITE_DETECT_AREAS <= m_spriteDetDwordPos.size()) {
            view.det_dword_pos = m_spriteDetDwordPos.data() + detDwordOffset;
        }
    }

    SerumEditorDataView dataView;
    dataView.width = static_cast<uint32_t>(reference.cols);
    dataView.height = static_cast<uint32_t>(reference.rows);
    dataView.width_extra = static_cast<uint32_t>(frame->cols);
    dataView.height_extra = static_cast<uint32_t>(frame->rows);
    dataView.nocolors = m_noColors;
    dataView.nsprites = static_cast<uint32_t>(spriteCount);
    dataView.sprites = spriteViews.data();

    SerumEditorFrameView frameView;
    frameView.original = reference.data;
    frameView.frame_sprites = m_frameSpriteAssignments.data() + baseSlot;
    frameView.frame_sprite_bboxes = m_frameSpriteBBoxes.data() + spriteBbBase;

    SerumEditorSpriteMatch matches[MAX_SPRITES_PER_FRAME];
    std::memset(matches, 0, sizeof(matches));
    const uint8_t matchCount = SerumEditor_MatchSprites(&dataView, &frameView,
                                                        matches, MAX_SPRITES_PER_FRAME);
    if (matchCount == 0) {
        return cv::Mat();
    }

    cv::Mat mask(frame->rows, frame->cols, CV_8UC1, cv::Scalar(0));
    bool hasContent = false;
    const bool extraIs64 = dataView.height_extra == 64;
    for (uint8_t i = 0; i < matchCount; ++i) {
        const uint8_t spriteId = matches[i].sprite_index;
        if (spriteId == 255 || spriteId >= spriteCount) {
            continue;
        }
        const SerumEditorSpriteView& sprite = spriteViews[spriteId];
        if (!sprite.original) {
            continue;
        }
        if (!useHd) {
            for (uint16_t y = 0; y < matches[i].hei; ++y) {
                const int frameY = matches[i].fry + y;
                if (frameY < 0 || frameY >= mask.rows) {
                    continue;
                }
                uint8_t* maskRow = mask.ptr<uint8_t>(frameY);
                for (uint16_t x = 0; x < matches[i].wid; ++x) {
                    const int frameX = matches[i].frx + x;
                    if (frameX < 0 || frameX >= mask.cols) {
                        continue;
                    }
                    const uint32_t spriteIndex =
                        static_cast<uint32_t>(matches[i].spy + y) * MAX_SPRITE_WIDTH +
                        static_cast<uint32_t>(matches[i].spx + x);
                    const uint8_t value = sprite.original[spriteIndex];
                    if (value == 255) {
                        continue;
                    }
                    maskRow[frameX] = 1;
                    hasContent = true;
                }
            }
            continue;
        }

        const uint16_t thei = extraIs64 ? matches[i].hei * 2 : matches[i].hei / 2;
        const uint16_t twid = extraIs64 ? matches[i].wid * 2 : matches[i].wid / 2;
        const uint16_t tfrx = extraIs64 ? matches[i].frx * 2 : matches[i].frx / 2;
        const uint16_t tfry = extraIs64 ? matches[i].fry * 2 : matches[i].fry / 2;
        const uint16_t tspx = extraIs64 ? matches[i].spx * 2 : matches[i].spx / 2;
        const uint16_t tspy = extraIs64 ? matches[i].spy * 2 : matches[i].spy / 2;
        for (uint16_t y = 0; y < thei; ++y) {
            const int frameY = tfry + y;
            if (frameY < 0 || frameY >= mask.rows) {
                continue;
            }
            uint8_t* maskRow = mask.ptr<uint8_t>(frameY);
            for (uint16_t x = 0; x < twid; ++x) {
                const int frameX = tfrx + x;
                if (frameX < 0 || frameX >= mask.cols) {
                    continue;
                }
                const uint32_t spriteIndex =
                    static_cast<uint32_t>(tspy + y) * MAX_SPRITE_WIDTH +
                    static_cast<uint32_t>(tspx + x);
                    if (sprite.mask_extra) {
                        if (sprite.mask_extra[spriteIndex] == 255) {
                            continue;
                        }
                    } else {
                    const int srcx = extraIs64 ? (matches[i].spx + x / 2)
                                               : (matches[i].spx + x * 2);
                    const int srcy = extraIs64 ? (matches[i].spy + y / 2)
                                               : (matches[i].spy + y * 2);
                    if (srcx < 0 || srcy < 0 || srcx >= MAX_SPRITE_WIDTH || srcy >= MAX_SPRITE_HEIGHT) {
                        continue;
                    }
                    const uint32_t srcIndex =
                        static_cast<uint32_t>(srcy) * MAX_SPRITE_WIDTH +
                        static_cast<uint32_t>(srcx);
                    const uint8_t value = sprite.original[srcIndex];
                    if (value == 255) {
                        continue;
                    }
                }
                maskRow[frameX] = 1;
                hasContent = true;
            }
        }
    }
    if (!hasContent) {
        return cv::Mat();
    }
    return mask;
}

void MainWindow::restoreSpriteCoverage(cv::Mat& target,
                                       const cv::Mat& backup,
                                       const cv::Mat& spriteMask) const
{
    if (target.empty() || backup.empty() || spriteMask.empty()) {
        return;
    }
    const int rows = std::min(target.rows, spriteMask.rows);
    const int cols = std::min(target.cols, spriteMask.cols);
    for (int y = 0; y < rows; ++y) {
        const uint8_t* mrow = spriteMask.ptr<uint8_t>(y);
        if (target.type() == CV_8UC3) {
            cv::Vec3b* row = target.ptr<cv::Vec3b>(y);
            const cv::Vec3b* brow = backup.ptr<cv::Vec3b>(y);
            for (int x = 0; x < cols; ++x) {
                if (mrow[x]) {
                    row[x] = brow[x];
                }
            }
        } else if (target.type() == CV_8UC4) {
            cv::Vec4b* row = target.ptr<cv::Vec4b>(y);
            const cv::Vec4b* brow = backup.ptr<cv::Vec4b>(y);
            for (int x = 0; x < cols; ++x) {
                if (mrow[x]) {
                    row[x] = brow[x];
                }
            }
        }
    }
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

bool MainWindow::renderFrameWithSerumRaw(int index,
                                         bool useHd,
                                         const cv::Mat& overrideColorized,
                                         std::vector<uint16_t>& out565,
                                         int& outWidth,
                                         int& outHeight,
                                         std::vector<uint16_t>* rotationsInFrame,
                                         const uint32_t* rotationShifts) const
{
    out565.clear();
    outWidth = 0;
    outHeight = 0;
    if (index < 0 || !m_frameStore) {
        return false;
    }
    const cv::Mat* frameImage = nullptr;
    cv::Mat hdFrame;
    const bool hasOverride = !overrideColorized.empty();
    const bool useSerumData = m_serumDataLoaded &&
        index >= 0 && index < static_cast<int>(m_serumData.nframes);
    const_cast<MainWindow*>(this)->logLine(
        QString("renderFrameWithSerum: index=%1 hd=%2 override=%3 serum=%4")
            .arg(index)
            .arg(useHd ? "true" : "false")
            .arg(hasOverride ? "true" : "false")
            .arg(useSerumData ? "true" : "false"));
    const uint16_t* serumFrame = nullptr;
    const uint16_t* serumFrameExtra = nullptr;
    if (useSerumData) {
        if (useHd) {
            const uint8_t* extraFlag = m_serumData.isextraframe[static_cast<uint32_t>(index)];
            if (extraFlag && extraFlag[0] != 0) {
                serumFrame = m_serumData.cframes_v2_extra[static_cast<uint32_t>(index)];
            }
        } else {
            serumFrame = m_serumData.cframes_v2[static_cast<uint32_t>(index)];
            const uint8_t* extraFlag = m_serumData.isextraframe[static_cast<uint32_t>(index)];
            if (extraFlag && extraFlag[0] != 0) {
                serumFrameExtra = m_serumData.cframes_v2_extra[static_cast<uint32_t>(index)];
            }
        }
    }
    if (hasOverride) {
        frameImage = &overrideColorized;
    } else if (useHd && index >= 0 && index < static_cast<int>(m_frameExtraFrames.size())) {
        hdFrame = m_frameExtraFrames[static_cast<std::size_t>(index)];
        if (!hdFrame.empty()) {
            frameImage = &hdFrame;
        }
    }
    if (!frameImage && !serumFrame) {
        frameImage = m_frameStore->at(index);
    }
    const bool useLocalColorized = hasOverride || (frameImage && !frameImage->empty());
    if (!useLocalColorized && !serumFrame) {
        return false;
    }

    cv::Mat reference = buildOriginalPreviewForIndex(index);
    if (reference.type() != CV_8UC1) {
        cv::Mat gray;
        cv::cvtColor(EnsureBgr(reference), gray, cv::COLOR_BGR2GRAY);
        reference = gray;
    }
    if (reference.empty()) {
        return false;
    }
    if (m_noColors > 0 && m_noColors < 64) {
        cv::Mat clamped = reference.clone();
        const uint8_t maxValue = static_cast<uint8_t>(m_noColors - 1);
        for (int y = 0; y < clamped.rows; ++y) {
            uint8_t* row = clamped.ptr<uint8_t>(y);
            for (int x = 0; x < clamped.cols; ++x) {
                if (row[x] > maxValue) {
                    row[x] = maxValue;
                }
            }
        }
        reference = clamped;
    }

    const int baseWidth = reference.cols;
    const int baseHeight = reference.rows;
    if (serumFrame && !useLocalColorized) {
        outWidth = useHd ? static_cast<int>(m_serumData.fwidth_extra) : static_cast<int>(m_serumData.fwidth);
        outHeight = useHd ? static_cast<int>(m_serumData.fheight_extra) : static_cast<int>(m_serumData.fheight);
    } else {
        outWidth = frameImage ? frameImage->cols : 0;
        outHeight = frameImage ? frameImage->rows : 0;
    }

    std::vector<uint16_t> frame565;
    std::vector<uint16_t> frame565Extra;
    if (useLocalColorized && frameImage) {
        frame565 = ConvertBgrMatToRgb565(*frameImage);
    }
    if (!useHd && !hasOverride) {
        const cv::Mat* extra = (index >= 0 && index < static_cast<int>(m_frameExtraFrames.size()))
            ? &m_frameExtraFrames[static_cast<std::size_t>(index)]
            : nullptr;
        if (extra && !extra->empty()) {
            frame565Extra = ConvertBgrMatToRgb565(*extra);
        }
    }

    SerumEditorFrameView frameView;
    frameView.original = reference.data;
    if (useLocalColorized) {
        frameView.colorized = useHd ? nullptr : frame565.data();
        frameView.colorized_extra = useHd
            ? frame565.data()
            : (frame565Extra.empty() ? nullptr : frame565Extra.data());
    } else {
        frameView.colorized = useHd ? nullptr : serumFrame;
        frameView.colorized_extra = useHd ? serumFrame : (serumFrameExtra ? serumFrameExtra : nullptr);
    }

    const uint32_t frameId = static_cast<uint32_t>(index);
    const cv::Mat* localDynamicMap = (index >= 0 && index < static_cast<int>(m_frameDynamicMaskMaps.size()))
        ? &m_frameDynamicMaskMaps[static_cast<std::size_t>(index)]
        : nullptr;
    const cv::Mat* localDynamicMapExtra = (index >= 0 && index < static_cast<int>(m_frameDynamicMaskMapsX.size()))
        ? &m_frameDynamicMaskMapsX[static_cast<std::size_t>(index)]
        : nullptr;
    const bool hasLocalDynamicMap = localDynamicMap && !localDynamicMap->empty();
    const bool hasLocalDynamicMapExtra = localDynamicMapExtra && !localDynamicMapExtra->empty();
    const bool hasLocalDynamicColors = index >= 0 &&
        index < static_cast<int>(m_frameDynamicColors.size()) &&
        !m_frameDynamicColors[static_cast<std::size_t>(index)].empty();
    const uint8_t* serumDynamask = nullptr;
    const uint8_t* serumDynamaskExtra = nullptr;
    const uint16_t* serumDynaCols = nullptr;
    const uint16_t* serumDynaColsExtra = nullptr;
    if (useSerumData) {
        if (!hasLocalDynamicMap && m_serumData.dynamasks.hasData(frameId)) {
            serumDynamask = m_serumData.dynamasks[frameId];
        }
        if (!hasLocalDynamicMapExtra && m_serumData.dynamasks_extra.hasData(frameId)) {
            serumDynamaskExtra = m_serumData.dynamasks_extra[frameId];
        }
        if (!hasLocalDynamicColors && m_serumData.dyna4cols_v2.hasData(frameId)) {
            serumDynaCols = m_serumData.dyna4cols_v2[frameId];
        }
        if (!hasLocalDynamicColors && m_serumData.dyna4cols_v2_extra.hasData(frameId)) {
            serumDynaColsExtra = m_serumData.dyna4cols_v2_extra[frameId];
        }
    }
    if (serumDynamask) {
        frameView.dynamask = serumDynamask;
    } else if (localDynamicMap) {
        frameView.dynamask = localDynamicMap->data;
    }
    if (serumDynamaskExtra) {
        frameView.dynamask_extra = serumDynamaskExtra;
    } else if (localDynamicMapExtra) {
        frameView.dynamask_extra = localDynamicMapExtra->data;
    }
    if (serumDynaCols) {
        frameView.dyna4cols = serumDynaCols;
    }
    if (serumDynaColsExtra) {
        frameView.dyna4cols_extra = serumDynaColsExtra;
    }
    if (useHd && serumDynaCols && !serumDynaColsExtra) {
        frameView.dyna4cols_extra = serumDynaCols;
    }
    if (!serumDynaCols && index >= 0 && index < static_cast<int>(m_frameDynamicColors.size())) {
        const std::vector<uint16_t>& colors = m_frameDynamicColors[static_cast<std::size_t>(index)];
        if (!colors.empty()) {
            frameView.dyna4cols = colors.data();
            if (!serumDynaColsExtra) {
                frameView.dyna4cols_extra = colors.data();
            }
        }
    }
    cv::Mat scaledDynamicMask;
    if (useHd) {
        const cv::Mat* hdMap = (index >= 0 && index < static_cast<int>(m_frameDynamicMaskMapsX.size()))
            ? &m_frameDynamicMaskMapsX[static_cast<std::size_t>(index)]
            : nullptr;
        const cv::Mat* sdMap = (index >= 0 && index < static_cast<int>(m_frameDynamicMaskMaps.size()))
            ? &m_frameDynamicMaskMaps[static_cast<std::size_t>(index)]
            : nullptr;
        if ((!hdMap || hdMap->empty()) && sdMap && !sdMap->empty()) {
            cv::resize(*sdMap, scaledDynamicMask, cv::Size(outWidth, outHeight), 0.0, 0.0, cv::INTER_NEAREST);
            frameView.dynamask_extra = scaledDynamicMask.data;
        } else if ((!hdMap || hdMap->empty()) && (!sdMap || sdMap->empty()) &&
                   serumDynamask && m_serumData.fwidth > 0 && m_serumData.fheight > 0) {
            cv::Mat serumMap(static_cast<int>(m_serumData.fheight),
                             static_cast<int>(m_serumData.fwidth),
                             CV_8UC1,
                             const_cast<uint8_t*>(serumDynamask));
            cv::resize(serumMap, scaledDynamicMask, cv::Size(outWidth, outHeight),
                       0.0, 0.0, cv::INTER_NEAREST);
            frameView.dynamask_extra = scaledDynamicMask.data;
        }
    }

    uint16_t backgroundId = 0xffff;
    if (m_showBackgroundLayer) {
        if (index >= 0 && index < static_cast<int>(m_frameBackgroundIds.size())) {
            backgroundId = m_frameBackgroundIds[static_cast<std::size_t>(index)];
        } else if (useSerumData && m_serumData.backgroundIDs.hasData(frameId)) {
            const uint16_t* bg = m_serumData.backgroundIDs[frameId];
            backgroundId = bg ? bg[0] : 0xffff;
        }
    }
    frameView.background_id = backgroundId;
    const cv::Mat* localBgMask = (index >= 0 && index < static_cast<int>(m_frameBackgroundMasks.size()))
        ? &m_frameBackgroundMasks[static_cast<std::size_t>(index)]
        : nullptr;
    const cv::Mat* localBgMaskExtra = (index >= 0 && index < static_cast<int>(m_frameBackgroundMasksX.size()))
        ? &m_frameBackgroundMasksX[static_cast<std::size_t>(index)]
        : nullptr;
    const bool hasLocalBgMask = localBgMask && !localBgMask->empty();
    const bool hasLocalBgMaskExtra = localBgMaskExtra && !localBgMaskExtra->empty();
    const uint8_t* serumBgMask = nullptr;
    const uint8_t* serumBgMaskExtra = nullptr;
    if (useSerumData && backgroundId != 0xffff) {
        if (!hasLocalBgMask && m_serumData.backgroundmask.hasData(frameId)) {
            serumBgMask = m_serumData.backgroundmask[frameId];
        }
        if (!hasLocalBgMaskExtra && m_serumData.backgroundmask_extra.hasData(frameId)) {
            serumBgMaskExtra = m_serumData.backgroundmask_extra[frameId];
        }
    }
    if (serumBgMask) {
        frameView.background_mask = serumBgMask;
    } else if (localBgMask) {
        frameView.background_mask = localBgMask->data;
    }
    if (serumBgMaskExtra) {
        frameView.background_mask_extra = serumBgMaskExtra;
    } else if (localBgMaskExtra) {
        frameView.background_mask_extra = localBgMaskExtra->data;
    }

    std::vector<uint16_t> background565;
    std::vector<uint16_t> background565Extra;
    if (backgroundId != 0xffff) {
        const cv::Mat* backgroundOverride = nullptr;
        if (useSerumData && m_backgroundStore &&
            m_backgroundStore->isDirty(static_cast<int>(backgroundId))) {
            backgroundOverride = m_backgroundStore->peek(static_cast<int>(backgroundId));
        }
        if (backgroundOverride && !backgroundOverride->empty()) {
            background565 = ConvertBgrMatToRgb565(*backgroundOverride);
            frameView.background_frame = background565.data();
        } else if (useSerumData &&
                   backgroundId < m_serumData.nbackgrounds &&
                   m_serumData.backgroundframes_v2.hasData(backgroundId)) {
            frameView.background_frame = m_serumData.backgroundframes_v2[backgroundId];
        } else if (m_backgroundStore && backgroundId < m_backgroundStore->count()) {
            const cv::Mat* background = m_backgroundStore->at(static_cast<int>(backgroundId));
            if (background && !background->empty()) {
                background565 = ConvertBgrMatToRgb565(*background);
                frameView.background_frame = background565.data();
            }
        }
        if (backgroundId < m_backgroundFramesX.size() &&
            !m_backgroundFramesX[static_cast<std::size_t>(backgroundId)].empty()) {
            background565Extra = ConvertBgrMatToRgb565(m_backgroundFramesX[static_cast<std::size_t>(backgroundId)]);
            frameView.background_frame_extra = background565Extra.data();
        } else if (useSerumData &&
                   backgroundId < m_serumData.nbackgrounds &&
                   m_serumData.backgroundframes_v2_extra.hasData(backgroundId)) {
            frameView.background_frame_extra = m_serumData.backgroundframes_v2_extra[backgroundId];
        }
    }
    cv::Mat scaledBackgroundMask;
    cv::Mat scaledBackgroundFrame;
    if (useHd && backgroundId != 0xffff) {
        const cv::Mat* hdMask = (index >= 0 && index < static_cast<int>(m_frameBackgroundMasksX.size()))
            ? &m_frameBackgroundMasksX[static_cast<std::size_t>(index)]
            : nullptr;
        const cv::Mat* sdMask = (index >= 0 && index < static_cast<int>(m_frameBackgroundMasks.size()))
            ? &m_frameBackgroundMasks[static_cast<std::size_t>(index)]
            : nullptr;
        if (!frameView.background_mask_extra) {
            cv::Mat maskSource;
            if (sdMask && !sdMask->empty()) {
                maskSource = *sdMask;
            } else if (serumBgMask) {
                maskSource = cv::Mat(static_cast<int>(m_serumData.fheight),
                                     static_cast<int>(m_serumData.fwidth),
                                     CV_8UC1,
                                     const_cast<uint8_t*>(serumBgMask));
            }
            if (!maskSource.empty()) {
                cv::resize(maskSource, scaledBackgroundMask, cv::Size(outWidth, outHeight),
                           0.0, 0.0, cv::INTER_NEAREST);
                frameView.background_mask_extra = scaledBackgroundMask.data;
            }
        }
        if (!frameView.background_frame_extra) {
            cv::Mat backgroundSource;
            if (useSerumData && m_backgroundStore &&
                m_backgroundStore->isDirty(static_cast<int>(backgroundId))) {
                const cv::Mat* sdBackground = m_backgroundStore->peek(static_cast<int>(backgroundId));
                if (sdBackground && !sdBackground->empty()) {
                    backgroundSource = *sdBackground;
                }
            } else if (frameView.background_frame) {
                backgroundSource = ConvertRgb565ToBgrMat(frameView.background_frame,
                                                         static_cast<int>(m_serumData.fwidth),
                                                         static_cast<int>(m_serumData.fheight));
            } else if (m_backgroundStore && backgroundId < m_backgroundStore->count()) {
                const cv::Mat* sdBackground = m_backgroundStore->at(static_cast<int>(backgroundId));
                if (sdBackground && !sdBackground->empty()) {
                    backgroundSource = *sdBackground;
                }
            }
            if (!backgroundSource.empty()) {
                cv::resize(backgroundSource, scaledBackgroundFrame, cv::Size(outWidth, outHeight), 0.0, 0.0, cv::INTER_NEAREST);
                background565Extra = ConvertBgrMatToRgb565(scaledBackgroundFrame);
                frameView.background_frame_extra = background565Extra.data();
            }
        }
    }

    const uint8_t* frameSprites = nullptr;
    const uint16_t* frameSpriteBBoxes = nullptr;
    const std::size_t spriteBase = static_cast<std::size_t>(index) * MAX_SPRITES_PER_FRAME;
    const bool hasLocalFrameSprites = spriteBase + MAX_SPRITES_PER_FRAME <= m_frameSpriteAssignments.size();
    if (hasLocalFrameSprites) {
        frameSprites = m_frameSpriteAssignments.data() + spriteBase;
    } else if (useSerumData && m_serumData.framesprites.hasData(frameId)) {
        frameSprites = m_serumData.framesprites[frameId];
    }
    const std::size_t spriteBbBase = static_cast<std::size_t>(index) * MAX_SPRITES_PER_FRAME * 4;
    const bool hasLocalFrameSpriteBBoxes = spriteBbBase + MAX_SPRITES_PER_FRAME * 4 <= m_frameSpriteBBoxes.size();
    if (hasLocalFrameSpriteBBoxes) {
        frameSpriteBBoxes = m_frameSpriteBBoxes.data() + spriteBbBase;
    } else if (useSerumData && m_serumData.framespriteBB.hasData(frameId)) {
        frameSpriteBBoxes = m_serumData.framespriteBB[frameId];
    }
    frameView.frame_sprites = frameSprites;
    frameView.frame_sprite_bboxes = frameSpriteBBoxes;

    std::size_t spriteCount = m_spriteOriginals.size();
    if (useSerumData) {
        spriteCount = std::max(spriteCount, static_cast<std::size_t>(m_serumData.nsprites));
    }
    std::vector<SerumEditorSpriteView> spriteViews(spriteCount);
    std::vector<std::vector<uint16_t>> sprite565(spriteCount);
    std::vector<std::vector<uint16_t>> sprite565Extra(spriteCount);
    const QSize spriteBaseSize = serumSpriteBaseSize();
    const int spriteSrcWidth = spriteBaseSize.width();
    const int spriteSrcHeight = spriteBaseSize.height();
    const bool spriteNeedsPadding =
        spriteSrcWidth != MAX_SPRITE_WIDTH || spriteSrcHeight != MAX_SPRITE_HEIGHT;
    const int spriteCopyWidth = std::min(spriteSrcWidth, MAX_SPRITE_WIDTH);
    const int spriteCopyHeight = std::min(spriteSrcHeight, MAX_SPRITE_HEIGHT);
    if (frameSprites && spriteCount > 0) {
        for (int slot = 0; slot < MAX_SPRITES_PER_FRAME; ++slot) {
            const uint8_t spriteId = frameSprites[slot];
            if (spriteId == 255 || spriteId >= spriteCount) {
                continue;
            }
            SerumEditorSpriteView& view = spriteViews[spriteId];
            if (view.original) {
                continue;
            }
            const std::size_t spriteIndex = static_cast<std::size_t>(spriteId);
            const uint32_t spriteId32 = static_cast<uint32_t>(spriteId);
            const uint8_t* serumOriginal = useSerumData ? serumSpriteOriginalData(spriteId32) : nullptr;
            if (spriteIndex < m_spriteOriginals.size() &&
                !m_spriteOriginals[spriteIndex].empty()) {
                view.original = m_spriteOriginals[spriteIndex].data;
            } else if (serumOriginal) {
                view.original = serumOriginal;
            }
            const cv::Mat* spriteOverride = nullptr;
            if (useSerumData && m_spriteStore &&
                m_spriteStore->isDirty(static_cast<int>(spriteId))) {
                spriteOverride = m_spriteStore->peek(static_cast<int>(spriteId));
            }
            if (spriteOverride && !spriteOverride->empty()) {
                sprite565[spriteIndex] = ConvertBgrMatToRgb565(*spriteOverride);
                view.colored = sprite565[spriteIndex].data();
            } else if (spriteIndex < m_spriteColored.size()) {
                const cv::Mat& colored = m_spriteColored[spriteIndex];
                if (!colored.empty()) {
                    sprite565[spriteIndex] = ConvertBgrMatToRgb565(colored);
                    view.colored = sprite565[spriteIndex].data();
                }
            }
            if (!view.colored && useSerumData && m_serumData.spritecolored.elementCount() > 0) {
                const uint16_t* data = m_serumData.spritecolored[spriteId32];
                const std::size_t count = m_serumData.spritecolored.elementCount();
                if (data && count > 0) {
                    sprite565[spriteIndex].assign(data, data + count);
                    view.colored = sprite565[spriteIndex].data();
                }
            }
            if (spriteIndex < m_spriteColoredX.size()) {
                const cv::Mat& colored = m_spriteColoredX[spriteIndex];
                if (!colored.empty()) {
                    sprite565Extra[spriteIndex] = ConvertBgrMatToRgb565(colored);
                    view.colored_extra = sprite565Extra[spriteIndex].data();
                }
            }
            if (!view.colored_extra && useSerumData &&
                m_serumData.spritecolored_extra.elementCount() > 0) {
                const uint16_t* data = m_serumData.spritecolored_extra[spriteId32];
                const std::size_t count = m_serumData.spritecolored_extra.elementCount();
                if (data && count > 0) {
                    sprite565Extra[spriteIndex].assign(data, data + count);
                    view.colored_extra = sprite565Extra[spriteIndex].data();
                }
            }
            if (spriteIndex < m_spriteMasksX.size() &&
                !m_spriteMasksX[spriteIndex].empty()) {
                view.mask_extra = m_spriteMasksX[spriteIndex].data;
            } else if (useSerumData && m_serumData.spritemask_extra.hasData(spriteId32)) {
                view.mask_extra = m_serumData.spritemask_extra[spriteId32];
            }
            if (spriteIndex < m_spriteDynamicMasks.size() &&
                !m_spriteDynamicMasks[spriteIndex].empty()) {
                view.dynasprite_mask = m_spriteDynamicMasks[spriteIndex].data;
            }
            if (!view.dynasprite_mask && useSerumData &&
                m_serumData.dynaspritemasks.hasData(spriteId32)) {
                view.dynasprite_mask = m_serumData.dynaspritemasks[spriteId32];
            }
            if (spriteIndex < m_spriteDynamicMasksX.size() &&
                !m_spriteDynamicMasksX[spriteIndex].empty()) {
                view.dynasprite_mask_extra = m_spriteDynamicMasksX[spriteIndex].data;
            }
            if (!view.dynasprite_mask_extra && useSerumData &&
                m_serumData.dynaspritemasks_extra.hasData(spriteId32)) {
                view.dynasprite_mask_extra = m_serumData.dynaspritemasks_extra[spriteId32];
            }
            if (spriteIndex < m_spriteDynamicColors.size()) {
                const auto& dynCols = m_spriteDynamicColors[spriteIndex];
                if (!dynCols.empty()) {
                    view.dynasprite_cols = dynCols.data();
                }
            }
            if (!view.dynasprite_cols && useSerumData &&
                m_serumData.dynasprite4cols.hasData(spriteId32)) {
                view.dynasprite_cols = m_serumData.dynasprite4cols[spriteId32];
            }
            if (spriteIndex < m_spriteDynamicColorsX.size()) {
                const auto& dynCols = m_spriteDynamicColorsX[spriteIndex];
                if (!dynCols.empty()) {
                    view.dynasprite_cols_extra = dynCols.data();
                }
            }
            if (!view.dynasprite_cols_extra && useSerumData &&
                m_serumData.dynasprite4cols_extra.hasData(spriteId32)) {
                view.dynasprite_cols_extra = m_serumData.dynasprite4cols_extra[spriteId32];
            }
            if (spriteIndex < m_spriteShapeModes.size()) {
                view.shape_mode = m_spriteShapeModes[spriteIndex];
            }
            const std::size_t detAreaOffset = spriteIndex * MAX_SPRITE_DETECT_AREAS * 4;
            if (detAreaOffset + MAX_SPRITE_DETECT_AREAS * 4 <= m_spriteDetAreas.size()) {
                view.det_areas = m_spriteDetAreas.data() + detAreaOffset;
            }
            const std::size_t detDwordOffset = spriteIndex * MAX_SPRITE_DETECT_AREAS;
            if (detDwordOffset + MAX_SPRITE_DETECT_AREAS <= m_spriteDetDwords.size()) {
                view.det_dwords = m_spriteDetDwords.data() + detDwordOffset;
            }
            if (detDwordOffset + MAX_SPRITE_DETECT_AREAS <= m_spriteDetDwordPos.size()) {
                view.det_dword_pos = m_spriteDetDwordPos.data() + detDwordOffset;
            }
            if (useSerumData) {
                if (!view.original && serumOriginal) {
                    view.original = serumOriginal;
                }
                if (!view.colored && m_serumData.spritecolored.elementCount() > 0) {
                    const uint16_t* data = m_serumData.spritecolored[spriteId32];
                    const std::size_t count = m_serumData.spritecolored.elementCount();
                    if (data && count > 0) {
                        sprite565[spriteIndex].assign(data, data + count);
                        view.colored = sprite565[spriteIndex].data();
                    }
                }
                if (!view.colored_extra && m_serumData.spritecolored_extra.elementCount() > 0) {
                    const uint16_t* data = m_serumData.spritecolored_extra[spriteId32];
                    const std::size_t count = m_serumData.spritecolored_extra.elementCount();
                    if (data && count > 0) {
                        sprite565Extra[spriteIndex].assign(data, data + count);
                        view.colored_extra = sprite565Extra[spriteIndex].data();
                    }
                }
                if (!view.mask_extra && m_serumData.spritemask_extra.hasData(spriteId32)) {
                    view.mask_extra = m_serumData.spritemask_extra[spriteId32];
                }
                if (!view.dynasprite_mask && m_serumData.dynaspritemasks.hasData(spriteId32)) {
                    view.dynasprite_mask = m_serumData.dynaspritemasks[spriteId32];
                }
                if (!view.dynasprite_mask_extra && m_serumData.dynaspritemasks_extra.hasData(spriteId32)) {
                    view.dynasprite_mask_extra = m_serumData.dynaspritemasks_extra[spriteId32];
                }
                if (!view.dynasprite_cols && m_serumData.dynasprite4cols.hasData(spriteId32)) {
                    view.dynasprite_cols = m_serumData.dynasprite4cols[spriteId32];
                }
                if (!view.dynasprite_cols_extra && m_serumData.dynasprite4cols_extra.hasData(spriteId32)) {
                    view.dynasprite_cols_extra = m_serumData.dynasprite4cols_extra[spriteId32];
                }
                if (view.shape_mode == 0 && m_serumData.sprshapemode.hasData(spriteId32)) {
                    const uint8_t* mode = m_serumData.sprshapemode[spriteId32];
                    view.shape_mode = mode ? mode[0] : 0;
                }
                if (!view.det_areas && m_serumData.spritedetareas.hasData(spriteId32)) {
                    view.det_areas = m_serumData.spritedetareas[spriteId32];
                }
                if (!view.det_dwords && m_serumData.spritedetdwords.hasData(spriteId32)) {
                    view.det_dwords = m_serumData.spritedetdwords[spriteId32];
                }
                if (!view.det_dword_pos && m_serumData.spritedetdwordpos.hasData(spriteId32)) {
                    view.det_dword_pos = m_serumData.spritedetdwordpos[spriteId32];
                }
            }
        }
    }

    SerumEditorDataView dataView;
    dataView.width = static_cast<uint32_t>(baseWidth);
    dataView.height = static_cast<uint32_t>(baseHeight);
    dataView.width_extra = static_cast<uint32_t>(outWidth);
    dataView.height_extra = static_cast<uint32_t>(outHeight);
    dataView.nocolors = m_noColors;
    dataView.nsprites = static_cast<uint32_t>(spriteCount);
    dataView.sprites = spriteViews.data();

    SerumEditorSpriteMatch matches[MAX_SPRITES_PER_FRAME];
    std::memset(matches, 0, sizeof(matches));
    const uint8_t matchCount =
        SerumEditor_MatchSprites(&dataView, &frameView, matches, MAX_SPRITES_PER_FRAME);

    out565.assign(static_cast<std::size_t>(outWidth) * outHeight, 0);

    const uint16_t* rotationsData = rotationBlockForRead(index, useHd);
    if (rotationsInFrame) {
        rotationsInFrame->assign(static_cast<std::size_t>(outWidth) * outHeight * 2, 0xffff);
    }

    bool ok = false;
    if (rotationsInFrame && rotationsData) {
        ok = SerumEditor_RenderFrameWithRotations(&dataView, &frameView, matches,
                                                  matchCount, useHd, rotationsData,
                                                  rotationShifts, out565.data(),
                                                  rotationsInFrame->data());
    } else {
        ok = SerumEditor_RenderFrame(&dataView, &frameView, matches, matchCount, useHd,
                                     out565.data());
    }
    return ok;
}

cv::Mat MainWindow::renderFrameWithSerum(int index, bool useHd) const
{
    return renderFrameWithSerum(index, useHd, cv::Mat());
}

cv::Mat MainWindow::renderFrameWithSerum(int index, bool useHd, const cv::Mat& overrideColorized) const
{
    if (index < 0 || !m_frameStore) {
        return cv::Mat();
    }
    const cv::Mat* frameImage = nullptr;
    cv::Mat hdFrame;
    const bool hasOverride = !overrideColorized.empty();
    const bool useSerumData = m_serumDataLoaded &&
        index >= 0 && index < static_cast<int>(m_serumData.nframes);
    const uint16_t* serumFrame = nullptr;
    const uint16_t* serumFrameExtra = nullptr;
    if (useSerumData) {
        if (useHd) {
            const uint8_t* extraFlag = m_serumData.isextraframe[static_cast<uint32_t>(index)];
            if (extraFlag && extraFlag[0] != 0) {
                serumFrame = m_serumData.cframes_v2_extra[static_cast<uint32_t>(index)];
            }
        } else {
            serumFrame = m_serumData.cframes_v2[static_cast<uint32_t>(index)];
            const uint8_t* extraFlag = m_serumData.isextraframe[static_cast<uint32_t>(index)];
            if (extraFlag && extraFlag[0] != 0) {
                serumFrameExtra = m_serumData.cframes_v2_extra[static_cast<uint32_t>(index)];
            }
        }
    }
    if (hasOverride) {
        frameImage = &overrideColorized;
    } else if (useHd && index >= 0 && index < static_cast<int>(m_frameExtraFrames.size())) {
        hdFrame = m_frameExtraFrames[static_cast<std::size_t>(index)];
        if (!hdFrame.empty()) {
            frameImage = &hdFrame;
        }
    }
    if (!frameImage && !serumFrame) {
        const_cast<MainWindow*>(this)->logLine(
            QString("renderFrameWithSerum: missing frame data index=%1 hd=%2 override=%3")
                .arg(index)
                .arg(useHd ? "true" : "false")
                .arg(hasOverride ? "true" : "false"));
        frameImage = m_frameStore->at(index);
    }
    const bool useLocalColorized = hasOverride || (frameImage && !frameImage->empty());
    if (!useLocalColorized && !serumFrame) {
        const_cast<MainWindow*>(this)->logLine(
            QString("renderFrameWithSerum: no colorized data index=%1 hd=%2")
                .arg(index)
                .arg(useHd ? "true" : "false"));
        return cv::Mat();
    }

    cv::Mat reference = buildOriginalPreviewForIndex(index);
    if (reference.type() != CV_8UC1) {
        cv::Mat gray;
        cv::cvtColor(EnsureBgr(reference), gray, cv::COLOR_BGR2GRAY);
        reference = gray;
    }
    if (reference.empty()) {
        const_cast<MainWindow*>(this)->logLine(
            QString("renderFrameWithSerum: empty original index=%1 hd=%2")
                .arg(index)
                .arg(useHd ? "true" : "false"));
        return EnsureBgr(*frameImage);
    }
    if (m_noColors > 0 && m_noColors < 64) {
        cv::Mat clamped = reference.clone();
        const uint8_t maxValue = static_cast<uint8_t>(m_noColors - 1);
        for (int y = 0; y < clamped.rows; ++y) {
            uint8_t* row = clamped.ptr<uint8_t>(y);
            for (int x = 0; x < clamped.cols; ++x) {
                if (row[x] > maxValue) {
                    row[x] = maxValue;
                }
            }
        }
        reference = clamped;
    }

    const int baseWidth = reference.cols;
    const int baseHeight = reference.rows;
    const int outWidth = (serumFrame && !useLocalColorized)
        ? (useHd ? static_cast<int>(m_serumData.fwidth_extra) : static_cast<int>(m_serumData.fwidth))
        : (frameImage ? frameImage->cols : 0);
    const int outHeight = (serumFrame && !useLocalColorized)
        ? (useHd ? static_cast<int>(m_serumData.fheight_extra) : static_cast<int>(m_serumData.fheight))
        : (frameImage ? frameImage->rows : 0);

    std::vector<uint16_t> frame565;
    std::vector<uint16_t> frame565Extra;
    if (useLocalColorized && frameImage) {
        frame565 = ConvertBgrMatToRgb565(*frameImage);
    }
    if (!useHd && !hasOverride) {
        const cv::Mat* extra = (index >= 0 && index < static_cast<int>(m_frameExtraFrames.size()))
            ? &m_frameExtraFrames[static_cast<std::size_t>(index)]
            : nullptr;
        if (extra && !extra->empty()) {
            frame565Extra = ConvertBgrMatToRgb565(*extra);
        }
    }

    SerumEditorFrameView frameView;
    frameView.original = reference.data;
    if (useLocalColorized) {
        frameView.colorized = useHd ? nullptr : frame565.data();
        frameView.colorized_extra = useHd
            ? frame565.data()
            : (frame565Extra.empty() ? nullptr : frame565Extra.data());
    } else {
        frameView.colorized = useHd ? nullptr : serumFrame;
        frameView.colorized_extra = useHd ? serumFrame : (serumFrameExtra ? serumFrameExtra : nullptr);
    }

    const uint32_t frameId = static_cast<uint32_t>(index);
    const cv::Mat* localDynamicMap = (index >= 0 && index < static_cast<int>(m_frameDynamicMaskMaps.size()))
        ? &m_frameDynamicMaskMaps[static_cast<std::size_t>(index)]
        : nullptr;
    const cv::Mat* localDynamicMapExtra = (index >= 0 && index < static_cast<int>(m_frameDynamicMaskMapsX.size()))
        ? &m_frameDynamicMaskMapsX[static_cast<std::size_t>(index)]
        : nullptr;
    const bool hasLocalDynamicMap = localDynamicMap && !localDynamicMap->empty();
    const bool hasLocalDynamicMapExtra = localDynamicMapExtra && !localDynamicMapExtra->empty();
    const bool hasLocalDynamicColors = index >= 0 &&
        index < static_cast<int>(m_frameDynamicColors.size()) &&
        !m_frameDynamicColors[static_cast<std::size_t>(index)].empty();
    const uint8_t* serumDynamask = nullptr;
    const uint8_t* serumDynamaskExtra = nullptr;
    const uint16_t* serumDynaCols = nullptr;
    const uint16_t* serumDynaColsExtra = nullptr;
    if (useSerumData) {
        if (!hasLocalDynamicMap && m_serumData.dynamasks.hasData(frameId)) {
            serumDynamask = m_serumData.dynamasks[frameId];
        }
        if (!hasLocalDynamicMapExtra && m_serumData.dynamasks_extra.hasData(frameId)) {
            serumDynamaskExtra = m_serumData.dynamasks_extra[frameId];
        }
        if (!hasLocalDynamicColors && m_serumData.dyna4cols_v2.hasData(frameId)) {
            serumDynaCols = m_serumData.dyna4cols_v2[frameId];
        }
        if (!hasLocalDynamicColors && m_serumData.dyna4cols_v2_extra.hasData(frameId)) {
            serumDynaColsExtra = m_serumData.dyna4cols_v2_extra[frameId];
        }
    }
    if (serumDynamask) {
        frameView.dynamask = serumDynamask;
    } else if (localDynamicMap) {
        frameView.dynamask = localDynamicMap->data;
    }
    if (serumDynamaskExtra) {
        frameView.dynamask_extra = serumDynamaskExtra;
    } else if (localDynamicMapExtra) {
        frameView.dynamask_extra = localDynamicMapExtra->data;
    }
    if (serumDynaCols) {
        frameView.dyna4cols = serumDynaCols;
    }
    if (serumDynaColsExtra) {
        frameView.dyna4cols_extra = serumDynaColsExtra;
    }
    if (useHd && serumDynaCols && !serumDynaColsExtra) {
        frameView.dyna4cols_extra = serumDynaCols;
    }
    if (!serumDynaCols && index >= 0 && index < static_cast<int>(m_frameDynamicColors.size())) {
        const std::vector<uint16_t>& colors = m_frameDynamicColors[static_cast<std::size_t>(index)];
        if (!colors.empty()) {
            frameView.dyna4cols = colors.data();
            if (!serumDynaColsExtra) {
                frameView.dyna4cols_extra = colors.data();
            }
        }
    }
    cv::Mat scaledDynamicMask;
    if (useHd) {
        const cv::Mat* hdMap = (index >= 0 && index < static_cast<int>(m_frameDynamicMaskMapsX.size()))
            ? &m_frameDynamicMaskMapsX[static_cast<std::size_t>(index)]
            : nullptr;
        const cv::Mat* sdMap = (index >= 0 && index < static_cast<int>(m_frameDynamicMaskMaps.size()))
            ? &m_frameDynamicMaskMaps[static_cast<std::size_t>(index)]
            : nullptr;
        if ((!hdMap || hdMap->empty()) && sdMap && !sdMap->empty()) {
            cv::resize(*sdMap, scaledDynamicMask, cv::Size(outWidth, outHeight), 0.0, 0.0, cv::INTER_NEAREST);
            frameView.dynamask_extra = scaledDynamicMask.data;
        } else if ((!hdMap || hdMap->empty()) && (!sdMap || sdMap->empty()) &&
                   serumDynamask && m_serumData.fwidth > 0 && m_serumData.fheight > 0) {
            cv::Mat serumMap(static_cast<int>(m_serumData.fheight),
                             static_cast<int>(m_serumData.fwidth),
                             CV_8UC1,
                             const_cast<uint8_t*>(serumDynamask));
            cv::resize(serumMap, scaledDynamicMask, cv::Size(outWidth, outHeight),
                       0.0, 0.0, cv::INTER_NEAREST);
            frameView.dynamask_extra = scaledDynamicMask.data;
        }
    }

    uint16_t backgroundId = 0xffff;
    if (m_showBackgroundLayer) {
        if (index >= 0 && index < static_cast<int>(m_frameBackgroundIds.size())) {
            backgroundId = m_frameBackgroundIds[static_cast<std::size_t>(index)];
        } else if (useSerumData && m_serumData.backgroundIDs.hasData(frameId)) {
            const uint16_t* bg = m_serumData.backgroundIDs[frameId];
            backgroundId = bg ? bg[0] : 0xffff;
        }
    }
    frameView.background_id = backgroundId;
    const cv::Mat* localBgMask = (index >= 0 && index < static_cast<int>(m_frameBackgroundMasks.size()))
        ? &m_frameBackgroundMasks[static_cast<std::size_t>(index)]
        : nullptr;
    const cv::Mat* localBgMaskExtra = (index >= 0 && index < static_cast<int>(m_frameBackgroundMasksX.size()))
        ? &m_frameBackgroundMasksX[static_cast<std::size_t>(index)]
        : nullptr;
    const bool hasLocalBgMask = localBgMask && !localBgMask->empty();
    const bool hasLocalBgMaskExtra = localBgMaskExtra && !localBgMaskExtra->empty();
    const uint8_t* serumBgMask = nullptr;
    const uint8_t* serumBgMaskExtra = nullptr;
    if (useSerumData && backgroundId != 0xffff) {
        if (!hasLocalBgMask && m_serumData.backgroundmask.hasData(frameId)) {
            serumBgMask = m_serumData.backgroundmask[frameId];
        }
        if (!hasLocalBgMaskExtra && m_serumData.backgroundmask_extra.hasData(frameId)) {
            serumBgMaskExtra = m_serumData.backgroundmask_extra[frameId];
        }
    }
    if (serumBgMask) {
        frameView.background_mask = serumBgMask;
    } else if (localBgMask) {
        frameView.background_mask = localBgMask->data;
    }
    if (serumBgMaskExtra) {
        frameView.background_mask_extra = serumBgMaskExtra;
    } else if (localBgMaskExtra) {
        frameView.background_mask_extra = localBgMaskExtra->data;
    }

    std::vector<uint16_t> background565;
    std::vector<uint16_t> background565Extra;
    if (backgroundId != 0xffff) {
        const cv::Mat* backgroundOverride = nullptr;
        if (useSerumData && m_backgroundStore &&
            m_backgroundStore->isDirty(static_cast<int>(backgroundId))) {
            backgroundOverride = m_backgroundStore->peek(static_cast<int>(backgroundId));
        }
        if (backgroundOverride && !backgroundOverride->empty()) {
            background565 = ConvertBgrMatToRgb565(*backgroundOverride);
            frameView.background_frame = background565.data();
        } else if (useSerumData &&
                   backgroundId < m_serumData.nbackgrounds &&
                   m_serumData.backgroundframes_v2.hasData(backgroundId)) {
            frameView.background_frame = m_serumData.backgroundframes_v2[backgroundId];
        } else if (m_backgroundStore && backgroundId < m_backgroundStore->count()) {
            const cv::Mat* background = m_backgroundStore->at(static_cast<int>(backgroundId));
            if (background && !background->empty()) {
                background565 = ConvertBgrMatToRgb565(*background);
                frameView.background_frame = background565.data();
            }
        }
        if (useSerumData &&
            backgroundId < m_serumData.nbackgrounds &&
            m_serumData.backgroundframes_v2_extra.hasData(backgroundId)) {
            frameView.background_frame_extra = m_serumData.backgroundframes_v2_extra[backgroundId];
        } else if (backgroundId < m_backgroundFramesX.size() &&
                   !m_backgroundFramesX[static_cast<std::size_t>(backgroundId)].empty()) {
            background565Extra = ConvertBgrMatToRgb565(m_backgroundFramesX[static_cast<std::size_t>(backgroundId)]);
            frameView.background_frame_extra = background565Extra.data();
        }
    }
    cv::Mat scaledBackgroundMask;
    cv::Mat scaledBackgroundFrame;
    if (useHd && backgroundId != 0xffff) {
        const cv::Mat* hdMask = (index >= 0 && index < static_cast<int>(m_frameBackgroundMasksX.size()))
            ? &m_frameBackgroundMasksX[static_cast<std::size_t>(index)]
            : nullptr;
        const cv::Mat* sdMask = (index >= 0 && index < static_cast<int>(m_frameBackgroundMasks.size()))
            ? &m_frameBackgroundMasks[static_cast<std::size_t>(index)]
            : nullptr;
        if (!frameView.background_mask_extra) {
            cv::Mat maskSource;
            if (sdMask && !sdMask->empty()) {
                maskSource = *sdMask;
            } else if (serumBgMask) {
                maskSource = cv::Mat(static_cast<int>(m_serumData.fheight),
                                     static_cast<int>(m_serumData.fwidth),
                                     CV_8UC1,
                                     const_cast<uint8_t*>(serumBgMask));
            }
            if (!maskSource.empty()) {
                cv::resize(maskSource, scaledBackgroundMask, cv::Size(outWidth, outHeight),
                           0.0, 0.0, cv::INTER_NEAREST);
                frameView.background_mask_extra = scaledBackgroundMask.data;
            }
        }
        if (!frameView.background_frame_extra) {
            cv::Mat backgroundSource;
            if (useSerumData && m_backgroundStore &&
                m_backgroundStore->isDirty(static_cast<int>(backgroundId))) {
                const cv::Mat* sdBackground = m_backgroundStore->peek(static_cast<int>(backgroundId));
                if (sdBackground && !sdBackground->empty()) {
                    backgroundSource = *sdBackground;
                }
            } else if (frameView.background_frame) {
                backgroundSource = ConvertRgb565ToBgrMat(frameView.background_frame,
                                                         static_cast<int>(m_serumData.fwidth),
                                                         static_cast<int>(m_serumData.fheight));
            } else if (m_backgroundStore && backgroundId < m_backgroundStore->count()) {
                const cv::Mat* sdBackground = m_backgroundStore->at(static_cast<int>(backgroundId));
                if (sdBackground && !sdBackground->empty()) {
                    backgroundSource = *sdBackground;
                }
            }
            if (!backgroundSource.empty()) {
                cv::resize(backgroundSource, scaledBackgroundFrame, cv::Size(outWidth, outHeight), 0.0, 0.0, cv::INTER_NEAREST);
                background565Extra = ConvertBgrMatToRgb565(scaledBackgroundFrame);
                frameView.background_frame_extra = background565Extra.data();
            }
        }
    }

    const uint8_t* frameSprites = nullptr;
    const uint16_t* frameSpriteBBoxes = nullptr;
    const std::size_t spriteBase = static_cast<std::size_t>(index) * MAX_SPRITES_PER_FRAME;
    const bool hasLocalFrameSprites = spriteBase + MAX_SPRITES_PER_FRAME <= m_frameSpriteAssignments.size();
    if (hasLocalFrameSprites) {
        frameSprites = m_frameSpriteAssignments.data() + spriteBase;
    } else if (useSerumData && m_serumData.framesprites.hasData(frameId)) {
        frameSprites = m_serumData.framesprites[frameId];
    }
    const std::size_t spriteBbBase = static_cast<std::size_t>(index) * MAX_SPRITES_PER_FRAME * 4;
    const bool hasLocalFrameSpriteBBoxes = spriteBbBase + MAX_SPRITES_PER_FRAME * 4 <= m_frameSpriteBBoxes.size();
    if (hasLocalFrameSpriteBBoxes) {
        frameSpriteBBoxes = m_frameSpriteBBoxes.data() + spriteBbBase;
    } else if (useSerumData && m_serumData.framespriteBB.hasData(frameId)) {
        frameSpriteBBoxes = m_serumData.framespriteBB[frameId];
    }
    frameView.frame_sprites = frameSprites;
    frameView.frame_sprite_bboxes = frameSpriteBBoxes;

    std::size_t spriteCount = m_spriteOriginals.size();
    if (useSerumData) {
        spriteCount = std::max(spriteCount, static_cast<std::size_t>(m_serumData.nsprites));
    }
    std::vector<SerumEditorSpriteView> spriteViews(spriteCount);
    std::vector<std::vector<uint16_t>> sprite565(spriteCount);
    std::vector<std::vector<uint16_t>> sprite565Extra(spriteCount);
    const QSize spriteBaseSize = serumSpriteBaseSize();
    const int spriteSrcWidth = spriteBaseSize.width();
    const int spriteSrcHeight = spriteBaseSize.height();
    const bool spriteNeedsPadding =
        spriteSrcWidth != MAX_SPRITE_WIDTH || spriteSrcHeight != MAX_SPRITE_HEIGHT;
    const int spriteCopyWidth = std::min(spriteSrcWidth, MAX_SPRITE_WIDTH);
    const int spriteCopyHeight = std::min(spriteSrcHeight, MAX_SPRITE_HEIGHT);
    if (frameSprites && spriteCount > 0) {
        for (int slot = 0; slot < MAX_SPRITES_PER_FRAME; ++slot) {
            const uint8_t spriteId = frameSprites[slot];
            if (spriteId == 255 || spriteId >= spriteCount) {
                continue;
            }
            SerumEditorSpriteView& view = spriteViews[spriteId];
            if (view.original) {
                continue;
            }
            const std::size_t spriteIndex = static_cast<std::size_t>(spriteId);
            const uint32_t spriteId32 = static_cast<uint32_t>(spriteId);
            const uint8_t* serumOriginal = useSerumData ? serumSpriteOriginalData(spriteId32) : nullptr;
            if (spriteIndex < m_spriteOriginals.size() &&
                !m_spriteOriginals[spriteIndex].empty()) {
                view.original = m_spriteOriginals[spriteIndex].data;
            } else if (spriteNeedsPadding && useSerumData) {
                if (cv::Mat* padded = const_cast<MainWindow*>(this)->ensureSpriteOriginalLocal(static_cast<int>(spriteId))) {
                    if (!padded->empty()) {
                        view.original = padded->data;
                    }
                }
            }
            if (!view.original && serumOriginal) {
                view.original = serumOriginal;
            }
            const cv::Mat* spriteOverride = nullptr;
            if (useSerumData && m_spriteStore &&
                m_spriteStore->isDirty(static_cast<int>(spriteId))) {
                spriteOverride = m_spriteStore->peek(static_cast<int>(spriteId));
            }
            if (spriteOverride && !spriteOverride->empty()) {
                sprite565[spriteIndex] = ConvertBgrMatToRgb565(*spriteOverride);
                view.colored = sprite565[spriteIndex].data();
            } else if (spriteIndex < m_spriteColored.size()) {
                const cv::Mat& colored = m_spriteColored[spriteIndex];
                if (!colored.empty()) {
                    sprite565[spriteIndex] = ConvertBgrMatToRgb565(colored);
                    view.colored = sprite565[spriteIndex].data();
                }
            }
            if (spriteIndex < m_spriteColoredX.size()) {
                const cv::Mat& colored = m_spriteColoredX[spriteIndex];
                if (!colored.empty()) {
                    sprite565Extra[spriteIndex] = ConvertBgrMatToRgb565(colored);
                    view.colored_extra = sprite565Extra[spriteIndex].data();
                }
            }
            if (spriteIndex < m_spriteMasksX.size() &&
                !m_spriteMasksX[spriteIndex].empty()) {
                view.mask_extra = m_spriteMasksX[spriteIndex].data;
            }
            if (spriteIndex < m_spriteDynamicMasks.size() &&
                !m_spriteDynamicMasks[spriteIndex].empty()) {
                view.dynasprite_mask = m_spriteDynamicMasks[spriteIndex].data;
            } else if (spriteNeedsPadding && useSerumData) {
                if (cv::Mat* padded = const_cast<MainWindow*>(this)->ensureSpriteDynamicMaskLocal(static_cast<int>(spriteId))) {
                    if (!padded->empty()) {
                        view.dynasprite_mask = padded->data;
                    }
                }
            }
            if (spriteIndex < m_spriteDynamicMasksX.size() &&
                !m_spriteDynamicMasksX[spriteIndex].empty()) {
                view.dynasprite_mask_extra = m_spriteDynamicMasksX[spriteIndex].data;
            }
            if (spriteIndex < m_spriteDynamicColors.size()) {
                const auto& dynCols = m_spriteDynamicColors[spriteIndex];
                if (!dynCols.empty()) {
                    view.dynasprite_cols = dynCols.data();
                }
            }
            if (spriteIndex < m_spriteDynamicColorsX.size()) {
                const auto& dynCols = m_spriteDynamicColorsX[spriteIndex];
                if (!dynCols.empty()) {
                    view.dynasprite_cols_extra = dynCols.data();
                }
            }
            if (spriteIndex < m_spriteShapeModes.size()) {
                view.shape_mode = m_spriteShapeModes[spriteIndex];
            }
            const std::size_t detAreaOffset = spriteIndex * MAX_SPRITE_DETECT_AREAS * 4;
            if (detAreaOffset + MAX_SPRITE_DETECT_AREAS * 4 <= m_spriteDetAreas.size()) {
                view.det_areas = m_spriteDetAreas.data() + detAreaOffset;
            }
            const std::size_t detDwordOffset = spriteIndex * MAX_SPRITE_DETECT_AREAS;
            if (detDwordOffset + MAX_SPRITE_DETECT_AREAS <= m_spriteDetDwords.size()) {
                view.det_dwords = m_spriteDetDwords.data() + detDwordOffset;
            }
            if (detDwordOffset + MAX_SPRITE_DETECT_AREAS <= m_spriteDetDwordPos.size()) {
                view.det_dword_pos = m_spriteDetDwordPos.data() + detDwordOffset;
            }
            if (useSerumData) {
                if (!view.colored && m_serumData.spritecolored.elementCount() > 0) {
                    const uint16_t* data = m_serumData.spritecolored[spriteId32];
                    if (data && spriteNeedsPadding && spriteCopyWidth > 0 && spriteCopyHeight > 0) {
                        sprite565[spriteIndex].assign(
                            static_cast<std::size_t>(MAX_SPRITE_WIDTH) * MAX_SPRITE_HEIGHT,
                            static_cast<uint16_t>(0));
                        for (int y = 0; y < spriteCopyHeight; ++y) {
                            const std::size_t srcOffset = static_cast<std::size_t>(y) * spriteSrcWidth;
                            const std::size_t dstOffset = static_cast<std::size_t>(y) * MAX_SPRITE_WIDTH;
                            for (int x = 0; x < spriteCopyWidth; ++x) {
                                sprite565[spriteIndex][dstOffset + static_cast<std::size_t>(x)] =
                                    data[srcOffset + static_cast<std::size_t>(x)];
                            }
                        }
                    } else {
                        const std::size_t count = m_serumData.spritecolored.elementCount();
                        sprite565[spriteIndex].assign(data, data + count);
                    }
                    if (!sprite565[spriteIndex].empty()) {
                        view.colored = sprite565[spriteIndex].data();
                    }
                }
                if (!view.colored_extra && m_serumData.spritecolored_extra.hasData(spriteId32)) {
                    const uint16_t* data = m_serumData.spritecolored_extra[spriteId32];
                    const std::size_t count = m_serumData.spritecolored_extra.elementCount();
                    if (data && count > 0) {
                        sprite565Extra[spriteIndex].assign(data, data + count);
                        view.colored_extra = sprite565Extra[spriteIndex].data();
                    }
                }
                if (!view.mask_extra && m_serumData.spritemask_extra.hasData(spriteId32)) {
                    view.mask_extra = m_serumData.spritemask_extra[spriteId32];
                }
                if (!view.dynasprite_mask && m_serumData.dynaspritemasks.hasData(spriteId32)) {
                    view.dynasprite_mask = m_serumData.dynaspritemasks[spriteId32];
                }
                if (!view.dynasprite_mask_extra && m_serumData.dynaspritemasks_extra.hasData(spriteId32)) {
                    view.dynasprite_mask_extra = m_serumData.dynaspritemasks_extra[spriteId32];
                }
                if (!view.dynasprite_cols && m_serumData.dynasprite4cols.hasData(spriteId32)) {
                    view.dynasprite_cols = m_serumData.dynasprite4cols[spriteId32];
                }
                if (!view.dynasprite_cols_extra && m_serumData.dynasprite4cols_extra.hasData(spriteId32)) {
                    view.dynasprite_cols_extra = m_serumData.dynasprite4cols_extra[spriteId32];
                }
                if (view.shape_mode == 0 && m_serumData.sprshapemode.hasData(spriteId32)) {
                    const uint8_t* mode = m_serumData.sprshapemode[spriteId32];
                    view.shape_mode = mode ? mode[0] : 0;
                }
                if (!view.det_areas && m_serumData.spritedetareas.hasData(spriteId32)) {
                    view.det_areas = m_serumData.spritedetareas[spriteId32];
                }
                if (!view.det_dwords && m_serumData.spritedetdwords.hasData(spriteId32)) {
                    view.det_dwords = m_serumData.spritedetdwords[spriteId32];
                }
                if (!view.det_dword_pos && m_serumData.spritedetdwordpos.hasData(spriteId32)) {
                    view.det_dword_pos = m_serumData.spritedetdwordpos[spriteId32];
                }
            }
        }
    }

    SerumEditorDataView dataView;
    dataView.width = static_cast<uint32_t>(baseWidth);
    dataView.height = static_cast<uint32_t>(baseHeight);
    dataView.width_extra = static_cast<uint32_t>(outWidth);
    dataView.height_extra = static_cast<uint32_t>(outHeight);
    dataView.nocolors = m_noColors;
    dataView.nsprites = static_cast<uint32_t>(spriteCount);
    dataView.sprites = spriteViews.data();

    SerumEditorSpriteMatch matches[MAX_SPRITES_PER_FRAME];
    std::memset(matches, 0, sizeof(matches));
    uint8_t matchCount = SerumEditor_MatchSprites(&dataView, &frameView, matches, MAX_SPRITES_PER_FRAME);

    std::vector<uint16_t> out565(static_cast<std::size_t>(outWidth) * outHeight);
    if (!SerumEditor_RenderFrame(&dataView, &frameView, matches, matchCount, useHd, out565.data())) {
        if (useLocalColorized && frameImage) {
            return EnsureBgr(*frameImage);
        }
        if (serumFrame) {
            return ConvertRgb565ToBgrMat(serumFrame, outWidth, outHeight);
        }
        return cv::Mat();
    }
    cv::Mat output = ConvertRgb565ToBgrMat(out565.data(), outWidth, outHeight);
    if (m_showBackgroundLayer && backgroundId != 0xffff && !hasOverride) {
        const uint8_t* bgMaskData = useHd ? frameView.background_mask_extra : frameView.background_mask;
        if (bgMaskData) {
            cv::Mat maskView(outHeight, outWidth, CV_8UC1, const_cast<uint8_t*>(bgMaskData));
            cv::Mat refView = reference;
            if (refView.size() != output.size()) {
                cv::resize(refView, refView, output.size(), 0.0, 0.0, cv::INTER_NEAREST);
            }
            cv::Mat colorized;
            if (hasOverride) {
                colorized = EnsureBgr(overrideColorized);
            } else if (frameImage && !frameImage->empty()) {
                colorized = EnsureBgr(*frameImage);
            } else if (serumFrame) {
                colorized = ConvertRgb565ToBgrMat(serumFrame, outWidth, outHeight);
            }
            if (colorized.size() != output.size()) {
                cv::resize(colorized, colorized, output.size(), 0.0, 0.0, cv::INTER_NEAREST);
            }
            for (int y = 0; y < output.rows; ++y) {
                cv::Vec3b* outRow = output.ptr<cv::Vec3b>(y);
                const cv::Vec3b* colorRow = colorized.ptr<cv::Vec3b>(y);
                const uint8_t* refRow = refView.ptr<uint8_t>(y);
                const uint8_t* maskRow = maskView.ptr<uint8_t>(y);
                for (int x = 0; x < output.cols; ++x) {
                    if (refRow[x] != 0 || maskRow[x] == 0) {
                        continue;
                    }
                    const cv::Vec3b color = colorRow[x];
                    if (color[0] == 0 && color[1] == 0 && color[2] == 0) {
                        continue;
                    }
                    outRow[x] = color;
                }
            }
        }
    }
    return output;
}

cv::Mat MainWindow::applyDynamicColors(int index, const cv::Mat& frame, bool useHd) const
{
    cv::Mat output = EnsureBgr(frame);
    if (output.empty() || index < 0 || index >= static_cast<int>(m_frameDynamicColors.size())) {
        return output;
    }
    int stride = 0;
    const uint16_t* colors = frameDynamicColorsData(index, useHd, &stride);
    if (stride <= 0) {
        return output;
    }
    const cv::Mat* map = nullptr;
    if (useHd && index < static_cast<int>(m_frameDynamicMaskMapsX.size())) {
        map = &m_frameDynamicMaskMapsX[static_cast<std::size_t>(index)];
    }
    if ((!map || map->empty()) && index < static_cast<int>(m_frameDynamicMaskMaps.size())) {
        map = &m_frameDynamicMaskMaps[static_cast<std::size_t>(index)];
    }
    if (!map || map->empty()) {
        return output;
    }
    cv::Mat mapScaled;
    if (map->size() != output.size()) {
        cv::resize(*map, mapScaled, output.size(), 0.0, 0.0, cv::INTER_NEAREST);
    } else {
        mapScaled = *map;
    }
    cv::Mat ref = buildReferenceForSize(index, output.size());
    if (ref.empty()) {
        return output;
    }
    if (ref.type() != CV_8UC1) {
        cv::Mat gray;
        cv::cvtColor(EnsureBgr(ref), gray, cv::COLOR_BGR2GRAY);
        ref = gray;
    }
    int levels = m_noColors > 0 ? static_cast<int>(m_noColors) : 64;
    levels = std::max(1, levels);
    const int maxLevel = std::max(1, levels - 1);
    for (int y = 0; y < output.rows; ++y) {
        cv::Vec3b* row = output.ptr<cv::Vec3b>(y);
        const uint8_t* mrow = mapScaled.ptr<uint8_t>(y);
        const uint8_t* rrow = ref.ptr<uint8_t>(y);
        for (int x = 0; x < output.cols; ++x) {
            const int setId = static_cast<int>(mrow[x]);
            if (setId < 0 || setId >= MAX_DYNA_SETS_PER_FRAMEN) {
                continue;
            }
            int slot = static_cast<int>(rrow[x]);
            if (levels > stride) {
                slot = (slot * (stride - 1) + maxLevel / 2) / maxLevel;
            }
            slot = std::clamp(slot, 0, stride - 1);
            const std::size_t offset = static_cast<std::size_t>(setId) * stride +
                static_cast<std::size_t>(slot);
            const std::size_t size =
                static_cast<std::size_t>(MAX_DYNA_SETS_PER_FRAMEN) * static_cast<std::size_t>(stride);
            if (colors && offset < size) {
                row[x] = Rgb565ToBgr(colors[offset]);
            }
        }
    }
    return output;
}

cv::Mat MainWindow::applySpritesToFrame(int index,
                                        const cv::Mat& frame,
                                        bool useHd,
                                        bool drawOutline) const
{
    (void)drawOutline;
    cv::Mat output = EnsureBgr(frame);
    if (output.empty() || index < 0 || !m_spriteStore) {
        return output;
    }
    if (m_frameSpriteAssignments.empty() || m_spriteRects.empty()) {
        return output;
    }
    const std::size_t baseSlot = static_cast<std::size_t>(index) * MAX_SPRITES_PER_FRAME;
    if (baseSlot >= m_frameSpriteAssignments.size()) {
        return output;
    }
    const int frameWidth = output.cols;
    const int frameHeight = output.rows;
    for (int slot = 0; slot < MAX_SPRITES_PER_FRAME; ++slot) {
        const std::size_t slotIndex = baseSlot + static_cast<std::size_t>(slot);
        if (slotIndex >= m_frameSpriteAssignments.size()) {
            break;
        }
        const uint8_t spriteId = m_frameSpriteAssignments[slotIndex];
        if (spriteId == 255) {
            continue;
        }
        const int spriteIndex = static_cast<int>(spriteId);
        const cv::Mat* spriteImage = nullptr;
        if (useHd && spriteIndex >= 0) {
            spriteImage = const_cast<MainWindow*>(this)->ensureHdSpriteLocal(spriteIndex);
        } else if (spriteIndex >= 0 && spriteIndex < m_spriteStore->count()) {
            spriteImage = m_spriteStore->at(spriteIndex);
        }
        if (!spriteImage || spriteImage->empty()) {
            continue;
        }
        cv::Mat spriteDisplay = useHd
            ? EnsureBgr(*spriteImage)
            : applySpriteDynamicColors(spriteIndex, *spriteImage);

        const cv::Mat* spriteMask = nullptr;
        if (useHd && spriteIndex >= 0) {
            spriteMask = const_cast<MainWindow*>(this)->ensureHdSpriteMaskLocal(spriteIndex);
        } else if (spriteIndex >= 0) {
            spriteMask = const_cast<MainWindow*>(this)->ensureSpriteOriginalLocal(spriteIndex);
        }
        if (!spriteMask || spriteMask->empty()) {
            continue;
        }
        const std::size_t rectBase = static_cast<std::size_t>(spriteIndex) * 4;
        if (rectBase + 3 >= m_spriteRects.size()) {
            continue;
        }
        const uint16_t rawLeft = m_spriteRects[rectBase];
        if (rawLeft == 0xffff) {
            continue;
        }
        const bool rectExtra = (rawLeft & 0x8000) != 0;
        int left = static_cast<int>(rawLeft & 0x7fff);
        int top = static_cast<int>(m_spriteRects[rectBase + 1]);
        int right = static_cast<int>(m_spriteRects[rectBase + 2]);
        int bottom = static_cast<int>(m_spriteRects[rectBase + 3]);
        if (useHd && !rectExtra) {
            left *= 2;
            top *= 2;
            right = right * 2 + 1;
            bottom = bottom * 2 + 1;
        } else if (!useHd && rectExtra) {
            left /= 2;
            top /= 2;
            right /= 2;
            bottom /= 2;
        }
        if (right < left || bottom < top) {
            continue;
        }
        const int rectWidth = right - left + 1;
        const int rectHeight = bottom - top + 1;
        bool mirrorX = false;
        bool mirrorY = false;
        const std::size_t mirrorBase = static_cast<std::size_t>(spriteIndex) * 2;
        if (mirrorBase + 1 < m_spriteRectMirror.size()) {
            mirrorX = m_spriteRectMirror[mirrorBase] != 0;
            mirrorY = m_spriteRectMirror[mirrorBase + 1] != 0;
        }
        for (int y = 0; y < rectHeight; ++y) {
            const int frameY = top + y;
            if (frameY < 0 || frameY >= frameHeight) {
                continue;
            }
            const int spriteY = mirrorY ? (rectHeight - 1 - y) : y;
            if (spriteY < 0 || spriteY >= spriteMask->rows || spriteY >= spriteDisplay.rows) {
                continue;
            }
            const uint8_t* maskRow = spriteMask->ptr<uint8_t>(spriteY);
            const cv::Vec3b* spriteRow = spriteDisplay.ptr<cv::Vec3b>(spriteY);
            cv::Vec3b* outRow = output.ptr<cv::Vec3b>(frameY);
            for (int x = 0; x < rectWidth; ++x) {
                const int frameX = left + x;
                if (frameX < 0 || frameX >= frameWidth) {
                    continue;
                }
                const int spriteX = mirrorX ? (rectWidth - 1 - x) : x;
                if (spriteX < 0 || spriteX >= spriteMask->cols || spriteX >= spriteDisplay.cols) {
                    continue;
                }
                if (maskRow[spriteX] == 255) {
                    continue;
                }
                outRow[frameX] = spriteRow[spriteX];
            }
        }
    }
    return output;
}

cv::Mat MainWindow::applySpriteDynamicColors(int index, const cv::Mat& sprite) const
{
    cv::Mat output = EnsureBgr(sprite);
    if (output.empty() || index < 0) {
        return output;
    }
    const std::vector<uint16_t>* colors = nullptr;
    if (m_useHdSprite) {
        if (index < static_cast<int>(m_spriteDynamicColorsX.size()) &&
            !m_spriteDynamicColorsX[static_cast<std::size_t>(index)].empty()) {
            colors = &m_spriteDynamicColorsX[static_cast<std::size_t>(index)];
        } else {
            colors = const_cast<MainWindow*>(this)->ensureHdSpriteDynamicColorsLocal(index);
        }
    } else {
        colors = const_cast<MainWindow*>(this)->ensureSpriteDynamicColorsLocal(index);
    }
    const int stride = colors ? spriteDynamicColorsPerSet(*colors) : 0;
    if (stride <= 0) {
        return output;
    }
    const cv::Mat* map = nullptr;
    if (m_useHdSprite) {
        if (index < static_cast<int>(m_spriteDynamicMasksX.size()) &&
            !m_spriteDynamicMasksX[static_cast<std::size_t>(index)].empty()) {
            map = &m_spriteDynamicMasksX[static_cast<std::size_t>(index)];
        } else {
            map = const_cast<MainWindow*>(this)->ensureHdSpriteDynamicMaskLocal(index);
        }
    } else {
        map = const_cast<MainWindow*>(this)->ensureSpriteDynamicMaskLocal(index);
    }
    const cv::Mat* original = const_cast<MainWindow*>(this)->ensureSpriteOriginalLocal(index);
    if (!map || map->empty() || !original || original->empty() || !colors) {
        return output;
    }
    cv::Mat mapScaled;
    if (map->size() != output.size()) {
        cv::resize(*map, mapScaled, output.size(), 0.0, 0.0, cv::INTER_NEAREST);
    } else {
        mapScaled = *map;
    }
    cv::Mat originalScaled;
    if (original->size() != output.size()) {
        cv::resize(*original, originalScaled, output.size(), 0.0, 0.0, cv::INTER_NEAREST);
    } else {
        originalScaled = *original;
    }
    int levels = m_noColors > 0 ? static_cast<int>(m_noColors) : 64;
    levels = std::max(1, levels);
    const int maxLevel = std::max(1, levels - 1);
    for (int y = 0; y < output.rows; ++y) {
        cv::Vec3b* row = output.ptr<cv::Vec3b>(y);
        const uint8_t* mrow = mapScaled.ptr<uint8_t>(y);
        const uint8_t* orow = originalScaled.ptr<uint8_t>(y);
        for (int x = 0; x < output.cols; ++x) {
            const uint8_t originalValue = orow[x];
            if (originalValue == 255) {
                continue;
            }
            const int setId = static_cast<int>(mrow[x]);
            if (setId < 0 || setId >= MAX_DYNA_SETS_PER_SPRITE || setId == 255) {
                continue;
            }
            int slot = static_cast<int>(originalValue);
            if (levels > stride) {
                slot = (slot * (stride - 1) + maxLevel / 2) / maxLevel;
            }
            slot = std::clamp(slot, 0, stride - 1);
            const std::size_t offset = static_cast<std::size_t>(setId) * stride +
                static_cast<std::size_t>(slot);
            if (offset < colors->size()) {
                row[x] = Rgb565ToBgr((*colors)[offset]);
            }
        }
    }
    return output;
}

const cv::Mat* MainWindow::activeSpriteImage(int index) const
{
    if (!m_spriteStore || index < 0) {
        return nullptr;
    }
    if (m_useHdSprite) {
        if (auto* hd = const_cast<MainWindow*>(this)->ensureHdSpriteLocal(index)) {
            return hd;
        }
    }
    return m_spriteStore->at(index);
}

cv::Mat* MainWindow::activeSpriteImageMutable(int index)
{
    if (!m_spriteStore || index < 0) {
        return nullptr;
    }
    if (m_useHdSprite) {
        if (auto* hd = ensureHdSpriteLocal(index)) {
            return hd;
        }
    }
    cv::Mat* sprite = m_spriteStore->atMutable(index);
    if (sprite && !m_serumDataLoaded) {
        if (index >= static_cast<int>(m_spriteColored.size())) {
            m_spriteColored.resize(static_cast<std::size_t>(index + 1));
        }
        m_spriteColored[static_cast<std::size_t>(index)] = *sprite;
    }
    return sprite;
}

const cv::Mat* MainWindow::spriteOriginalForDisplay(int index) const
{
    if (index < 0) {
        return nullptr;
    }
    if (index < static_cast<int>(m_spriteOriginals.size()) &&
        !m_spriteOriginals[static_cast<std::size_t>(index)].empty()) {
        return &m_spriteOriginals[static_cast<std::size_t>(index)];
    }
    return const_cast<MainWindow*>(this)->ensureSpriteOriginalLocal(index);
}

QRect MainWindow::spriteDisplayRect(int index, const cv::Mat& image) const
{
    if (image.empty()) {
        return QRect();
    }
    QRect rect = spriteDisplayContentRect(index);
    if (!rect.isValid() || rect.isEmpty()) {
        return QRect(0, 0, image.cols, image.rows);
    }
    if (m_useHdSprite) {
        const cv::Mat* hdMask = const_cast<MainWindow*>(this)->ensureHdSpriteMaskLocal(index);
        const bool hasHdMask = hdMask && !hdMask->empty();
        if (!hasHdMask) {
            rect = QRect(rect.x() * 2,
                         rect.y() * 2,
                         rect.width() * 2,
                         rect.height() * 2);
        }
    }
    return rect.intersected(QRect(0, 0, image.cols, image.rows));
}

namespace {
QRect ContentRectFromMask(const cv::Mat& mask, uint8_t emptyValue)
{
    if (mask.empty()) {
        return QRect();
    }
    int minx = mask.cols;
    int miny = mask.rows;
    int maxx = -1;
    int maxy = -1;
    for (int y = 0; y < mask.rows; ++y) {
        const uint8_t* row = mask.ptr<uint8_t>(y);
        for (int x = 0; x < mask.cols; ++x) {
            if (row[x] != emptyValue) {
                minx = std::min(minx, x);
                miny = std::min(miny, y);
                maxx = std::max(maxx, x);
                maxy = std::max(maxy, y);
            }
        }
    }
    if (maxx >= minx && maxy >= miny) {
        return QRect(minx, miny, maxx - minx + 1, maxy - miny + 1);
    }
    return QRect();
}
}

QRect MainWindow::spriteContentRect(int index) const
{
    if (index < 0) {
        return QRect();
    }
    const cv::Mat* sprite = (m_spriteStore && index >= 0) ? m_spriteStore->at(index) : nullptr;
    int width = sprite ? sprite->cols : MAX_SPRITE_WIDTH;
    int height = sprite ? sprite->rows : MAX_SPRITE_HEIGHT;
    const cv::Mat* original = spriteOriginalForDisplay(index);
    if (original && !original->empty()) {
        width = original->cols;
        height = original->rows;
        int minx = width;
        int miny = height;
        int maxx = -1;
        int maxy = -1;
        for (int y = 0; y < original->rows; ++y) {
            const uint8_t* row = original->ptr<uint8_t>(y);
            for (int x = 0; x < original->cols; ++x) {
                if (row[x] != 255) {
                    minx = std::min(minx, x);
                    miny = std::min(miny, y);
                    maxx = std::max(maxx, x);
                    maxy = std::max(maxy, y);
                }
            }
        }
        if (maxx >= minx && maxy >= miny) {
            return QRect(minx, miny, maxx - minx + 1, maxy - miny + 1);
        }
    }
    if (width <= 0 || height <= 0) {
        return QRect();
    }
    return QRect(0, 0, width, height);
}

QRect MainWindow::spriteDisplayContentRect(int index) const
{
    if (index < 0) {
        return QRect();
    }
    if (m_useHdSprite &&
        index < static_cast<int>(m_spriteMasksX.size())) {
        if (const cv::Mat* mask = const_cast<MainWindow*>(this)->ensureHdSpriteMaskLocal(index)) {
            if (!mask->empty()) {
                return ContentRectFromMask(*mask, 255);
            }
        }
    }
    return spriteContentRect(index);
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
            bg = const_cast<MainWindow*>(this)->ensureHdBackgroundLocal(static_cast<int>(bgId));
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
        m_framesCanvas->canvas()->clearTertiaryOutline();
        return;
    }
    const cv::Mat* image = activeFrameImage(index, false);
    if (!image || image->empty()) {
        logLine(QString("Frame canvas: missing image index=%1").arg(index));
        m_framesCanvas->setImage(cv::Mat());
        m_framesCanvas->canvas()->clearTertiaryOutline();
        return;
    }
    if (m_canvasRotateEnabled) {
        logLine(QString("Frame canvas: rotation enabled index=%1").arg(index));
        updateCanvasRotationFrame();
        return;
    }
    const cv::Mat* overrideFrame = nullptr;
    const bool dirty = m_frameStore && m_frameStore->isDirty(index);
    if (m_frameCanvasOverrideIndex == index && !m_frameCanvasOverrideImage.empty()) {
        overrideFrame = &m_frameCanvasOverrideImage;
    }
    if (overrideFrame && !overrideFrame->empty()) {
        int nonzero = 0;
        for (int y = 0; y < overrideFrame->rows; ++y) {
            const cv::Vec3b* row = overrideFrame->ptr<cv::Vec3b>(y);
            for (int x = 0; x < overrideFrame->cols; ++x) {
                const cv::Vec3b px = row[x];
                if (px[0] || px[1] || px[2]) {
                    ++nonzero;
                }
            }
        }
        logLine(QString("Frame canvas: override stats index=%1 size=%2x%3 nonzero=%4")
                    .arg(index)
                    .arg(overrideFrame->cols)
                    .arg(overrideFrame->rows)
                    .arg(nonzero));
    }
    logLine(QString("Frame canvas: render index=%1 hd=%2 dirty=%3 override=%4")
                .arg(index)
                .arg(m_useHdFrame ? "true" : "false")
                .arg(dirty ? "true" : "false")
                .arg(overrideFrame ? "true" : "false"));
    const cv::Mat composed = overrideFrame
        ? renderFrameWithSerum(index, m_useHdFrame, *overrideFrame)
        : renderFrameWithSerum(index, m_useHdFrame);
    if (composed.empty()) {
        logLine(QString("Frame canvas: composed empty index=%1 hd=%2")
                    .arg(index)
                    .arg(m_useHdFrame ? "true" : "false"));
    }
    setFrameCanvasFromComposed(index, composed);
}

void MainWindow::setFrameCanvasFromComposed(int index, const cv::Mat& composed)
{
    if (!m_framesCanvas || index < 0) {
        return;
    }
    cv::Mat reference = buildOriginalPreviewForIndex(index);
    cv::Mat original = buildOriginalFrame(reference);
    if (original.empty() || !m_showOriginalFrame) {
        m_framesCanvas->setImage(composed);
        m_framesCanvas->canvas()->setGridSegments(0, 0, 0);
        m_framesCanvas->canvas()->setGridScales(1, 1);
        m_framesCanvas->canvas()->setGridRegions(QRect(), QRect());
        const cv::Mat spriteOutline = buildSpriteCoverageMask(index, m_useHdFrame);
        if (!spriteOutline.empty()) {
            m_framesCanvas->canvas()->setTertiaryMaskOutline(spriteOutline, QColor(255, 220, 0));
        } else {
            m_framesCanvas->canvas()->clearTertiaryOutline();
        }
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
    const int gapPixels = FrameGapForWidth(composed.cols);
    m_framesCanvas->canvas()->setGridSegments(composed.rows, gapPixels, displayOriginal.rows);
    m_framesCanvas->canvas()->setGridScales(1, bottomScale);
    FrameLayout layout = BuildFrameLayout(composed, displayOriginal);
    const QRect topRegion(layout.topX, 0, layout.topWidth, layout.topHeight);
    const QRect bottomRegion(layout.bottomX,
                             layout.topHeight + gapPixels,
                             layout.bottomWidth,
                             layout.bottomHeight);
    m_framesCanvas->canvas()->setGridRegions(topRegion, bottomRegion);
    const cv::Mat spriteOutline = buildSpriteCoverageMask(index, m_useHdFrame);
    if (!spriteOutline.empty()) {
        m_framesCanvas->canvas()->setTertiaryMaskOutline(spriteOutline, QColor(255, 220, 0), topRegion);
    } else {
        m_framesCanvas->canvas()->clearTertiaryOutline();
    }
}

void MainWindow::updateSpriteCanvasImage(int index)
{
    if (!m_spritesCanvas) {
        return;
    }
    if (index < 0) {
        m_spritesCanvas->setImage(cv::Mat());
        return;
    }
    const cv::Mat* image = activeSpriteImage(index);
    if (!image || image->empty()) {
        m_spritesCanvas->setImage(cv::Mat());
        return;
    }
    const QRect contentRect = spriteContentRect(index);
    const QRect displayRect = spriteDisplayRect(index, *image);
    cv::Rect roi(0, 0, image->cols, image->rows);
    if (displayRect.isValid() && !displayRect.isEmpty()) {
        roi = cv::Rect(displayRect.x(), displayRect.y(), displayRect.width(), displayRect.height());
    }
    cv::Mat baseFull = m_spriteDynamicMaskMode
        ? applySpriteDynamicColors(index, *image)
        : EnsureBgr(*image);
    cv::Mat base = baseFull(roi).clone();
    cv::Mat display = base;
    if (m_spriteDynamicMaskMode) {
        const cv::Mat* map = nullptr;
        if (m_useHdSprite && index >= 0) {
            map = ensureHdSpriteDynamicMaskLocal(index);
        } else if (index >= 0) {
            map = ensureSpriteDynamicMaskLocal(index);
        }
        if (map && !map->empty()) {
            cv::Mat maskFull = buildDynamicMaskFromMap(*map, m_spriteDynamicSetIndex);
            cv::Mat mask = maskFull(roi).clone();
            if (MaskHasContent(mask)) {
                display = buildMaskPreview(base, mask, cv::Vec3b(0, 200, 255));
            }
        }
    }
    cv::Mat originalRef;
    if (const cv::Mat* originalSource = spriteOriginalForDisplay(index)) {
        if (contentRect.isValid() && !contentRect.isEmpty()) {
            const QRect bounds(0, 0, originalSource->cols, originalSource->rows);
            const QRect clipped = contentRect.intersected(bounds);
            if (!clipped.isEmpty()) {
                originalRef = (*originalSource)(cv::Rect(clipped.x(),
                                                        clipped.y(),
                                                        clipped.width(),
                                                        clipped.height())).clone();
            }
        } else {
            originalRef = originalSource->clone();
        }
    }
    cv::Mat original;
    if (!originalRef.empty()) {
        cv::Mat cleaned = originalRef.clone();
        for (int y = 0; y < cleaned.rows; ++y) {
            uint8_t* row = cleaned.ptr<uint8_t>(y);
            for (int x = 0; x < cleaned.cols; ++x) {
                if (row[x] == 255) {
                    row[x] = 0;
                }
            }
        }
        original = buildOriginalFrame(cleaned);
    }
    if (!original.empty() && m_showSpriteOriginal) {
        const cv::Size originalTargetSize = m_useHdSprite ? original.size() : display.size();
        cv::Mat displayOriginal = BuildDisplayOriginal(original, originalTargetSize);
        const QColor gap = m_spritesCanvas->palette().color(QPalette::Window);
        cv::Mat combined = buildCombinedFrame(display,
                                              displayOriginal,
                                              cv::Scalar(gap.blue(), gap.green(), gap.red()));
        m_spritesCanvas->setImage(combined);
        int bottomScale = 1;
        if (!displayOriginal.empty() && original.rows > 0 && original.cols > 0 &&
            displayOriginal.rows == original.rows * 2 && displayOriginal.cols == original.cols * 2) {
            bottomScale = 2;
        }
        const int gapPixels = FrameGapForWidth(display.cols);
        m_spritesCanvas->canvas()->setGridSegments(display.rows, gapPixels, displayOriginal.rows);
        m_spritesCanvas->canvas()->setGridScales(1, bottomScale);
        FrameLayout layout = BuildFrameLayout(display, displayOriginal);
        const QRect topRegion(layout.topX, 0, layout.topWidth, layout.topHeight);
        const QRect bottomRegion(layout.bottomX,
                                 layout.topHeight + gapPixels,
                                 layout.bottomWidth,
                                 layout.bottomHeight);
        m_spritesCanvas->canvas()->setGridRegions(topRegion, bottomRegion);
    } else {
        m_spritesCanvas->setImage(display);
        m_spritesCanvas->canvas()->setGridSegments(0, 0, 0);
        m_spritesCanvas->canvas()->setGridScales(1, 1);
        m_spritesCanvas->canvas()->setGridRegions(QRect(), QRect());
    }
    if (m_spriteDetAreaMode) {
        updateSpriteDetAreaOverlay(index);
    } else {
        m_spritesCanvas->canvas()->clearMaskOutline();
    }
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
    if (!m_backgroundFramesX[static_cast<std::size_t>(index)].empty()) {
        return !IsAllBlackFrame(m_backgroundFramesX[static_cast<std::size_t>(index)]);
    }
    if (m_serumDataLoaded && index < static_cast<int>(m_serumData.nbackgrounds) &&
        m_serumData.backgroundframes_v2_extra.hasData(static_cast<uint32_t>(index))) {
        const uint16_t* data = m_serumData.backgroundframes_v2_extra[static_cast<uint32_t>(index)];
        if (!data || m_serumData.fwidth_extra == 0 || m_serumData.fheight_extra == 0) {
            return false;
        }
        const std::size_t count =
            static_cast<std::size_t>(m_serumData.fwidth_extra) * m_serumData.fheight_extra;
        for (std::size_t i = 0; i < count; ++i) {
            if (data[i] != 0) {
                return true;
            }
        }
    }
    return false;
}

cv::Mat* MainWindow::ensureHdBackgroundLocal(int index)
{
    if (index < 0 || index >= static_cast<int>(m_backgroundFramesX.size())) {
        return nullptr;
    }
    cv::Mat& frame = m_backgroundFramesX[static_cast<std::size_t>(index)];
    if (!frame.empty()) {
        return &frame;
    }
    if (m_serumDataLoaded && index < static_cast<int>(m_serumData.nbackgrounds) &&
        m_serumData.backgroundframes_v2_extra.hasData(static_cast<uint32_t>(index))) {
        const uint16_t* data = m_serumData.backgroundframes_v2_extra[static_cast<uint32_t>(index)];
        if (data && m_serumData.fwidth_extra > 0 && m_serumData.fheight_extra > 0) {
            frame = ConvertRgb565ToBgrMat(data,
                                          static_cast<int>(m_serumData.fwidth_extra),
                                          static_cast<int>(m_serumData.fheight_extra));
        }
    }
    return frame.empty() ? nullptr : &frame;
}

bool MainWindow::hasHdSprite(int index) const
{
    if (index < 0) {
        return false;
    }
    if (index < static_cast<int>(m_spriteColoredX.size()) &&
        !m_spriteColoredX[static_cast<std::size_t>(index)].empty()) {
        return true;
    }
    if (index < static_cast<int>(m_spriteExtraFlags.size()) &&
        m_spriteExtraFlags[static_cast<std::size_t>(index)] != 0) {
        return true;
    }
    if (m_serumDataLoaded && index < static_cast<int>(m_serumData.nsprites) &&
        m_serumData.isextrasprite.hasData(static_cast<uint32_t>(index))) {
        const uint8_t* flag = m_serumData.isextrasprite[static_cast<uint32_t>(index)];
        return flag && flag[0] != 0;
    }
    if (m_serumDataLoaded && index < static_cast<int>(m_serumData.nsprites) &&
        m_serumData.spritecolored_extra.hasData(static_cast<uint32_t>(index))) {
        return true;
    }
    return false;
}

cv::Mat* MainWindow::ensureHdSpriteLocal(int index)
{
    if (index < 0) {
        return nullptr;
    }
    if (index >= static_cast<int>(m_spriteColoredX.size())) {
        m_spriteColoredX.resize(static_cast<std::size_t>(index + 1));
    }
    cv::Mat& sprite = m_spriteColoredX[static_cast<std::size_t>(index)];
    if (!sprite.empty()) {
        return &sprite;
    }
    if (m_serumDataLoaded && index < static_cast<int>(m_serumData.nsprites) &&
        m_serumData.spritecolored_extra.hasData(static_cast<uint32_t>(index))) {
        const uint16_t* data = m_serumData.spritecolored_extra[static_cast<uint32_t>(index)];
        if (data) {
            sprite = ConvertRgb565ToBgrMat(data, MAX_SPRITE_WIDTH, MAX_SPRITE_HEIGHT);
        }
    }
    return sprite.empty() ? nullptr : &sprite;
}

cv::Mat* MainWindow::ensureHdSpriteMaskLocal(int index)
{
    if (index < 0) {
        return nullptr;
    }
    if (index >= static_cast<int>(m_spriteMasksX.size())) {
        m_spriteMasksX.resize(static_cast<std::size_t>(index + 1));
    }
    cv::Mat& mask = m_spriteMasksX[static_cast<std::size_t>(index)];
    if (!mask.empty()) {
        return &mask;
    }
    if (m_serumDataLoaded && index < static_cast<int>(m_serumData.nsprites) &&
        m_serumData.spritemask_extra.hasData(static_cast<uint32_t>(index))) {
        const uint8_t* data = m_serumData.spritemask_extra[static_cast<uint32_t>(index)];
        if (data) {
            mask = cv::Mat(MAX_SPRITE_HEIGHT, MAX_SPRITE_WIDTH, CV_8UC1, cv::Scalar(255));
            std::memcpy(mask.data, data, static_cast<std::size_t>(MAX_SPRITE_WIDTH) * MAX_SPRITE_HEIGHT);
        }
    }
    return mask.empty() ? nullptr : &mask;
}

cv::Mat* MainWindow::ensureHdSpriteDynamicMaskLocal(int index)
{
    if (index < 0) {
        return nullptr;
    }
    if (index >= static_cast<int>(m_spriteDynamicMasksX.size())) {
        m_spriteDynamicMasksX.resize(static_cast<std::size_t>(index + 1));
    }
    cv::Mat& mask = m_spriteDynamicMasksX[static_cast<std::size_t>(index)];
    if (!mask.empty()) {
        return &mask;
    }
    if (m_serumDataLoaded && index < static_cast<int>(m_serumData.nsprites) &&
        m_serumData.dynaspritemasks_extra.hasData(static_cast<uint32_t>(index))) {
        const uint8_t* data = m_serumData.dynaspritemasks_extra[static_cast<uint32_t>(index)];
        if (data) {
            mask = cv::Mat(MAX_SPRITE_HEIGHT, MAX_SPRITE_WIDTH, CV_8UC1, cv::Scalar(255));
            std::memcpy(mask.data, data, static_cast<std::size_t>(MAX_SPRITE_WIDTH) * MAX_SPRITE_HEIGHT);
        }
    }
    return mask.empty() ? nullptr : &mask;
}

std::vector<uint16_t>* MainWindow::ensureHdSpriteDynamicColorsLocal(int index)
{
    if (index < 0) {
        return nullptr;
    }
    if (index >= static_cast<int>(m_spriteDynamicColorsX.size())) {
        m_spriteDynamicColorsX.resize(static_cast<std::size_t>(index + 1));
    }
    std::vector<uint16_t>& colors = m_spriteDynamicColorsX[static_cast<std::size_t>(index)];
    if (!colors.empty()) {
        return &colors;
    }
    if (m_serumDataLoaded && index < static_cast<int>(m_serumData.nsprites) &&
        m_serumData.dynasprite4cols_extra.hasData(static_cast<uint32_t>(index))) {
        const uint16_t* data = m_serumData.dynasprite4cols_extra[static_cast<uint32_t>(index)];
        if (data) {
            const int stride = serumDynamicStride();
            const std::size_t count =
                static_cast<std::size_t>(MAX_DYNA_SETS_PER_SPRITE) * static_cast<std::size_t>(stride);
            colors.assign(data, data + count);
        }
    }
    return colors.empty() ? nullptr : &colors;
}

QSize MainWindow::serumSpriteBaseSize() const
{
    if (!m_serumDataLoaded) {
        return QSize(MAX_SPRITE_WIDTH, MAX_SPRITE_HEIGHT);
    }
    const std::size_t baseCount = m_serumData.spriteoriginal.elementCount() > 0
        ? m_serumData.spriteoriginal.elementCount()
        : m_serumData.spritecolored.elementCount();
    const std::size_t sdCount = static_cast<std::size_t>(m_serumData.fwidth) * m_serumData.fheight;
    if (sdCount > 0 && baseCount == sdCount) {
        return QSize(static_cast<int>(m_serumData.fwidth),
                     static_cast<int>(m_serumData.fheight));
    }
    const std::size_t maxCount =
        static_cast<std::size_t>(MAX_SPRITE_WIDTH) * MAX_SPRITE_HEIGHT;
    if (baseCount == maxCount) {
        return QSize(MAX_SPRITE_WIDTH, MAX_SPRITE_HEIGHT);
    }
    if (baseCount > 0 && (baseCount % MAX_SPRITE_WIDTH) == 0) {
        return QSize(MAX_SPRITE_WIDTH,
                     static_cast<int>(baseCount / MAX_SPRITE_WIDTH));
    }
    return QSize(MAX_SPRITE_WIDTH, MAX_SPRITE_HEIGHT);
}

const uint8_t* MainWindow::serumSpriteOriginalData(uint32_t spriteId) const
{
    if (!m_serumDataLoaded || spriteId >= m_serumData.nsprites) {
        return nullptr;
    }
    const std::size_t spriteCount = m_serumData.spriteoriginal.elementCount();
    const std::size_t legacyCount = m_serumData.spritedescriptionso.elementCount();
    const uint8_t* spriteOriginal =
        (spriteCount > 0) ? m_serumData.spriteoriginal[spriteId] : nullptr;
    const uint8_t* legacyOriginal =
        (legacyCount > 0) ? m_serumData.spritedescriptionso[spriteId] : nullptr;
    if (spriteOriginal && legacyOriginal && spriteCount > 0 && legacyCount > 0) {
        const bool serumHasOpaque = SpriteHasNonTransparentPixel(spriteOriginal, spriteCount);
        const bool legacyHasOpaque = SpriteHasNonTransparentPixel(legacyOriginal, legacyCount);
        if (!serumHasOpaque && legacyHasOpaque) {
            return legacyOriginal;
        }
    }
    if (spriteOriginal) {
        return spriteOriginal;
    }
    if (legacyOriginal) {
        return legacyOriginal;
    }
    return nullptr;
}

cv::Mat* MainWindow::ensureSpriteOriginalLocal(int index)
{
    if (index < 0) {
        return nullptr;
    }
    if (index >= static_cast<int>(m_spriteOriginals.size())) {
        m_spriteOriginals.resize(static_cast<std::size_t>(index + 1));
    }
    cv::Mat& original = m_spriteOriginals[static_cast<std::size_t>(index)];
    if (!original.empty()) {
        return &original;
    }
    if (m_serumDataLoaded && index < static_cast<int>(m_serumData.nsprites)) {
        const uint32_t spriteId = static_cast<uint32_t>(index);
        const uint8_t* data = serumSpriteOriginalData(spriteId);
        if (data) {
            const QSize sourceSize = serumSpriteBaseSize();
            const int srcWidth = sourceSize.width();
            const int srcHeight = sourceSize.height();
            original = cv::Mat(MAX_SPRITE_HEIGHT,
                               MAX_SPRITE_WIDTH,
                               CV_8UC1,
                               cv::Scalar(255));
            if (srcWidth == MAX_SPRITE_WIDTH && srcHeight == MAX_SPRITE_HEIGHT) {
                std::memcpy(original.data,
                            data,
                            static_cast<std::size_t>(MAX_SPRITE_WIDTH) * MAX_SPRITE_HEIGHT);
            } else if (srcWidth > 0 && srcHeight > 0) {
                for (int y = 0; y < std::min(srcHeight, MAX_SPRITE_HEIGHT); ++y) {
                    std::memcpy(original.ptr<uint8_t>(y),
                                data + static_cast<std::size_t>(y) * srcWidth,
                                static_cast<std::size_t>(std::min(srcWidth, MAX_SPRITE_WIDTH)));
                }
            }
        }
    }
    return original.empty() ? nullptr : &original;
}

cv::Mat* MainWindow::ensureSpriteDynamicMaskLocal(int index)
{
    if (index < 0) {
        return nullptr;
    }
    if (index >= static_cast<int>(m_spriteDynamicMasks.size())) {
        m_spriteDynamicMasks.resize(static_cast<std::size_t>(index + 1));
    }
    cv::Mat& mask = m_spriteDynamicMasks[static_cast<std::size_t>(index)];
    if (!mask.empty()) {
        return &mask;
    }
    if (m_serumDataLoaded && index < static_cast<int>(m_serumData.nsprites) &&
        m_serumData.dynaspritemasks.elementCount() > 0) {
        const uint8_t* data = m_serumData.dynaspritemasks[static_cast<uint32_t>(index)];
        if (data) {
            const QSize sourceSize = serumSpriteBaseSize();
            const int srcWidth = sourceSize.width();
            const int srcHeight = sourceSize.height();
            mask = cv::Mat(MAX_SPRITE_HEIGHT,
                           MAX_SPRITE_WIDTH,
                           CV_8UC1,
                           cv::Scalar(255));
            if (srcWidth == MAX_SPRITE_WIDTH && srcHeight == MAX_SPRITE_HEIGHT) {
                std::memcpy(mask.data,
                            data,
                            static_cast<std::size_t>(MAX_SPRITE_WIDTH) * MAX_SPRITE_HEIGHT);
            } else if (srcWidth > 0 && srcHeight > 0) {
                for (int y = 0; y < std::min(srcHeight, MAX_SPRITE_HEIGHT); ++y) {
                    std::memcpy(mask.ptr<uint8_t>(y),
                                data + static_cast<std::size_t>(y) * srcWidth,
                                static_cast<std::size_t>(std::min(srcWidth, MAX_SPRITE_WIDTH)));
                }
            }
        }
    }
    return mask.empty() ? nullptr : &mask;
}

std::vector<uint16_t>* MainWindow::ensureSpriteDynamicColorsLocal(int index)
{
    if (index < 0) {
        return nullptr;
    }
    if (index >= static_cast<int>(m_spriteDynamicColors.size())) {
        m_spriteDynamicColors.resize(static_cast<std::size_t>(index + 1));
    }
    std::vector<uint16_t>& colors = m_spriteDynamicColors[static_cast<std::size_t>(index)];
    if (!colors.empty()) {
        return &colors;
    }
    if (m_serumDataLoaded && index < static_cast<int>(m_serumData.nsprites) &&
        m_serumData.dynasprite4cols.hasData(static_cast<uint32_t>(index))) {
        const uint16_t* data = m_serumData.dynasprite4cols[static_cast<uint32_t>(index)];
        if (data) {
            const int stride = serumDynamicStride();
            const std::size_t count =
                static_cast<std::size_t>(MAX_DYNA_SETS_PER_SPRITE) * static_cast<std::size_t>(stride);
            colors.assign(data, data + count);
        }
    }
    return colors.empty() ? nullptr : &colors;
}

int MainWindow::spriteDynamicColorsPerSet(const std::vector<uint16_t>& colors) const
{
    if (colors.empty()) {
        return 0;
    }
    if (colors.size() % MAX_DYNA_SETS_PER_SPRITE == 0) {
        return static_cast<int>(colors.size() / MAX_DYNA_SETS_PER_SPRITE);
    }
    return 16;
}

void MainWindow::updateHdControlsForContext()
{
    if (!m_hdCreateButton || !m_hdDeleteButton || !m_hdScaleCombo || !m_hdSourceCombo) {
        return;
    }
    const bool inBackgrounds = m_canvasTabs && m_canvasTabs->currentWidget() == m_backgroundsCanvas;
    const bool inSprites = m_canvasTabs && m_canvasTabs->currentWidget() == m_spritesCanvas;
    if (inBackgrounds) {
        const int bgIndex = m_backgroundList ? m_backgroundList->currentRow() : -1;
        m_hdSourceCombo->setEnabled(false);
        m_hdScaleCombo->setEnabled(bgIndex >= 0);
        m_hdCreateButton->setEnabled(bgIndex >= 0 && !hasHdBackground(bgIndex));
        m_hdDeleteButton->setEnabled(bgIndex >= 0 && hasHdBackground(bgIndex));
    } else if (inSprites) {
        const int spriteIndex = m_spritesList ? m_spritesList->currentRow() : -1;
        const bool hasHd = hasHdSprite(spriteIndex);
        m_hdSourceCombo->setEnabled(false);
        m_hdScaleCombo->setEnabled(spriteIndex >= 0);
        m_hdCreateButton->setEnabled(spriteIndex >= 0 && !hasHd);
        m_hdDeleteButton->setEnabled(hasHd);
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

std::vector<uint16_t> ConvertBgrMatToRgb565(const cv::Mat& source)
{
    cv::Mat bgr = EnsureBgr(source);
    if (bgr.empty()) {
        return {};
    }
    std::vector<uint16_t> output(static_cast<std::size_t>(bgr.rows) * bgr.cols);
    for (int y = 0; y < bgr.rows; ++y) {
        const cv::Vec3b* row = bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < bgr.cols; ++x) {
            output[static_cast<std::size_t>(y) * bgr.cols + x] = BgrToRgb565(row[x]);
        }
    }
    return output;
}

cv::Mat ConvertRgb565ToBgrMat(const uint16_t* data, int width, int height)
{
    if (!data || width <= 0 || height <= 0) {
        return cv::Mat();
    }
    cv::Mat output(height, width, CV_8UC3);
    for (int y = 0; y < height; ++y) {
        cv::Vec3b* row = output.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; ++x) {
            row[x] = Rgb565ToBgr(data[static_cast<std::size_t>(y) * width + x]);
        }
    }
    return output;
}

}

void MainWindow::ensureMaskDataSize()
{
    const int frameCount = m_frameStore ? m_frameStore->count() : 0;
    if (frameCount < 0) {
        return;
    }

    int width = 0;
    int height = 0;
    if (m_serumDataLoaded && m_serumData.fwidth > 0 && m_serumData.fheight > 0) {
        width = static_cast<int>(m_serumData.fwidth);
        height = static_cast<int>(m_serumData.fheight);
    } else {
        const cv::Mat* firstFrame = (frameCount > 0) ? m_frameStore->at(0) : nullptr;
        width = firstFrame ? firstFrame->cols : 0;
        height = firstFrame ? firstFrame->rows : 0;
    }

    if (m_compMasks.size() != MAX_MASKS) {
        m_compMasks.resize(MAX_MASKS);
    }
    for (int i = 0; i < MAX_MASKS; ++i) {
        cv::Mat& mask = m_compMasks[static_cast<std::size_t>(i)];
        if (width > 0 && height > 0 && (mask.empty() || mask.cols != width || mask.rows != height)) {
            mask = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
        }
    }

    m_frameRefs.resize(static_cast<std::size_t>(frameCount));
    m_frameDynamicColors.resize(static_cast<std::size_t>(frameCount));
    m_frameCompMaskIds.resize(static_cast<std::size_t>(frameCount), 255);
    if (m_frameDynamicMaskMaps.size() != static_cast<std::size_t>(frameCount)) {
        m_frameDynamicMaskMaps.resize(static_cast<std::size_t>(frameCount));
    }
    if (m_frameDynamicMaskMapsX.size() != static_cast<std::size_t>(frameCount)) {
        m_frameDynamicMaskMapsX.resize(static_cast<std::size_t>(frameCount));
    }
    m_frameShapeCompModes.resize(static_cast<std::size_t>(frameCount), 0);
    const std::size_t frameCountSize = static_cast<std::size_t>(frameCount);
    const std::size_t totalSpriteSlots = frameCountSize * MAX_SPRITES_PER_FRAME;
    if (frameCountSize != 0 && totalSpriteSlots / MAX_SPRITES_PER_FRAME != frameCountSize) {
        return;
    }
    if (m_frameSpriteAssignments.size() != totalSpriteSlots) {
        m_frameSpriteAssignments.resize(totalSpriteSlots, 255);
    }
    if (m_frameSpriteZoneFlags.size() != totalSpriteSlots) {
        m_frameSpriteZoneFlags.resize(totalSpriteSlots, 0);
    }
    const std::size_t totalSpriteBBoxes = totalSpriteSlots * 4;
    if (m_frameSpriteBBoxes.size() != totalSpriteBBoxes) {
        m_frameSpriteBBoxes.resize(totalSpriteBBoxes, 0);
    }

    cv::Size hdSize;
    for (const auto& hd : m_frameExtraFrames) {
        if (!hd.empty()) {
            hdSize = hd.size();
            break;
        }
    }

    for (int i = 0; i < frameCount; ++i) {
        if (!m_serumDataLoaded) {
            if (const cv::Mat* frame = m_frameStore->at(i)) {
                cv::Mat& ref = m_frameRefs[static_cast<std::size_t>(i)];
                if (ref.empty() || ref.rows != frame->rows || ref.cols != frame->cols) {
                    ref = buildReferenceFrame(*frame);
                }
            }
        }
        if (width > 0 && height > 0) {
            for (int slot = 0; slot < MAX_SPRITES_PER_FRAME; ++slot) {
                const std::size_t bboxIndex = static_cast<std::size_t>(i) * MAX_SPRITES_PER_FRAME * 4 +
                    static_cast<std::size_t>(slot) * 4;
                const std::size_t slotIndex = static_cast<std::size_t>(i) * MAX_SPRITES_PER_FRAME +
                    static_cast<std::size_t>(slot);
                if (bboxIndex + 3 >= m_frameSpriteBBoxes.size()) {
                    break;
                }
                if (slotIndex >= m_frameSpriteAssignments.size()) {
                    continue;
                }
                if (m_frameSpriteAssignments[slotIndex] != 255) {
                    if (slotIndex < m_frameSpriteZoneFlags.size()) {
                        m_frameSpriteZoneFlags[slotIndex] = 1;
                    }
                }
                if (m_frameSpriteAssignments[slotIndex] != 255 &&
                    m_frameSpriteBBoxes[bboxIndex + 2] == 0 &&
                    m_frameSpriteBBoxes[bboxIndex + 3] == 0) {
                    m_frameSpriteBBoxes[bboxIndex] = 0;
                    m_frameSpriteBBoxes[bboxIndex + 1] = 0;
                    m_frameSpriteBBoxes[bboxIndex + 2] = static_cast<uint16_t>(width - 1);
                    m_frameSpriteBBoxes[bboxIndex + 3] = static_cast<uint16_t>(height - 1);
                }
            }
        }
        if (width > 0 && height > 0) {
            cv::Mat& map = m_frameDynamicMaskMaps[static_cast<std::size_t>(i)];
            if (map.empty()) {
                if (!m_serumDataLoaded) {
                    map = cv::Mat(height, width, CV_8UC1, cv::Scalar(255));
                }
            } else if (map.cols != width || map.rows != height) {
                cv::Mat resized;
                cv::resize(map, resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_NEAREST);
                map = resized;
            }
        }
        if (hdSize.width > 0 && hdSize.height > 0) {
            cv::Mat& mapX = m_frameDynamicMaskMapsX[static_cast<std::size_t>(i)];
            if (mapX.empty()) {
                if (!m_serumDataLoaded) {
                    const cv::Mat& map = m_frameDynamicMaskMaps[static_cast<std::size_t>(i)];
                    if (!map.empty()) {
                        cv::resize(map, mapX, hdSize, 0.0, 0.0, cv::INTER_NEAREST);
                    } else {
                        mapX = cv::Mat(hdSize.height, hdSize.width, CV_8UC1, cv::Scalar(255));
                    }
                }
            } else if (mapX.cols != hdSize.width || mapX.rows != hdSize.height) {
                cv::Mat resized;
                cv::resize(mapX, resized, hdSize, 0.0, 0.0, cv::INTER_NEAREST);
                mapX = resized;
            }
        }
        if (!m_serumDataLoaded) {
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
    }

    const QListWidgetItem* firstFrameItem = (m_framesList && m_framesList->count() > 0)
        ? m_framesList->item(0)
        : nullptr;
    const bool hasFrames = frameCount > 0 && m_framesList &&
        !(frameCount == 1 && firstFrameItem && firstFrameItem->text().startsWith("No frames"));
    m_frameMaskAssign->setEnabled(hasFrames);
    m_frameDynamicMaskAssign->setEnabled(hasFrames);
    if (m_frameDynamicCopyButton) {
        m_frameDynamicCopyButton->setEnabled(hasFrames);
    }
    m_shapeCompToggle->setEnabled(hasFrames);
    if (m_triggerIdSpin) {
        m_triggerIdSpin->setEnabled(hasFrames);
    }
    if (m_triggerMonochromeCheck) {
        m_triggerMonochromeCheck->setEnabled(hasFrames);
    }
    if (m_framesCanvas) {
        m_framesCanvas->setMaskButtonsEnabled(hasFrames);
        m_framesCanvas->setBackgroundMaskEnabled(hasFrames);
        m_framesCanvas->setZoneButtonEnabled(hasFrames);
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

void MainWindow::ensureSpriteDataSize()
{
    const int spriteCount = m_spriteStore ? m_spriteStore->count() : 0;
    if (spriteCount < 0) {
        return;
    }
    const cv::Size spriteSize(MAX_SPRITE_WIDTH, MAX_SPRITE_HEIGHT);
    const std::size_t spriteCountSize = static_cast<std::size_t>(spriteCount);
    if (m_serumDataLoaded) {
        if (m_spriteOriginals.size() > spriteCountSize) {
            m_spriteOriginals.resize(spriteCountSize);
        }
        if (m_spriteDynamicMasks.size() > spriteCountSize) {
            m_spriteDynamicMasks.resize(spriteCountSize);
        }
        if (m_spriteDynamicColors.size() > spriteCountSize) {
            m_spriteDynamicColors.resize(spriteCountSize);
        }
        if (m_spriteColored.size() > spriteCountSize) {
            m_spriteColored.resize(spriteCountSize);
        }
        if (m_spriteColoredX.size() > spriteCountSize) {
            m_spriteColoredX.resize(spriteCountSize);
        }
        if (m_spriteMasksX.size() > spriteCountSize) {
            m_spriteMasksX.resize(spriteCountSize);
        }
        if (m_spriteDynamicMasksX.size() > spriteCountSize) {
            m_spriteDynamicMasksX.resize(spriteCountSize);
        }
        if (m_spriteDynamicColorsX.size() > spriteCountSize) {
            m_spriteDynamicColorsX.resize(spriteCountSize);
        }
    } else {
        if (m_spriteOriginals.size() != spriteCountSize) {
            m_spriteOriginals.resize(spriteCountSize);
        }
        if (m_spriteDynamicMasks.size() != spriteCountSize) {
            m_spriteDynamicMasks.resize(spriteCountSize);
        }
        if (m_spriteDynamicColors.size() != spriteCountSize) {
            m_spriteDynamicColors.resize(spriteCountSize);
        }
        if (m_spriteColored.size() != spriteCountSize) {
            m_spriteColored.resize(spriteCountSize);
        }
        if (m_spriteColoredX.size() != spriteCountSize) {
            m_spriteColoredX.resize(spriteCountSize);
        }
        if (m_spriteMasksX.size() != spriteCountSize) {
            m_spriteMasksX.resize(spriteCountSize);
        }
        if (m_spriteDynamicMasksX.size() != spriteCountSize) {
            m_spriteDynamicMasksX.resize(spriteCountSize);
        }
        if (m_spriteDynamicColorsX.size() != spriteCountSize) {
            m_spriteDynamicColorsX.resize(spriteCountSize);
        }
    }
    if (m_spriteExtraFlags.size() != static_cast<std::size_t>(spriteCount)) {
        m_spriteExtraFlags.resize(static_cast<std::size_t>(spriteCount), 0);
    }
    if (m_spriteShapeModes.size() != static_cast<std::size_t>(spriteCount)) {
        m_spriteShapeModes.resize(static_cast<std::size_t>(spriteCount), 0);
    }
    const std::size_t detSize = static_cast<std::size_t>(spriteCount) * MAX_SPRITE_DETECT_AREAS * 4;
    if (m_spriteDetAreas.size() != detSize) {
        m_spriteDetAreas.resize(detSize, 0xffff);
    }

    if (!m_serumDataLoaded) {
        for (int i = 0; i < spriteCount; ++i) {
            cv::Mat& original = m_spriteOriginals[static_cast<std::size_t>(i)];
            if (original.empty() || original.size() != spriteSize) {
                original = cv::Mat(spriteSize, CV_8UC1, cv::Scalar(0));
            }
            cv::Mat& map = m_spriteDynamicMasks[static_cast<std::size_t>(i)];
            if (map.empty() || map.size() != spriteSize) {
                map = cv::Mat(spriteSize, CV_8UC1, cv::Scalar(255));
            }
            std::vector<uint16_t>& colors = m_spriteDynamicColors[static_cast<std::size_t>(i)];
            if (colors.empty()) {
                colors.resize(MAX_DYNA_SETS_PER_SPRITE * 64, 0);
                for (int set = 0; set < MAX_DYNA_SETS_PER_SPRITE; ++set) {
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
    }

    const QListWidgetItem* firstSpriteItem = (m_spritesList && m_spritesList->count() > 0)
        ? m_spritesList->item(0)
        : nullptr;
    const bool hasSprites = spriteCount > 0 && m_spritesList &&
        !(spriteCount == 1 && firstSpriteItem && firstSpriteItem->text().startsWith("No sprites"));
    if (m_spriteDynamicSetCombo) {
        m_spriteDynamicSetCombo->setEnabled(hasSprites);
        const int clamped = std::clamp(m_spriteDynamicSetIndex, 0, MAX_DYNA_SETS_PER_SPRITE - 1);
        if (clamped != m_spriteDynamicSetIndex) {
            m_spriteDynamicSetIndex = clamped;
        }
        QSignalBlocker blocker(m_spriteDynamicSetCombo);
        m_spriteDynamicSetCombo->setCurrentIndex(m_spriteDynamicSetIndex);
    }
    if (m_spritesCanvas) {
        m_spritesCanvas->setMaskButtonsEnabled(hasSprites);
        if (!hasSprites) {
            m_spriteDynamicMaskMode = false;
            m_spriteDetAreaMode = false;
            m_spritesCanvas->setMaskButtonsChecked(false, false);
        } else {
            m_spritesCanvas->setMaskButtonsChecked(m_spriteDetAreaMode, m_spriteDynamicMaskMode);
        }
    }
}

void MainWindow::ensureBackgroundDataSize()
{
    const int frameCount = m_frameStore ? m_frameStore->count() : 0;
    if (frameCount < 0) {
        return;
    }
    const QListWidgetItem* firstFrameItem = (m_framesList && m_framesList->count() > 0)
        ? m_framesList->item(0)
        : nullptr;
    const bool hasFrames = frameCount > 0 && m_framesList &&
        !(frameCount == 1 && firstFrameItem && firstFrameItem->text().startsWith("No frames"));
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
        if (width > 0 && height > 0) {
            if (mask.empty()) {
                if (!m_serumDataLoaded) {
                    mask = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
                }
            } else if (mask.cols != width || mask.rows != height) {
                cv::Mat resized;
                cv::resize(mask, resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_NEAREST);
                mask = resized;
            }
        }
        cv::Mat& maskX = m_frameBackgroundMasksX[static_cast<std::size_t>(i)];
        if (hdSize.width > 0 && hdSize.height > 0) {
            if (maskX.empty()) {
                if (!m_serumDataLoaded) {
                    if (!mask.empty()) {
                        cv::resize(mask, maskX, hdSize, 0.0, 0.0, cv::INTER_NEAREST);
                    } else {
                        maskX = cv::Mat(hdSize.height, hdSize.width, CV_8UC1, cv::Scalar(0));
                    }
                }
            } else if (maskX.cols != hdSize.width || maskX.rows != hdSize.height) {
                cv::Mat resized;
                cv::resize(maskX, resized, hdSize, 0.0, 0.0, cv::INTER_NEAREST);
                maskX = resized;
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
        logLine(QString("Mask preview: invalid index=%1").arg(index));
        m_framesCanvas->canvas()->clearPreviewImage();
        m_framesCanvas->canvas()->clearMaskOutline();
        m_framesCanvas->canvas()->clearTertiaryOutline();
        return;
    }
    const cv::Mat* frame = activeFrameImage(index, false);
    if (!frame || frame->empty()) {
        logLine(QString("Mask preview: empty frame index=%1").arg(index));
        m_framesCanvas->canvas()->clearMaskOutline();
        m_framesCanvas->canvas()->clearTertiaryOutline();
        return;
    }
    ensureMaskDataSize();
    ensureBackgroundDataSize();
    if (m_serumDataLoaded && index >= 0 && index < static_cast<int>(m_frameDynamicMaskMaps.size())) {
        if (m_frameDynamicMaskMaps[static_cast<std::size_t>(index)].empty() &&
            m_serumData.dynamasks.hasData(static_cast<uint32_t>(index))) {
            activeDynamicMaskMap(index);
        }
    }
    if (m_spriteZoneMode) {
        updateSpriteZoneOverlay(index);
        return;
    }
    cv::Mat reference = buildOriginalPreviewForIndex(index);
    const QColor gap = m_framesCanvas->palette().color(QPalette::Window);
    const cv::Scalar gapColor(gap.blue(), gap.green(), gap.red());
    cv::Mat base = renderFrameWithSerum(index, m_useHdFrame);
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
        const cv::Mat spriteOutline = buildSpriteCoverageMask(index, m_useHdFrame);
        if (!spriteOutline.empty()) {
            m_framesCanvas->canvas()->setTertiaryMaskOutline(spriteOutline, QColor(255, 220, 0));
        } else {
            m_framesCanvas->canvas()->clearTertiaryOutline();
        }
        return;
    }
    if (!m_showOriginalFrame && m_maskMode == MaskMode::None && !hasTopMask) {
        logLine(QString("Mask preview: cleared (no masks) index=%1").arg(index));
    }

    cv::Mat bottomPreview;
    cv::Mat dynamicMask;
    QColor bottomOutlineColor;
    if (m_showOriginalFrame) {
        cv::Mat original = buildOriginalFrame(reference);
        if (m_maskMode == MaskMode::Dynamic) {
            const int setId = currentFrameDynamicMaskId();
            cv::Mat* map = activeDynamicMaskMap(index);
            if (setId >= 0 && map && !map->empty()) {
                dynamicMask = buildDynamicMaskFromMap(*map, setId);
                if (!dynamicMask.empty()) {
                    bottomPreview = buildMaskPreview(original, dynamicMask, cv::Vec3b(0, 200, 255));
                    hasBottomMask = MaskHasContent(dynamicMask);
                    bottomOutlineColor = QColor(255, 200, 0);
                }
            }
        } else if (m_maskMode == MaskMode::Comparison) {
            const int maskId = currentFrameMaskId();
            cv::Mat mask;
            if (maskId >= 0) {
                if (m_serumDataLoaded && m_serumData.compmasks.hasData(static_cast<uint32_t>(maskId))) {
                    mask = cv::Mat(static_cast<int>(m_serumData.fheight),
                                   static_cast<int>(m_serumData.fwidth),
                                   CV_8UC1,
                                   const_cast<uint8_t*>(m_serumData.compmasks[static_cast<uint32_t>(maskId)]));
                } else if (maskId < static_cast<int>(m_compMasks.size())) {
                    mask = m_compMasks[static_cast<std::size_t>(maskId)];
                }
            }
            if (!mask.empty()) {
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
        const int gap = FrameGapForWidth(layout.topWidth);
        bottomRegion = QRect(layout.bottomX,
                             layout.topHeight + gap,
                             layout.bottomWidth,
                             layout.bottomHeight);
        cv::Mat combined = buildCombinedFrame(topPreview, displayOriginal, gapColor);
        if (hasTopMask || hasBottomMask) {
            m_framesCanvas->canvas()->setPreviewImage(combined);
        } else {
            m_framesCanvas->canvas()->clearPreviewImage();
        }
    } else {
        if (hasTopMask) {
            m_framesCanvas->canvas()->setPreviewImage(topPreview);
        } else {
            m_framesCanvas->canvas()->clearPreviewImage();
        }
    }

    if (hasBottomMask && m_maskMode != MaskMode::None) {
        const cv::Mat* mask = nullptr;
        if (m_maskMode == MaskMode::Dynamic) {
            if (!dynamicMask.empty()) {
                mask = &dynamicMask;
            }
        } else if (m_maskMode == MaskMode::Comparison) {
            const int maskId = currentFrameMaskId();
            if (maskId >= 0) {
                if (m_serumDataLoaded && m_serumData.compmasks.hasData(static_cast<uint32_t>(maskId))) {
                    dynamicMask = cv::Mat(static_cast<int>(m_serumData.fheight),
                                          static_cast<int>(m_serumData.fwidth),
                                          CV_8UC1,
                                          const_cast<uint8_t*>(m_serumData.compmasks[static_cast<uint32_t>(maskId)]));
                    mask = &dynamicMask;
                } else if (maskId < static_cast<int>(m_compMasks.size())) {
                    mask = &m_compMasks[static_cast<std::size_t>(maskId)];
                }
            }
        }
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
    const cv::Mat spriteOutline = buildSpriteCoverageMask(index, m_useHdFrame);
    if (!spriteOutline.empty()) {
        QRect topRegion;
        if (m_showOriginalFrame) {
            cv::Mat baseOriginal = bottomPreview.empty() ? buildOriginalFrame(reference) : bottomPreview;
            cv::Mat displayOriginal = BuildDisplayOriginal(baseOriginal, topPreview.size());
            FrameLayout layout = BuildFrameLayout(topPreview, displayOriginal);
            topRegion = QRect(layout.topX, 0, layout.topWidth, layout.topHeight);
        }
        m_framesCanvas->canvas()->setTertiaryMaskOutline(spriteOutline, QColor(255, 220, 0), topRegion);
    } else {
        m_framesCanvas->canvas()->clearTertiaryOutline();
    }
}

void MainWindow::updateSpriteZoneOverlay(int index)
{
    if (!m_framesCanvas || index < 0) {
        return;
    }
    const cv::Mat* frame = activeFrameImage(index, false);
    if (!frame || frame->empty()) {
        m_framesCanvas->canvas()->clearMaskOutline();
        return;
    }
    cv::Mat reference = buildOriginalPreviewForIndex(index);
    cv::Mat original = buildOriginalFrame(reference);
    const bool useOriginal = m_showOriginalFrame && !original.empty();
    cv::Size baseSize = useOriginal ? original.size() : frame->size();
    if (baseSize.width <= 0 || baseSize.height <= 0) {
        return;
    }
    std::vector<SpriteZoneGroup> zones = buildSpriteZonesForFrame(index);
    cv::Mat allMask(baseSize, CV_8UC1, cv::Scalar(0));
    cv::Mat selectedMask(baseSize, CV_8UC1, cv::Scalar(0));
    for (std::size_t i = 0; i < zones.size(); ++i) {
        const SpriteZoneGroup& zone = zones[i];
        cv::Rect rect(zone.rect.x(), zone.rect.y(), zone.rect.width(), zone.rect.height());
        rect &= cv::Rect(0, 0, baseSize.width, baseSize.height);
        if (rect.width <= 0 || rect.height <= 0) {
            continue;
        }
        cv::rectangle(allMask, rect, cv::Scalar(1), cv::FILLED);
    }
    bool hasSelected = false;
    if (m_selectedSpriteZoneIndex >= 0 && m_selectedSpriteZoneIndex < static_cast<int>(zones.size())) {
        const SpriteZoneGroup& zone = zones[static_cast<std::size_t>(m_selectedSpriteZoneIndex)];
        cv::Rect rect(zone.rect.x(), zone.rect.y(), zone.rect.width(), zone.rect.height());
        rect &= cv::Rect(0, 0, baseSize.width, baseSize.height);
        if (rect.width > 0 && rect.height > 0) {
            cv::rectangle(selectedMask, rect, cv::Scalar(1), cv::FILLED);
            hasSelected = true;
        }
    } else if (m_selectedSpriteSlot >= 0 && m_selectedSpriteSlot < MAX_SPRITES_PER_FRAME) {
        const std::size_t bboxIndex = static_cast<std::size_t>(index) * MAX_SPRITES_PER_FRAME * 4 +
            static_cast<std::size_t>(m_selectedSpriteSlot) * 4;
        if (bboxIndex + 3 < m_frameSpriteBBoxes.size()) {
            int minx = static_cast<int>(m_frameSpriteBBoxes[bboxIndex]);
            int miny = static_cast<int>(m_frameSpriteBBoxes[bboxIndex + 1]);
            int maxx = static_cast<int>(m_frameSpriteBBoxes[bboxIndex + 2]);
            int maxy = static_cast<int>(m_frameSpriteBBoxes[bboxIndex + 3]);
            if (maxx >= minx && maxy >= miny) {
                minx = std::clamp(minx, 0, baseSize.width - 1);
                miny = std::clamp(miny, 0, baseSize.height - 1);
                maxx = std::clamp(maxx, minx, baseSize.width - 1);
                maxy = std::clamp(maxy, miny, baseSize.height - 1);
                cv::Rect rect(minx, miny, maxx - minx + 1, maxy - miny + 1);
                cv::rectangle(selectedMask, rect, cv::Scalar(1), cv::FILLED);
                hasSelected = true;
            }
        }
    }
    QRect region;
    if (useOriginal) {
        cv::Mat displayOriginal = BuildDisplayOriginal(original, frame->size());
        FrameLayout layout = BuildFrameLayout(*frame, displayOriginal);
        const int gap = FrameGapForWidth(layout.topWidth);
        region = QRect(layout.bottomX,
                       layout.topHeight + gap,
                       layout.bottomWidth,
                       layout.bottomHeight);
    } else {
        FrameLayout layout = BuildFrameLayout(*frame, reference);
        region = QRect(layout.topX, 0, layout.topWidth, layout.topHeight);
    }
    if (hasSelected && MaskHasContent(selectedMask)) {
        m_framesCanvas->canvas()->setMaskOutline(selectedMask, QColor(255, 200, 0), region);
        m_framesCanvas->canvas()->setSecondaryMaskOutline(cv::Mat(), QColor());
    } else if (!hasSelected && MaskHasContent(allMask)) {
        m_framesCanvas->canvas()->setMaskOutline(allMask, QColor(255, 200, 0), region);
        m_framesCanvas->canvas()->setSecondaryMaskOutline(cv::Mat(), QColor());
    } else {
        m_framesCanvas->canvas()->clearMaskOutline();
    }
}

void MainWindow::updateSpriteDetAreaOverlay(int index)
{
    if (!m_spritesCanvas) {
        return;
    }
    if (index < 0 || index >= static_cast<int>(m_spriteDetAreas.size() / (MAX_SPRITE_DETECT_AREAS * 4))) {
        m_spritesCanvas->canvas()->clearMaskOutline();
        return;
    }
    const cv::Mat* displaySprite = activeSpriteImage(index);
    if (!displaySprite || displaySprite->empty()) {
        m_spritesCanvas->canvas()->clearMaskOutline();
        return;
    }
    const QRect contentRect = spriteContentRect(index);
    const int offsetX = contentRect.isValid() ? contentRect.x() : 0;
    const int offsetY = contentRect.isValid() ? contentRect.y() : 0;
    QRect displayRect = spriteDisplayRect(index, *displaySprite);
    if (!displayRect.isValid() || displayRect.isEmpty()) {
        displayRect = QRect(0, 0, displaySprite->cols, displaySprite->rows);
    }
    const int cropWidth = contentRect.isValid() ? contentRect.width() : displaySprite->cols;
    const int cropHeight = contentRect.isValid() ? contentRect.height() : displaySprite->rows;
    cv::Mat originalRef;
    if (const cv::Mat* originalSource = spriteOriginalForDisplay(index)) {
        cv::Rect crop(offsetX, offsetY, cropWidth, cropHeight);
        crop &= cv::Rect(0, 0, originalSource->cols, originalSource->rows);
        if (crop.width > 0 && crop.height > 0) {
            originalRef = (*originalSource)(crop).clone();
        }
    }
    if (originalRef.empty()) {
        m_spritesCanvas->canvas()->clearMaskOutline();
        return;
    }
    cv::Mat cleaned = originalRef.clone();
    for (int y = 0; y < cleaned.rows; ++y) {
        uint8_t* row = cleaned.ptr<uint8_t>(y);
        for (int x = 0; x < cleaned.cols; ++x) {
            if (row[x] == 255) {
                row[x] = 0;
            }
        }
    }
    cv::Mat original = buildOriginalFrame(cleaned);
    if (original.empty()) {
        m_spritesCanvas->canvas()->clearMaskOutline();
        return;
    }
    const cv::Size originalTargetSize = m_useHdSprite ? original.size()
                                                      : cv::Size(displayRect.width(), displayRect.height());
    cv::Mat displayOriginal = BuildDisplayOriginal(original, originalTargetSize);
    FrameLayout layout = BuildFrameLayout(displayRect.width(), displayRect.height(),
                                          displayOriginal.cols, displayOriginal.rows);
    const int gap = FrameGapForWidth(layout.topWidth);
    QRect region;
    if (m_showSpriteOriginal && !displayOriginal.empty()) {
        region = QRect(layout.bottomX,
                       layout.topHeight + gap,
                       layout.bottomWidth,
                       layout.bottomHeight);
    } else {
        region = QRect(layout.topX, 0, layout.topWidth, layout.topHeight);
    }
    const cv::Size size = original.size();
    cv::Mat allMask(size, CV_8UC1, cv::Scalar(0));
    cv::Mat selectedMask(size, CV_8UC1, cv::Scalar(0));
    const std::size_t base = static_cast<std::size_t>(index) * MAX_SPRITE_DETECT_AREAS * 4;
    int firstValidIndex = -1;
    cv::Rect firstValidRect;
    bool hasDifferentValid = false;
    for (int area = 0; area < MAX_SPRITE_DETECT_AREAS; ++area) {
        const std::size_t areaBase = base + static_cast<std::size_t>(area) * 4;
        if (areaBase + 3 >= m_spriteDetAreas.size()) {
            continue;
        }
        const uint16_t left = m_spriteDetAreas[areaBase];
        if (left == 0xffff) {
            continue;
        }
        const int fullX = static_cast<int>(left);
        const int fullY = static_cast<int>(m_spriteDetAreas[areaBase + 1]);
        const int w = static_cast<int>(m_spriteDetAreas[areaBase + 2]);
        const int h = static_cast<int>(m_spriteDetAreas[areaBase + 3]);
        if (w <= 0 || h <= 0) {
            continue;
        }
        cv::Rect rect(fullX - offsetX, fullY - offsetY, w, h);
        rect &= cv::Rect(0, 0, size.width, size.height);
        if (rect.width <= 0 || rect.height <= 0) {
            continue;
        }
        if (firstValidIndex < 0) {
            firstValidIndex = area;
            firstValidRect = rect;
        } else if (rect != firstValidRect) {
            hasDifferentValid = true;
        }
    }
    bool selectedValid = false;
    for (int area = 0; area < MAX_SPRITE_DETECT_AREAS; ++area) {
        const std::size_t areaBase = base + static_cast<std::size_t>(area) * 4;
        if (areaBase + 3 >= m_spriteDetAreas.size()) {
            continue;
        }
        const uint16_t left = m_spriteDetAreas[areaBase];
        if (left == 0xffff) {
            continue;
        }
        const int fullX = static_cast<int>(left);
        const int fullY = static_cast<int>(m_spriteDetAreas[areaBase + 1]);
        const int w = static_cast<int>(m_spriteDetAreas[areaBase + 2]);
        const int h = static_cast<int>(m_spriteDetAreas[areaBase + 3]);
        if (w <= 0 || h <= 0) {
            continue;
        }
        cv::Rect rect(fullX - offsetX, fullY - offsetY, w, h);
        rect &= cv::Rect(0, 0, size.width, size.height);
        if (rect.width <= 0 || rect.height <= 0) {
            continue;
        }
        if (!hasDifferentValid && area != firstValidIndex && rect == firstValidRect) {
            continue;
        }
        cv::rectangle(allMask, rect, cv::Scalar(1), cv::FILLED);
        if (area == m_spriteDetAreaIndex) {
            cv::rectangle(selectedMask, rect, cv::Scalar(1), cv::FILLED);
            selectedValid = true;
        }
    }
    cv::Mat otherMask = allMask.clone();
    if (!selectedValid && firstValidIndex >= 0 && !hasDifferentValid &&
        m_spriteDetAreaIndex != firstValidIndex) {
        m_spritesCanvas->canvas()->clearMaskOutline();
        m_spritesCanvas->canvas()->setSecondaryMaskOutline(cv::Mat(), QColor());
        return;
    }
    if (MaskHasContent(selectedMask)) {
        for (int y = 0; y < otherMask.rows; ++y) {
            uint8_t* row = otherMask.ptr<uint8_t>(y);
            const uint8_t* srow = selectedMask.ptr<uint8_t>(y);
            for (int x = 0; x < otherMask.cols; ++x) {
                if (srow[x]) {
                    row[x] = 0;
                }
            }
        }
        m_spritesCanvas->canvas()->setMaskOutline(selectedMask, QColor(255, 200, 0), region);
        if (MaskHasContent(otherMask)) {
            m_spritesCanvas->canvas()->setSecondaryMaskOutline(otherMask, QColor(0, 200, 255), region);
        } else {
            m_spritesCanvas->canvas()->setSecondaryMaskOutline(cv::Mat(), QColor());
        }
        return;
    }
    if (MaskHasContent(allMask)) {
        m_spritesCanvas->canvas()->setMaskOutline(allMask, QColor(255, 200, 0), region);
    } else {
        m_spritesCanvas->canvas()->clearMaskOutline();
    }
}

void MainWindow::setMaskMode(MaskMode mode)
{
    if (m_maskMode == mode) {
        return;
    }
    m_maskMode = mode;
    cancelCurrentDraw();
    if (m_spriteZoneMode) {
        m_spriteZoneMode = false;
        if (m_framesCanvas) {
            m_framesCanvas->setZoneButtonChecked(false);
        }
        refreshSpriteZoneList();
    }
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
    if (m_previewMaskOverlayEnabled) {
        refreshFramePreviews();
    }
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
        const QSize hint = m_maskList->gridSize().isValid()
            ? m_maskList->gridSize()
            : QSize(kPreviewIconWidth + 12, kPreviewIconHeight + 24);
        item->setSizeHint(hint);
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
        const QSize hint = m_dynamicMaskList->gridSize().isValid()
            ? m_dynamicMaskList->gridSize()
            : QSize(kDynamicMaskIconWidth + 12, kPreviewIconHeight + 24);
        item->setSizeHint(hint);
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
    updateFrameUsageHighlights(m_framesList ? m_framesList->currentRow() : -1);
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

bool MainWindow::dynamicMapHasSet(const cv::Mat& map, int setId) const
{
    if (map.empty() || setId < 0 || setId >= MAX_DYNA_SETS_PER_FRAMEN) {
        return false;
    }
    for (int y = 0; y < map.rows; ++y) {
        const uint8_t* row = map.ptr<uint8_t>(y);
        for (int x = 0; x < map.cols; ++x) {
            if (row[x] == static_cast<uint8_t>(setId)) {
                return true;
            }
        }
    }
    return false;
}

cv::Mat MainWindow::buildDynamicMaskFromMap(const cv::Mat& map, int setId) const
{
    if (map.empty() || setId < 0 || setId >= MAX_DYNA_SETS_PER_FRAMEN) {
        return cv::Mat();
    }
    cv::Mat mask(map.rows, map.cols, CV_8UC1, cv::Scalar(0));
    const uint8_t target = static_cast<uint8_t>(setId);
    for (int y = 0; y < map.rows; ++y) {
        const uint8_t* src = map.ptr<uint8_t>(y);
        uint8_t* dst = mask.ptr<uint8_t>(y);
        for (int x = 0; x < map.cols; ++x) {
            if (src[x] == target) {
                dst[x] = 1;
            }
        }
    }
    return mask;
}

void MainWindow::updateMaskPreviewIcons()
{
    if (!m_maskList) {
        return;
    }
    const cv::Vec3b color(200, 0, 200);
    const int padding = 6;
    const int textHeight = m_maskList->fontMetrics().height() + 4;
    const int gap = 2;
    const int width = kPreviewIconWidth;
    int height = kPreviewIconHeight;
    if (m_serumDataLoaded && m_serumData.fwidth > 0) {
        height = std::max(1, static_cast<int>(
            std::lround(static_cast<double>(width) * m_serumData.fheight / m_serumData.fwidth)));
    } else if (!m_compMasks.empty()) {
        const cv::Mat& first = m_compMasks.front();
        if (!first.empty() && first.cols > 0) {
            height = std::max(1, static_cast<int>(
                std::lround(static_cast<double>(width) * first.rows / first.cols)));
        }
    } else if (m_frameStore && m_frameStore->count() > 0) {
        if (const cv::Mat* frame = m_frameStore->at(0)) {
            if (frame && frame->cols > 0) {
                height = std::max(1, static_cast<int>(
                    std::lround(static_cast<double>(width) * frame->rows / frame->cols)));
            }
        }
    }
    m_maskList->setIconSize(QSize(width, height));
    m_maskList->setGridSize(QSize(width + padding * 2,
                                  height + textHeight + padding * 2 + gap));
    for (int i = 0; i < m_maskList->count() && i < MAX_MASKS; ++i) {
        cv::Mat mask;
        if (m_serumDataLoaded && m_serumData.compmasks.hasData(static_cast<uint32_t>(i))) {
            mask = cv::Mat(static_cast<int>(m_serumData.fheight),
                           static_cast<int>(m_serumData.fwidth),
                           CV_8UC1,
                           const_cast<uint8_t*>(m_serumData.compmasks[static_cast<uint32_t>(i)]));
        } else if (i < static_cast<int>(m_compMasks.size())) {
            mask = m_compMasks[static_cast<std::size_t>(i)];
        }
        cv::Mat iconMat = buildMaskIconImage(mask, color);
        if (iconMat.empty()) {
            if (QListWidgetItem* item = m_maskList->item(i)) {
                item->setIcon(QIcon());
                item->setData(kPreviewIconSizeRole, QSize(width, height));
                const int hintWidth = width + padding * 2;
                const int hintHeight = height + textHeight + padding * 2 + gap;
                item->setSizeHint(QSize(hintWidth, hintHeight));
            }
            continue;
        }
        cv::Mat rgb;
        cv::cvtColor(iconMat, rgb, cv::COLOR_BGR2RGB);
        QImage iconImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
        QPixmap pixmap = QPixmap::fromImage(iconImage.copy());
        pixmap = pixmap.scaled(QSize(width, height), Qt::IgnoreAspectRatio, Qt::FastTransformation);
        if (QListWidgetItem* item = m_maskList->item(i)) {
            item->setIcon(QIcon(pixmap));
            item->setData(kPreviewIconSizeRole, pixmap.size());
            const int hintWidth = pixmap.width() + padding * 2;
            const int hintHeight = pixmap.height() + textHeight + padding * 2 + gap;
            item->setSizeHint(QSize(hintWidth, hintHeight));
        }
    }
}

void MainWindow::updateDynamicMaskPreviewIcons()
{
    if (!m_dynamicMaskList) {
        return;
    }
    const int frameIndex = m_framesList ? m_framesList->currentRow() : -1;
    cv::Mat map;
    std::vector<uint16_t> serumColors;
    const std::vector<uint16_t>* colors = nullptr;
    const cv::Mat* localMap = (frameIndex >= 0 && frameIndex < static_cast<int>(m_frameDynamicMaskMaps.size()))
        ? &m_frameDynamicMaskMaps[static_cast<std::size_t>(frameIndex)]
        : nullptr;
    const std::vector<uint16_t>* localColors = (frameIndex >= 0 && frameIndex < static_cast<int>(m_frameDynamicColors.size()))
        ? &m_frameDynamicColors[static_cast<std::size_t>(frameIndex)]
        : nullptr;
    if (localMap && !localMap->empty()) {
        map = *localMap;
        if (localColors && !localColors->empty()) {
            colors = localColors;
        }
    }
    if (map.empty() && m_serumDataLoaded && frameIndex >= 0 &&
        frameIndex < static_cast<int>(m_serumData.nframes) &&
        m_serumData.dynamasks.hasData(static_cast<uint32_t>(frameIndex))) {
        map = cv::Mat(static_cast<int>(m_serumData.fheight),
                      static_cast<int>(m_serumData.fwidth),
                      CV_8UC1,
                      const_cast<uint8_t*>(m_serumData.dynamasks[static_cast<uint32_t>(frameIndex)]));
    }
    if (!colors) {
        if (localColors && !localColors->empty()) {
            colors = localColors;
        } else if (m_serumDataLoaded && frameIndex >= 0 &&
                   frameIndex < static_cast<int>(m_serumData.nframes) &&
                   m_serumData.dyna4cols_v2.hasData(static_cast<uint32_t>(frameIndex))) {
            const std::size_t size = static_cast<std::size_t>(MAX_DYNA_SETS_PER_FRAMEN) *
                std::max<uint32_t>(1, m_serumData.nocolors);
            const uint16_t* ptr = m_serumData.dyna4cols_v2[static_cast<uint32_t>(frameIndex)];
            serumColors.assign(ptr, ptr + size);
            colors = &serumColors;
        }
    }
    const int colorsPerSet = colors ? dynamicColorsPerSet(*colors) : 0;
    int previewHeight = kPreviewIconHeight;
    if (!map.empty() && map.cols > 0) {
        previewHeight = std::max(1, static_cast<int>(
            std::lround(static_cast<double>(kPreviewIconWidth) * map.rows / map.cols)));
    } else {
        previewHeight = std::max(1, kPreviewIconWidth / 4);
    }
    const int padding = 6;
    const int textHeight = m_dynamicMaskList->fontMetrics().height() + 4;
    const int gap = 2;
    m_dynamicMaskList->setGridSize(QSize(kDynamicMaskIconWidth + padding * 2,
                                         previewHeight + textHeight + padding * 2 + gap));
    const cv::Vec3b color(0, 200, 255);
    for (int i = 0; i < m_dynamicMaskList->count() && i < MAX_DYNA_SETS_PER_FRAMEN; ++i) {
        cv::Mat iconMat;
        if (!map.empty()) {
            cv::Mat mask = buildDynamicMaskFromMap(map, i);
            if (MaskHasContent(mask)) {
                iconMat = buildMaskIconImage(mask, color);
            }
        }
        QPixmap pixmap(kDynamicMaskIconWidth, previewHeight);
        pixmap.fill(Qt::transparent);
        QRect maskRect(0, 0, kPreviewIconWidth, previewHeight);
        QPainter painter(&pixmap);
        if (!iconMat.empty()) {
            cv::Mat rgb;
            cv::cvtColor(iconMat, rgb, cv::COLOR_BGR2RGB);
            QImage iconImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
            QPixmap maskPixmap = QPixmap::fromImage(iconImage.copy());
            const QPixmap scaled = maskPixmap.scaled(kPreviewIconWidth,
                                                     previewHeight,
                                                     Qt::IgnoreAspectRatio,
                                                     Qt::FastTransformation);
            const QPoint topLeft(0, 0);
            painter.drawPixmap(topLeft, scaled);
            maskRect = QRect(topLeft, scaled.size());
        } else {
            painter.fillRect(0, 0, kPreviewIconWidth, previewHeight, QColor(24, 24, 24));
        }
        if (colorsPerSet > 0 && colors && colors->size() >= static_cast<std::size_t>((i + 1) * colorsPerSet)) {
            const int cols = (colorsPerSet <= 4) ? colorsPerSet : 8;
            const int rows = (colorsPerSet <= 4) ? 1 : 2;
            const int stripX = kPreviewIconWidth + 8;
            const int stripWidth = kDynamicColorStripWidth;
            const int cellWidth = std::max(1, stripWidth / std::max(1, cols));
            const int stripHeight = maskRect.height();
            const int stripTop = maskRect.top();
            const int cellHeight = std::max(1, stripHeight / std::max(1, rows));
            painter.setPen(QColor(0, 0, 0));
            for (int slot = 0; slot < colorsPerSet; ++slot) {
                const int row = (colorsPerSet <= 4) ? 0 : (slot / cols);
                const int col = (colorsPerSet <= 4) ? slot : (slot % cols);
                if (row >= rows) {
                    break;
                }
                const std::size_t colorIndex = static_cast<std::size_t>(i) * colorsPerSet +
                    static_cast<std::size_t>(slot);
                const uint16_t value = (*colors)[colorIndex];
                const cv::Vec3b bgr = Rgb565ToBgr(value);
                const QColor swatch(bgr[2], bgr[1], bgr[0]);
                const int x = stripX + col * cellWidth;
                const int y = stripTop + row * cellHeight;
                painter.fillRect(QRect(x, y, cellWidth, cellHeight), swatch);
                painter.drawRect(QRect(x, y, cellWidth, cellHeight));
            }
        }
        if (QListWidgetItem* item = m_dynamicMaskList->item(i)) {
            item->setIcon(QIcon(pixmap));
            item->setData(kPreviewIconSizeRole, pixmap.size());
            const int padding = 6;
            const int textHeight = m_dynamicMaskList->fontMetrics().height() + 4;
            const int gap = 2;
            const int hintWidth = pixmap.width() + padding * 2;
            const int hintHeight = pixmap.height() + textHeight + padding * 2 + gap;
            item->setSizeHint(QSize(hintWidth, hintHeight));
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
            const QSize hint = m_maskList->gridSize().isValid()
                ? m_maskList->gridSize()
                : QSize(kPreviewIconWidth + 12, kPreviewIconHeight + 24);
            item->setSizeHint(hint);
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
    for (int i = 0; i < count; ++i) {
        QListWidgetItem* item = m_dynamicMaskList->item(i);
        const int oldIndex = item ? item->data(Qt::UserRole).toInt() : i;
        if (oldIndex >= 0 && oldIndex < count) {
            mapping[static_cast<std::size_t>(oldIndex)] = i;
        }
    }
    for (auto& map : m_frameDynamicMaskMaps) {
        if (map.empty()) {
            continue;
        }
        for (int y = 0; y < map.rows; ++y) {
            uint8_t* row = map.ptr<uint8_t>(y);
            for (int x = 0; x < map.cols; ++x) {
                const uint8_t value = row[x];
                if (value == 255) {
                    continue;
                }
                if (value < mapping.size() && mapping[value] >= 0) {
                    row[x] = static_cast<uint8_t>(mapping[value]);
                }
            }
        }
    }
    for (auto& map : m_frameDynamicMaskMapsX) {
        if (map.empty()) {
            continue;
        }
        for (int y = 0; y < map.rows; ++y) {
            uint8_t* row = map.ptr<uint8_t>(y);
            for (int x = 0; x < map.cols; ++x) {
                const uint8_t value = row[x];
                if (value == 255) {
                    continue;
                }
                if (value < mapping.size() && mapping[value] >= 0) {
                    row[x] = static_cast<uint8_t>(mapping[value]);
                }
            }
        }
    }
    for (auto& colors : m_frameDynamicColors) {
        const int colorsPerSet = dynamicColorsPerSet(colors);
        if (colorsPerSet <= 0 || colors.size() < static_cast<std::size_t>(MAX_DYNA_SETS_PER_FRAMEN * colorsPerSet)) {
            continue;
        }
        std::vector<uint16_t> reordered(colors.size(), 0);
        for (int oldIndex = 0; oldIndex < count; ++oldIndex) {
            const int newIndex = mapping[static_cast<std::size_t>(oldIndex)];
            if (newIndex < 0 || newIndex >= count) {
                continue;
            }
            const std::size_t src = static_cast<std::size_t>(oldIndex) * colorsPerSet;
            const std::size_t dst = static_cast<std::size_t>(newIndex) * colorsPerSet;
            std::memcpy(reordered.data() + dst, colors.data() + src, colorsPerSet * sizeof(uint16_t));
        }
        colors.swap(reordered);
    }
    for (int i = 0; i < count; ++i) {
        QListWidgetItem* item = m_dynamicMaskList->item(i);
        if (item) {
            item->setText(QString("Dynamic %1").arg(i));
            item->setData(Qt::UserRole, i);
            item->setData(Qt::UserRole + 1, QStringLiteral("dynamic"));
            const QSize hint = m_dynamicMaskList->gridSize().isValid()
                ? m_dynamicMaskList->gridSize()
                : QSize(kDynamicMaskIconWidth + 12, kPreviewIconHeight + 24);
            item->setSizeHint(hint);
        }
    }
    updateDynamicMaskPreviewIcons();
    refreshDynamicPaletteButtons();
    if (m_frameDynamicMaskAssign) {
        const int selected = currentFrameDynamicMaskId();
        QSignalBlocker blocker(m_frameDynamicMaskAssign);
        m_frameDynamicMaskAssign->setCurrentIndex(selected >= 0 ? selected + 1 : 0);
    }
    updateMaskPreviewForFrame(m_framesList->currentRow());
    m_dynamicMaskReorderActive = false;
}

int MainWindow::currentFrameMaskId() const
{
    const int row = m_framesList ? m_framesList->currentRow() : -1;
    if (row < 0) {
        return -1;
    }
    if (m_serumDataLoaded && row < static_cast<int>(m_serumData.nframes) &&
        m_serumData.compmaskID.hasData(static_cast<uint32_t>(row))) {
        const uint8_t* value = m_serumData.compmaskID[static_cast<uint32_t>(row)];
        if (!value) {
            return -1;
        }
        return value[0] == 255 ? -1 : static_cast<int>(value[0]);
    }
    if (row >= static_cast<int>(m_frameCompMaskIds.size())) {
        return -1;
    }
    const uint8_t value = m_frameCompMaskIds[static_cast<std::size_t>(row)];
    return value == 255 ? -1 : static_cast<int>(value);
}

int MainWindow::currentFrameDynamicMaskId() const
{
    if (m_dynamicMaskList && m_dynamicMaskList->currentRow() >= 0) {
        return m_dynamicMaskList->currentRow();
    }
    if (m_frameDynamicMaskAssign) {
        const int id = m_frameDynamicMaskAssign->currentData().toInt();
        if (id >= 0 && id < MAX_DYNA_SETS_PER_FRAMEN) {
            return id;
        }
    }
    return -1;
}

void MainWindow::setCurrentFrameMaskId(int id)
{
    const int row = m_framesList ? m_framesList->currentRow() : -1;
    if (row < 0 || row >= static_cast<int>(m_frameCompMaskIds.size())) {
        return;
    }
    const uint8_t value = (id < 0) ? 255 : static_cast<uint8_t>(id);
    m_frameCompMaskIds[static_cast<std::size_t>(row)] = value;
    if (m_serumDataLoaded && row < static_cast<int>(m_serumData.nframes)) {
        m_serumData.compmaskID.set(static_cast<uint32_t>(row), &value, 1);
    }
}

void MainWindow::setCurrentFrameDynamicMaskId(int id)
{
    const int clamped = (id >= 0 && id < MAX_DYNA_SETS_PER_FRAMEN) ? id : -1;
    if (m_dynamicMaskList) {
        QSignalBlocker blocker(m_dynamicMaskList);
        m_dynamicMaskList->setCurrentRow(clamped);
    }
    if (m_frameDynamicMaskAssign) {
        QSignalBlocker blocker(m_frameDynamicMaskAssign);
        m_frameDynamicMaskAssign->setCurrentIndex(clamped >= 0 ? clamped + 1 : 0);
    }
}

cv::Mat* MainWindow::activeComparisonMask()
{
    const int id = m_maskList ? m_maskList->currentRow() : -1;
    if (id < 0 || id >= MAX_MASKS) {
        return nullptr;
    }
    if (m_compMasks.size() != MAX_MASKS) {
        m_compMasks.resize(MAX_MASKS);
    }
    cv::Mat& mask = m_compMasks[static_cast<std::size_t>(id)];
    if (mask.empty() && m_serumDataLoaded && m_serumData.compmasks.hasData(static_cast<uint32_t>(id))) {
        const int width = static_cast<int>(m_serumData.fwidth);
        const int height = static_cast<int>(m_serumData.fheight);
        if (width > 0 && height > 0) {
            mask = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
            const uint8_t* src = m_serumData.compmasks[static_cast<uint32_t>(id)];
            if (src) {
                std::memcpy(mask.data, src, static_cast<std::size_t>(width) * height);
            }
        }
    }
    return &mask;
}

cv::Mat* MainWindow::activeDynamicMaskMap(int frameIndex)
{
    if (frameIndex < 0 || frameIndex >= static_cast<int>(m_frameDynamicMaskMaps.size())) {
        return nullptr;
    }
    cv::Mat& map = m_frameDynamicMaskMaps[static_cast<std::size_t>(frameIndex)];
    if (map.empty() && m_serumDataLoaded && frameIndex < static_cast<int>(m_serumData.nframes) &&
        m_serumData.dynamasks.hasData(static_cast<uint32_t>(frameIndex))) {
        const int width = static_cast<int>(m_serumData.fwidth);
        const int height = static_cast<int>(m_serumData.fheight);
        if (width > 0 && height > 0) {
            map = cv::Mat(height, width, CV_8UC1, cv::Scalar(255));
            const uint8_t* src = m_serumData.dynamasks[static_cast<uint32_t>(frameIndex)];
            if (src) {
                std::memcpy(map.data, src, static_cast<std::size_t>(width) * height);
            }
        }
    }
    return &map;
}

cv::Mat* MainWindow::activeSpriteDynamicMask(int spriteIndex)
{
    if (spriteIndex < 0) {
        return nullptr;
    }
    if (m_useHdSprite) {
        if (auto* mask = ensureHdSpriteDynamicMaskLocal(spriteIndex)) {
            return mask;
        }
    }
    return ensureSpriteDynamicMaskLocal(spriteIndex);
}

cv::Mat* MainWindow::activeBackgroundMask(int index)
{
    if (index < 0) {
        return nullptr;
    }
    if (m_useHdFrame && index < static_cast<int>(m_frameBackgroundMasksX.size())) {
        cv::Mat& mask = m_frameBackgroundMasksX[static_cast<std::size_t>(index)];
        if (mask.empty() && m_serumDataLoaded && index < static_cast<int>(m_serumData.nframes) &&
            m_serumData.backgroundmask_extra.hasData(static_cast<uint32_t>(index))) {
            const int width = static_cast<int>(m_serumData.fwidth_extra);
            const int height = static_cast<int>(m_serumData.fheight_extra);
            if (width > 0 && height > 0) {
                mask = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
                const uint8_t* src = m_serumData.backgroundmask_extra[static_cast<uint32_t>(index)];
                if (src) {
                    std::memcpy(mask.data, src, static_cast<std::size_t>(width) * height);
                }
            }
        }
        return &mask;
    }
    if (index < static_cast<int>(m_frameBackgroundMasks.size())) {
        cv::Mat& mask = m_frameBackgroundMasks[static_cast<std::size_t>(index)];
        if (mask.empty() && m_serumDataLoaded && index < static_cast<int>(m_serumData.nframes) &&
            m_serumData.backgroundmask.hasData(static_cast<uint32_t>(index))) {
            const int width = static_cast<int>(m_serumData.fwidth);
            const int height = static_cast<int>(m_serumData.fheight);
            if (width > 0 && height > 0) {
                mask = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
                const uint8_t* src = m_serumData.backgroundmask[static_cast<uint32_t>(index)];
                if (src) {
                    std::memcpy(mask.data, src, static_cast<std::size_t>(width) * height);
                }
            }
        }
        return &mask;
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

void MainWindow::setSpriteZoneMode(bool enabled)
{
    if (m_spriteZoneMode == enabled) {
        return;
    }
    m_spriteZoneMode = enabled;
    m_frameDrawOnZone = false;
    m_frameHasStart = false;
    if (enabled) {
        if (m_maskMode != MaskMode::None) {
            m_maskMode = MaskMode::None;
        }
        if (m_backgroundMaskMode) {
            m_backgroundMaskMode = false;
            if (m_framesCanvas) {
                m_framesCanvas->setBackgroundMaskChecked(false);
            }
        }
        if (m_framesCanvas) {
            m_framesCanvas->setMaskButtonsChecked(false, false);
        }
    }
    refreshSpriteZoneList();
    if (m_framesList && m_framesList->currentRow() >= 0) {
        updateFrameCanvasImage(m_framesList->currentRow());
    }
    updateMaskPreviewForFrame(m_framesList ? m_framesList->currentRow() : -1);
}

void MainWindow::setSpriteDetAreaMode(bool enabled)
{
    if (m_spriteDetAreaMode == enabled) {
        return;
    }
    m_spriteDetAreaMode = enabled;
    cancelCurrentDraw();
    if (enabled && m_spriteDynamicMaskMode) {
        m_spriteDynamicMaskMode = false;
    }
    if (m_spritesCanvas) {
        m_spritesCanvas->setMaskButtonsChecked(m_spriteDetAreaMode, m_spriteDynamicMaskMode);
    }
    updateSpriteCanvasImage(m_spritesList ? m_spritesList->currentRow() : -1);
    updateUndoActions();
}

void MainWindow::swapDynamicMaskEntries(int a, int b)
{
    if (a < 0 || b < 0 || a >= MAX_DYNA_SETS_PER_FRAMEN || b >= MAX_DYNA_SETS_PER_FRAMEN) {
        return;
    }
    for (auto& map : m_frameDynamicMaskMaps) {
        if (map.empty()) {
            continue;
        }
        for (int y = 0; y < map.rows; ++y) {
            uint8_t* row = map.ptr<uint8_t>(y);
            for (int x = 0; x < map.cols; ++x) {
                if (row[x] == a) {
                    row[x] = static_cast<uint8_t>(b);
                } else if (row[x] == b) {
                    row[x] = static_cast<uint8_t>(a);
                }
            }
        }
    }
    for (auto& map : m_frameDynamicMaskMapsX) {
        if (map.empty()) {
            continue;
        }
        for (int y = 0; y < map.rows; ++y) {
            uint8_t* row = map.ptr<uint8_t>(y);
            for (int x = 0; x < map.cols; ++x) {
                if (row[x] == a) {
                    row[x] = static_cast<uint8_t>(b);
                } else if (row[x] == b) {
                    row[x] = static_cast<uint8_t>(a);
                }
            }
        }
    }
    for (auto& colors : m_frameDynamicColors) {
        const int colorsPerSet = dynamicColorsPerSet(colors);
        if (colorsPerSet <= 0 ||
            colors.size() < static_cast<std::size_t>(MAX_DYNA_SETS_PER_FRAMEN * colorsPerSet)) {
            continue;
        }
        const std::size_t aOffset = static_cast<std::size_t>(a) * colorsPerSet;
        const std::size_t bOffset = static_cast<std::size_t>(b) * colorsPerSet;
        for (int i = 0; i < colorsPerSet; ++i) {
            std::swap(colors[aOffset + i], colors[bOffset + i]);
        }
    }
    updateDynamicMaskPreviewIcons();
    refreshDynamicPaletteButtons();
    if (m_frameDynamicMaskAssign) {
        const int selected = currentFrameDynamicMaskId();
        QSignalBlocker blocker(m_frameDynamicMaskAssign);
        m_frameDynamicMaskAssign->setCurrentIndex(selected >= 0 ? selected + 1 : 0);
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

void MainWindow::applyToolToDynamicMask(cv::Mat& map,
                                        int setId,
                                        DrawTool tool,
                                        const QPoint& start,
                                        const QPoint& end,
                                        bool erase)
{
    if (map.empty() || setId < 0 || setId >= MAX_DYNA_SETS_PER_FRAMEN) {
        return;
    }
    const uint8_t value = erase ? 255 : static_cast<uint8_t>(setId);
    if (tool == DrawTool::Point) {
        if (start.x() >= 0 && start.x() < map.cols && start.y() >= 0 && start.y() < map.rows) {
            map.at<uint8_t>(start.y(), start.x()) = value;
        }
        return;
    }
    if (tool == DrawTool::Line) {
        cv::line(map, cv::Point(start.x(), start.y()), cv::Point(end.x(), end.y()), cv::Scalar(value), 1);
        return;
    }
    if (tool == DrawTool::Rect || tool == DrawTool::RectFill) {
        const cv::Point tl(std::min(start.x(), end.x()), std::min(start.y(), end.y()));
        const cv::Point br(std::max(start.x(), end.x()), std::max(start.y(), end.y()));
        const int thickness = (tool == DrawTool::RectFill) ? -1 : 1;
        cv::rectangle(map, cv::Rect(tl, br), cv::Scalar(value), thickness);
        return;
    }
    if (tool == DrawTool::Circle || tool == DrawTool::CircleFill) {
        const int dx = end.x() - start.x();
        const int dy = end.y() - start.y();
        const int radius = static_cast<int>(std::sqrt(dx * dx + dy * dy));
        const int thickness = (tool == DrawTool::CircleFill) ? -1 : 1;
        cv::circle(map, cv::Point(start.x(), start.y()), radius, cv::Scalar(value), thickness);
        return;
    }
    if (tool == DrawTool::Ellipse || tool == DrawTool::EllipseFill) {
        const cv::Point center((start.x() + end.x()) / 2, (start.y() + end.y()) / 2);
        const cv::Size axes(std::abs(end.x() - start.x()) / 2, std::abs(end.y() - start.y()) / 2);
        const int thickness = (tool == DrawTool::EllipseFill) ? -1 : 1;
        cv::ellipse(map, center, axes, 0.0, 0.0, 360.0, cv::Scalar(value), thickness);
        return;
    }
}

void MainWindow::applyDynamicMaskFill(cv::Mat& map, int setId, int x, int y, bool erase)
{
    if (map.empty() || setId < 0 || setId >= MAX_DYNA_SETS_PER_FRAMEN) {
        return;
    }
    if (x < 0 || y < 0 || x >= map.cols || y >= map.rows) {
        return;
    }
    const uint8_t value = erase ? 255 : static_cast<uint8_t>(setId);
    const uint8_t target = map.at<uint8_t>(y, x);
    if (target == value) {
        return;
    }
    cv::Mat floodMask(map.rows + 2, map.cols + 2, CV_8UC1, cv::Scalar(0));
    cv::floodFill(map, floodMask, cv::Point(x, y), cv::Scalar(value), nullptr, cv::Scalar(0), cv::Scalar(0), 4);
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
    m_paletteUndo.clear();
    m_paletteRedo.clear();
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

void MainWindow::resetNavigationHistory()
{
    m_frameHistory = NavigationHistory{};
    m_spriteHistory = NavigationHistory{};
    m_imageHistory = NavigationHistory{};
    m_backgroundHistory = NavigationHistory{};
    m_frameHistory.current = m_framesList ? m_framesList->currentRow() : -1;
    m_spriteHistory.current = m_spritesList ? m_spritesList->currentRow() : -1;
    m_imageHistory.current = m_imagesList ? m_imagesList->currentRow() : -1;
    m_backgroundHistory.current = m_backgroundList ? m_backgroundList->currentRow() : -1;
    updateNavigationButtons();
}

void MainWindow::recordHistory(NavigationHistory& history, int newIndex)
{
    if (history.navigating) {
        history.current = newIndex;
        history.navigating = false;
        updateNavigationButtons();
        return;
    }
    if (newIndex < 0) {
        history.current = newIndex;
        updateNavigationButtons();
        return;
    }
    if (history.current >= 0 && history.current != newIndex) {
        history.back.push_back(history.current);
        while (history.back.size() > m_maxHistoryDepth) {
            history.back.pop_front();
        }
        history.forward.clear();
    }
    history.current = newIndex;
    updateNavigationButtons();
}

bool MainWindow::navigateHistory(NavigationHistory& history, QListWidget* list, bool forward)
{
    if (!list) {
        return false;
    }
    QVector<int>& source = forward ? history.forward : history.back;
    QVector<int>& target = forward ? history.back : history.forward;
    if (source.isEmpty()) {
        updateNavigationButtons();
        return false;
    }
    const int next = source.takeLast();
    if (history.current >= 0) {
        target.push_back(history.current);
        while (target.size() > m_maxHistoryDepth) {
            target.pop_front();
        }
    }
    history.navigating = true;
    list->setCurrentRow(next);
    history.current = next;
    updateNavigationButtons();
    return true;
}

void MainWindow::updateNavigationButtons()
{
    if (m_framesCanvas) {
        m_framesCanvas->setBackEnabled(!m_frameHistory.back.isEmpty());
        m_framesCanvas->setForwardEnabled(!m_frameHistory.forward.isEmpty());
    }
    if (m_spritesCanvas) {
        m_spritesCanvas->setBackEnabled(!m_spriteHistory.back.isEmpty());
        m_spritesCanvas->setForwardEnabled(!m_spriteHistory.forward.isEmpty());
    }
    if (m_imagesCanvas) {
        m_imagesCanvas->setBackEnabled(!m_imageHistory.back.isEmpty());
        m_imagesCanvas->setForwardEnabled(!m_imageHistory.forward.isEmpty());
    }
    if (m_backgroundsCanvas) {
        m_backgroundsCanvas->setBackEnabled(!m_backgroundHistory.back.isEmpty());
        m_backgroundsCanvas->setForwardEnabled(!m_backgroundHistory.forward.isEmpty());
    }
}

void MainWindow::trimUndoStacks()
{
    const auto trimStack = [this](UndoStack& stack) {
        while (stack.undo.size() > static_cast<std::size_t>(m_maxUndoDepth)) {
            stack.undo.erase(stack.undo.begin());
        }
        while (stack.redo.size() > static_cast<std::size_t>(m_maxUndoDepth)) {
            stack.redo.erase(stack.redo.begin());
        }
    };
    for (auto& stack : m_frameUndoStacks) {
        trimStack(stack);
    }
    for (auto& stack : m_frameHdUndoStacks) {
        trimStack(stack);
    }
    for (auto& stack : m_spriteUndoStacks) {
        trimStack(stack);
    }
    for (auto& stack : m_backgroundUndoStacks) {
        trimStack(stack);
    }
    for (auto& stack : m_backgroundHdUndoStacks) {
        trimStack(stack);
    }
    for (auto& stack : m_compMaskUndoStacks) {
        trimStack(stack);
    }
    for (auto& stack : m_dynMaskUndoStacks) {
        trimStack(stack);
    }
    for (auto& stack : m_backgroundMaskUndoStacks) {
        trimStack(stack);
    }
    while (m_paletteUndo.size() > static_cast<std::size_t>(m_maxUndoDepth)) {
        m_paletteUndo.erase(m_paletteUndo.begin());
    }
    while (m_paletteRedo.size() > static_cast<std::size_t>(m_maxUndoDepth)) {
        m_paletteRedo.erase(m_paletteRedo.begin());
    }
    updateUndoActions();
}

void MainWindow::showSettingsDialog()
{
    QDialog dialog(this);
    dialog.setWindowTitle("Settings");
    auto* layout = new QFormLayout(&dialog);
    auto* historySpin = new QSpinBox(&dialog);
    historySpin->setRange(1, 1000);
    historySpin->setValue(m_maxHistoryDepth);
    auto* undoSpin = new QSpinBox(&dialog);
    undoSpin->setRange(1, 1000);
    undoSpin->setValue(m_maxUndoDepth);
    auto* frameCacheSpin = new QSpinBox(&dialog);
    frameCacheSpin->setRange(1, 256);
    frameCacheSpin->setValue(m_frameCacheLimit);
    auto* spriteCacheSpin = new QSpinBox(&dialog);
    spriteCacheSpin->setRange(1, 256);
    spriteCacheSpin->setValue(m_spriteCacheLimit);
    auto* backgroundCacheSpin = new QSpinBox(&dialog);
    backgroundCacheSpin->setRange(1, 256);
    backgroundCacheSpin->setValue(m_backgroundCacheLimit);
    auto* loggingCheck = new QCheckBox(&dialog);
    loggingCheck->setChecked(m_loggingEnabled);
    auto* autosaveCheck = new QCheckBox(&dialog);
    autosaveCheck->setChecked(m_autosaveEnabled);
    auto* autosaveSpin = new QSpinBox(&dialog);
    autosaveSpin->setRange(1, 120);
    autosaveSpin->setValue(m_autosaveIntervalMinutes);
    autosaveSpin->setSuffix(" min");
    autosaveSpin->setEnabled(m_autosaveEnabled);
    connect(autosaveCheck, &QCheckBox::toggled, autosaveSpin, &QSpinBox::setEnabled);
    layout->addRow("History depth", historySpin);
    layout->addRow("Undo depth", undoSpin);
    layout->addRow("Frame cache size", frameCacheSpin);
    layout->addRow("Sprite cache size", spriteCacheSpin);
    layout->addRow("Background cache size", backgroundCacheSpin);
    layout->addRow("Enable logging", loggingCheck);
    layout->addRow("Enable autosave", autosaveCheck);
    layout->addRow("Autosave interval", autosaveSpin);
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    layout->addWidget(buttons);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    if (dialog.exec() != QDialog::Accepted) {
        return;
    }
    m_maxHistoryDepth = std::clamp(historySpin->value(), 1, 1000);
    m_maxUndoDepth = std::clamp(undoSpin->value(), 1, 1000);
    m_frameCacheLimit = std::clamp(frameCacheSpin->value(), 1, 256);
    m_spriteCacheLimit = std::clamp(spriteCacheSpin->value(), 1, 256);
    m_backgroundCacheLimit = std::clamp(backgroundCacheSpin->value(), 1, 256);
    m_loggingEnabled = loggingCheck->isChecked();
    m_autosaveEnabled = autosaveCheck->isChecked();
    m_autosaveIntervalMinutes = std::clamp(autosaveSpin->value(), 1, 120);
    const auto trimHistory = [this](NavigationHistory& history) {
        while (history.back.size() > m_maxHistoryDepth) {
            history.back.pop_front();
        }
        while (history.forward.size() > m_maxHistoryDepth) {
            history.forward.pop_front();
        }
    };
    trimHistory(m_frameHistory);
    trimHistory(m_spriteHistory);
    trimHistory(m_imageHistory);
    trimHistory(m_backgroundHistory);
    updateNavigationButtons();
    trimUndoStacks();
    if (m_frameStore) {
        m_frameStore->setCacheLimit(m_frameCacheLimit);
    }
    if (m_spriteStore) {
        m_spriteStore->setCacheLimit(m_spriteCacheLimit);
    }
    if (m_backgroundStore) {
        m_backgroundStore->setCacheLimit(m_backgroundCacheLimit);
    }
    if (m_loggingEnabled) {
        initLogging();
    } else {
        shutdownLogging();
    }
    updateAutosaveTimer();
    QSettings settings("PPUC", "PPUC-Serum-Colorizer");
    settings.setValue("maxHistoryDepth", m_maxHistoryDepth);
    settings.setValue("maxUndoDepth", m_maxUndoDepth);
    settings.setValue("frameCacheLimit", m_frameCacheLimit);
    settings.setValue("spriteCacheLimit", m_spriteCacheLimit);
    settings.setValue("backgroundCacheLimit", m_backgroundCacheLimit);
    settings.setValue("loggingEnabled", m_loggingEnabled);
    settings.setValue("logPath", m_logPath);
    settings.setValue("autosaveEnabled", m_autosaveEnabled);
    settings.setValue("autosaveIntervalMinutes", m_autosaveIntervalMinutes);
    settings.sync();
}

void MainWindow::closeEvent(QCloseEvent* event)
{
    logLine("Shutdown: clean");
    QSettings settings("PPUC", "PPUC-Serum-Colorizer");
    settings.setValue("lastShutdownClean", true);
    settings.setValue("autosaveEnabled", m_autosaveEnabled);
    settings.setValue("autosaveIntervalMinutes", m_autosaveIntervalMinutes);
    settings.sync();
    disposeSerumRuntime();
    shutdownLogging();
    QMainWindow::closeEvent(event);
}

void MainWindow::initLogging()
{
    if (!m_loggingEnabled) {
        shutdownLogging();
        return;
    }
    if (!m_logFile) {
        m_logFile = new QFile(this);
    }
    if (m_logPath.isEmpty()) {
        const QString defaultLogDir = QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation);
        const QString logDir = defaultLogDir.isEmpty()
            ? QDir::home().filePath(".ppuc-serum-colorizer")
            : defaultLogDir;
        QDir().mkpath(logDir);
        m_logPath = QDir(logDir).filePath("ppuc-serum-colorizer.log");
    }
    if (!m_logFile->isOpen()) {
        m_logFile->setFileName(m_logPath);
        if (!m_logFile->open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text)) {
            return;
        }
    }
    logLine(QString("Startup: %1").arg(QCoreApplication::applicationVersion()));
}

void MainWindow::shutdownLogging()
{
    if (!m_logFile) {
        return;
    }
    if (m_logFile->isOpen()) {
        m_logFile->flush();
        m_logFile->close();
    }
}

void MainWindow::logLine(const QString& message)
{
    if (!m_loggingEnabled || !m_logFile) {
        return;
    }
    if (!m_logFile->isOpen()) {
        m_logFile->setFileName(m_logPath);
        if (!m_logFile->open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text)) {
            return;
        }
    }
    constexpr qint64 kMaxLogBytes = 1024 * 1024;
    const QFileInfo info(m_logPath);
    if (info.exists() && info.size() > kMaxLogBytes) {
        m_logFile->close();
        if (m_logFile->open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
            const QByteArray header = QByteArrayLiteral("[log] truncated\n");
            m_logFile->write(header);
            m_logFile->flush();
        }
        m_logFile->close();
        if (!m_logFile->open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text)) {
            return;
        }
    }
    const QString stamp = QDateTime::currentDateTime().toString(Qt::ISODate);
    const std::uint64_t rssBytes = currentRssBytes();
    const double rssMb = rssBytes > 0 ? static_cast<double>(rssBytes) / (1024.0 * 1024.0) : 0.0;
    const QString mem = rssBytes > 0 ? QString(" (rss %1 MB)").arg(rssMb, 0, 'f', 1) : QString();
    const QByteArray line = QString("[%1] %2%3\n").arg(stamp, message, mem).toUtf8();
    m_logFile->write(line);
    m_logFile->flush();
}

void MainWindow::markProjectDirty()
{
    m_projectDirty = true;
}

std::uint64_t MainWindow::currentRssBytes() const
{
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS counters{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        return static_cast<std::uint64_t>(counters.WorkingSetSize);
    }
    return 0;
#elif defined(__APPLE__)
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
        return static_cast<std::uint64_t>(info.phys_footprint);
    }
    return 0;
#elif defined(__linux__)
    std::FILE* file = std::fopen("/proc/self/statm", "r");
    if (!file) {
        return 0;
    }
    unsigned long size = 0;
    unsigned long resident = 0;
    if (std::fscanf(file, "%lu %lu", &size, &resident) != 2) {
        std::fclose(file);
        return 0;
    }
    std::fclose(file);
    const long pageSize = sysconf(_SC_PAGESIZE);
    if (pageSize <= 0) {
        return 0;
    }
    return static_cast<std::uint64_t>(resident) * static_cast<std::uint64_t>(pageSize);
#else
    return 0;
#endif
}

void MainWindow::showCrashLogDialog()
{
    if (m_logPath.isEmpty()) {
        return;
    }
    logLine("Detected unclean shutdown");
    QMessageBox box(this);
    box.setWindowTitle("Previous Crash Detected");
    box.setIcon(QMessageBox::Warning);
    box.setText("The app did not shut down correctly last time. "
                "You can copy the log to send for debugging.");
    box.setInformativeText(QString("Log file: %1").arg(m_logPath));
    auto* copyButton = box.addButton("Copy Log", QMessageBox::ActionRole);
    box.addButton(QMessageBox::Ok);
    box.exec();
    if (box.clickedButton() == copyButton) {
        QFile file(m_logPath);
        if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            const QByteArray data = file.readAll();
            QGuiApplication::clipboard()->setText(QString::fromUtf8(data));
            logLine("Crash log copied to clipboard");
        }
    }
}

void MainWindow::runDebugCheck()
{
    logLine("Debug check: begin");
    if (m_projectDir.isEmpty() || m_projectBaseName.isEmpty()) {
        logLine("Debug check: no project loaded");
        statusBar()->showMessage("Debug check: no project loaded.", 4000);
        return;
    }
    const int frameCount = m_frameStore ? m_frameStore->count() : 0;
    logLine(QString("Debug check: frames=%1 sprites=%2 backgrounds=%3")
                .arg(frameCount)
                .arg(m_spriteStore ? m_spriteStore->count() : 0)
                .arg(m_backgroundStore ? m_backgroundStore->count() : 0));
    const int currentIndex = m_framesList ? m_framesList->currentRow() : -1;
    if (frameCount > 0) {
        const cv::Mat first = renderFrameWithSerum(0, false);
        logLine(QString("Debug check: render frame0 empty=%1 size=%2x%3")
                    .arg(first.empty() ? "true" : "false")
                    .arg(first.cols)
                    .arg(first.rows));
    }
    if (currentIndex >= 0) {
        const cv::Mat current = renderFrameWithSerum(currentIndex, m_useHdFrame);
        logLine(QString("Debug check: render current=%1 hd=%2 empty=%3 size=%4x%5")
                    .arg(currentIndex)
                    .arg(m_useHdFrame ? "true" : "false")
                    .arg(current.empty() ? "true" : "false")
                    .arg(current.cols)
                    .arg(current.rows));
    }
    setSerumRuntimePathsForPlayback(false);
    disposeSerumRuntime();
    const bool runtimeOk = ensureSerumRuntime();
    logLine(QString("Debug check: serum runtime=%1").arg(runtimeOk ? "ok" : "failed"));
    if (runtimeOk) {
        disposeSerumRuntime();
    }
    statusBar()->showMessage("Debug check complete. See log for details.", 4000);
    logLine("Debug check: end");
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
    cv::Mat* image = isFrame ? activeFrameImage(index, true) : activeSpriteImageMutable(index);
    if (!image || image->empty()) {
        return;
    }
    markProjectDirty();
    UndoStack& stack = (*stacks)[static_cast<std::size_t>(index)];
    UndoState state;
    state.image = image->clone();
    stack.undo.push_back(std::move(state));
    if (stack.undo.size() > m_maxUndoDepth) {
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
        if (index >= 0 && index < static_cast<int>(m_frameDynamicMaskMaps.size())) {
            const cv::Mat& map = m_frameDynamicMaskMaps[static_cast<std::size_t>(index)];
            if (!map.empty()) {
                state.mask = map.clone();
                state.mask_kind = MaskKind::Dynamic;
                state.mask_index = index;
            }
        }
    }
    if (state.mask.empty()) {
        return;
    }
    markProjectDirty();
    stack.undo.push_back(std::move(state));
    if (stack.undo.size() > m_maxUndoDepth) {
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
    markProjectDirty();
    UndoStack& stack = (*stacks)[static_cast<std::size_t>(index)];
    UndoState state;
    state.image = image->clone();
    stack.undo.push_back(std::move(state));
    if (stack.undo.size() > m_maxUndoDepth) {
        stack.undo.erase(stack.undo.begin());
    }
    stack.redo.clear();
    updateUndoActions();
}

void MainWindow::pushSpriteDynamicMaskUndoSnapshot(int index)
{
    if (index < 0 || index >= static_cast<int>(m_spriteDynamicMasks.size())) {
        return;
    }
    if (index >= static_cast<int>(m_spriteUndoStacks.size())) {
        return;
    }
    const cv::Mat& map = m_spriteDynamicMasks[static_cast<std::size_t>(index)];
    if (map.empty()) {
        return;
    }
    markProjectDirty();
    UndoStack& stack = m_spriteUndoStacks[static_cast<std::size_t>(index)];
    UndoState state;
    state.mask = map.clone();
    state.mask_kind = MaskKind::Dynamic;
    state.mask_index = index;
    stack.undo.push_back(std::move(state));
    if (stack.undo.size() > m_maxUndoDepth) {
        stack.undo.erase(stack.undo.begin());
    }
    stack.redo.clear();
    updateUndoActions();
}

void MainWindow::pushSpriteDetAreaUndoSnapshot(int index)
{
    if (index < 0 || index >= static_cast<int>(m_spriteUndoStacks.size())) {
        return;
    }
    const std::size_t base = static_cast<std::size_t>(index) * MAX_SPRITE_DETECT_AREAS * 4;
    if (base + MAX_SPRITE_DETECT_AREAS * 4 > m_spriteDetAreas.size()) {
        return;
    }
    markProjectDirty();
    cv::Mat snapshot(1, MAX_SPRITE_DETECT_AREAS * 4, CV_16UC1);
    std::memcpy(snapshot.data,
                m_spriteDetAreas.data() + base,
                MAX_SPRITE_DETECT_AREAS * 4 * sizeof(uint16_t));
    UndoStack& stack = m_spriteUndoStacks[static_cast<std::size_t>(index)];
    UndoState state;
    state.mask = snapshot.clone();
    state.mask_kind = MaskKind::SpriteDetAreas;
    state.mask_index = index;
    stack.undo.push_back(std::move(state));
    if (stack.undo.size() > m_maxUndoDepth) {
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
    markProjectDirty();
    UndoStack& stack = m_backgroundMaskUndoStacks[static_cast<std::size_t>(index)];
    UndoState state;
    state.mask = mask->clone();
    state.mask_kind = MaskKind::None;
    stack.undo.push_back(std::move(state));
    if (stack.undo.size() > m_maxUndoDepth) {
        stack.undo.erase(stack.undo.begin());
    }
    stack.redo.clear();
    updateUndoActions();
}

bool MainWindow::undoEdit(bool isFrame)
{
    const auto undoAtIndex = [&](int index) -> bool {
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
        if (!isFrame) {
            UndoState previous = stack.undo.back();
            stack.undo.pop_back();
            if (previous.mask_kind == MaskKind::Dynamic && !previous.mask.empty()) {
                cv::Mat* map = activeSpriteDynamicMask(index);
                if (!map) {
                    return false;
                }
                UndoState current;
                current.mask = map->clone();
                current.mask_kind = MaskKind::Dynamic;
                current.mask_index = index;
                stack.redo.push_back(std::move(current));
                *map = previous.mask.clone();
                updateSpriteCanvasImage(index);
                return true;
            }
            if (previous.mask_kind == MaskKind::SpriteDetAreas && !previous.mask.empty()) {
                const std::size_t base = static_cast<std::size_t>(index) * MAX_SPRITE_DETECT_AREAS * 4;
                if (base + MAX_SPRITE_DETECT_AREAS * 4 > m_spriteDetAreas.size()) {
                    return false;
                }
                cv::Mat snapshot(1, MAX_SPRITE_DETECT_AREAS * 4, CV_16UC1);
                std::memcpy(snapshot.data,
                            m_spriteDetAreas.data() + base,
                            MAX_SPRITE_DETECT_AREAS * 4 * sizeof(uint16_t));
                UndoState current;
                current.mask = snapshot.clone();
                current.mask_kind = MaskKind::SpriteDetAreas;
                current.mask_index = index;
                stack.redo.push_back(std::move(current));
                if (previous.mask.total() >= static_cast<std::size_t>(MAX_SPRITE_DETECT_AREAS * 4)) {
                    std::memcpy(m_spriteDetAreas.data() + base,
                                previous.mask.data,
                                MAX_SPRITE_DETECT_AREAS * 4 * sizeof(uint16_t));
                }
                updateSpriteCanvasImage(index);
                return true;
            }
            cv::Mat* image = activeSpriteImageMutable(index);
            if (!image || image->empty() || previous.image.empty()) {
                return false;
            }
            UndoState current;
            current.image = image->clone();
            stack.redo.push_back(std::move(current));
            *image = previous.image.clone();
            updateSpriteCanvasImage(index);
            return true;
        }
        cv::Mat* image = activeFrameImage(index, true);
        if (!image || image->empty()) {
            return false;
        }
        UndoState current;
        current.image = image->clone();
        stack.redo.push_back(std::move(current));
        UndoState previous = stack.undo.back();
        stack.undo.pop_back();
        *image = previous.image.clone();
        updateFramePreviewAt(index);
        return true;
    };
    if (!isFrame) {
        const int index = m_spritesList->currentRow();
        const bool handled = undoAtIndex(index);
        updateUndoActions();
        return handled;
    }
    const std::vector<int> targets = targetFrameIndices();
    bool handled = false;
    for (int index : targets) {
        handled = undoAtIndex(index) || handled;
    }
    const int current = m_framesList ? m_framesList->currentRow() : -1;
    if (current >= 0) {
        updateFrameCanvasImage(current);
        updateMaskPreviewForFrame(current);
    }
    updateUndoActions();
    return handled;
}

bool MainWindow::redoEdit(bool isFrame)
{
    const auto redoAtIndex = [&](int index) -> bool {
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
        if (!isFrame) {
            UndoState next = stack.redo.back();
            stack.redo.pop_back();
            if (next.mask_kind == MaskKind::Dynamic && !next.mask.empty()) {
                cv::Mat* map = activeSpriteDynamicMask(index);
                if (!map) {
                    return false;
                }
                UndoState current;
                current.mask = map->clone();
                current.mask_kind = MaskKind::Dynamic;
                current.mask_index = index;
                stack.undo.push_back(std::move(current));
                if (stack.undo.size() > m_maxUndoDepth) {
                    stack.undo.erase(stack.undo.begin());
                }
                *map = next.mask.clone();
                updateSpriteCanvasImage(index);
                return true;
            }
            if (next.mask_kind == MaskKind::SpriteDetAreas && !next.mask.empty()) {
                const std::size_t base = static_cast<std::size_t>(index) * MAX_SPRITE_DETECT_AREAS * 4;
                if (base + MAX_SPRITE_DETECT_AREAS * 4 > m_spriteDetAreas.size()) {
                    return false;
                }
                cv::Mat snapshot(1, MAX_SPRITE_DETECT_AREAS * 4, CV_16UC1);
                std::memcpy(snapshot.data,
                            m_spriteDetAreas.data() + base,
                            MAX_SPRITE_DETECT_AREAS * 4 * sizeof(uint16_t));
                UndoState current;
                current.mask = snapshot.clone();
                current.mask_kind = MaskKind::SpriteDetAreas;
                current.mask_index = index;
                stack.undo.push_back(std::move(current));
                if (stack.undo.size() > m_maxUndoDepth) {
                    stack.undo.erase(stack.undo.begin());
                }
                if (next.mask.total() >= static_cast<std::size_t>(MAX_SPRITE_DETECT_AREAS * 4)) {
                    std::memcpy(m_spriteDetAreas.data() + base,
                                next.mask.data,
                                MAX_SPRITE_DETECT_AREAS * 4 * sizeof(uint16_t));
                }
                updateSpriteCanvasImage(index);
                return true;
            }
            cv::Mat* image = activeSpriteImageMutable(index);
            if (!image || image->empty() || next.image.empty()) {
                return false;
            }
            UndoState current;
            current.image = image->clone();
            stack.undo.push_back(std::move(current));
            if (stack.undo.size() > m_maxUndoDepth) {
                stack.undo.erase(stack.undo.begin());
            }
            *image = next.image.clone();
            updateSpriteCanvasImage(index);
            return true;
        }
        cv::Mat* image = activeFrameImage(index, true);
        if (!image || image->empty()) {
            return false;
        }
        UndoState current;
        current.image = image->clone();
        stack.undo.push_back(std::move(current));
        if (stack.undo.size() > m_maxUndoDepth) {
            stack.undo.erase(stack.undo.begin());
        }
        UndoState next = stack.redo.back();
        stack.redo.pop_back();
        *image = next.image.clone();
        updateFramePreviewAt(index);
        return true;
    };
    if (!isFrame) {
        const int index = m_spritesList->currentRow();
        const bool handled = redoAtIndex(index);
        updateUndoActions();
        return handled;
    }
    const std::vector<int> targets = targetFrameIndices();
    bool handled = false;
    for (int index : targets) {
        handled = redoAtIndex(index) || handled;
    }
    const int current = m_framesList ? m_framesList->currentRow() : -1;
    if (current >= 0) {
        updateFrameCanvasImage(current);
        updateMaskPreviewForFrame(current);
    }
    updateUndoActions();
    return handled;
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
    if (stack.undo.size() > m_maxUndoDepth) {
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
    const std::vector<int> targets = targetFrameIndices();
    bool handled = false;
    for (int index : targets) {
        if (index < 0 || index >= static_cast<int>(m_backgroundMaskUndoStacks.size())) {
            continue;
        }
        UndoStack& stack = m_backgroundMaskUndoStacks[static_cast<std::size_t>(index)];
        if (stack.undo.empty()) {
            continue;
        }
        cv::Mat* mask = activeBackgroundMask(index);
        if (!mask || mask->empty()) {
            continue;
        }
        UndoState current;
        current.mask = mask->clone();
        stack.redo.push_back(std::move(current));
        UndoState previous = stack.undo.back();
        stack.undo.pop_back();
        *mask = previous.mask.clone();
        updateFramePreviewAt(index);
        handled = true;
    }
    const int current = m_framesList ? m_framesList->currentRow() : -1;
    if (current >= 0) {
        updateFrameCanvasImage(current);
    }
    updateUndoActions();
    return handled;
}

bool MainWindow::redoBackgroundMaskEdit()
{
    const std::vector<int> targets = targetFrameIndices();
    bool handled = false;
    for (int index : targets) {
        if (index < 0 || index >= static_cast<int>(m_backgroundMaskUndoStacks.size())) {
            continue;
        }
        UndoStack& stack = m_backgroundMaskUndoStacks[static_cast<std::size_t>(index)];
        if (stack.redo.empty()) {
            continue;
        }
        cv::Mat* mask = activeBackgroundMask(index);
        if (!mask || mask->empty()) {
            continue;
        }
        UndoState current;
        current.mask = mask->clone();
        stack.undo.push_back(std::move(current));
        if (stack.undo.size() > m_maxUndoDepth) {
            stack.undo.erase(stack.undo.begin());
        }
        UndoState next = stack.redo.back();
        stack.redo.pop_back();
        *mask = next.mask.clone();
        updateFramePreviewAt(index);
        handled = true;
    }
    const int current = m_framesList ? m_framesList->currentRow() : -1;
    if (current >= 0) {
        updateFrameCanvasImage(current);
    }
    updateUndoActions();
    return handled;
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
    if (mode == MaskMode::Comparison) {
        if (index < 0 || index >= static_cast<int>(stacks->size())) {
            return false;
        }
        UndoStack& stack = (*stacks)[static_cast<std::size_t>(index)];
        if (stack.undo.empty()) {
            return false;
        }
        UndoState current;
        const int maskId = m_maskList ? m_maskList->currentRow() : -1;
        if (maskId >= 0 && maskId < static_cast<int>(m_compMasks.size())) {
            current.mask = m_compMasks[static_cast<std::size_t>(maskId)].clone();
            current.mask_kind = MaskKind::Comparison;
            current.mask_index = maskId;
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
            updatePreviewsForMaskId(previous.mask_index);
        }
        updateMaskPreviewForFrame(index);
        updateUndoActions();
        return true;
    }
    const std::vector<int> targets = targetFrameIndices();
    bool handled = false;
    for (int frameIndex : targets) {
        if (frameIndex < 0 || frameIndex >= static_cast<int>(stacks->size())) {
            continue;
        }
        UndoStack& stack = (*stacks)[static_cast<std::size_t>(frameIndex)];
        if (stack.undo.empty()) {
            continue;
        }
        UndoState current;
        if (frameIndex < static_cast<int>(m_frameDynamicMaskMaps.size())) {
            current.mask = m_frameDynamicMaskMaps[static_cast<std::size_t>(frameIndex)].clone();
            current.mask_kind = MaskKind::Dynamic;
            current.mask_index = frameIndex;
        }
        if (!current.mask.empty()) {
            stack.redo.push_back(std::move(current));
        }
        UndoState previous = stack.undo.back();
        stack.undo.pop_back();
        if (previous.mask_kind == MaskKind::Dynamic &&
            previous.mask_index >= 0 && previous.mask_index < static_cast<int>(m_frameDynamicMaskMaps.size())) {
            m_frameDynamicMaskMaps[static_cast<std::size_t>(previous.mask_index)] = previous.mask.clone();
            handled = true;
        }
    }
    if (handled) {
        updateDynamicMaskPreviewIcons();
        updateMaskPreviewForFrame(index);
        const std::vector<int> targets = targetFrameIndices();
        for (int frameIndex : targets) {
            updateFramePreviewAt(frameIndex);
        }
    }
    updateUndoActions();
    return handled;
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
    if (mode == MaskMode::Comparison) {
        if (index < 0 || index >= static_cast<int>(stacks->size())) {
            return false;
        }
        UndoStack& stack = (*stacks)[static_cast<std::size_t>(index)];
        if (stack.redo.empty()) {
            return false;
        }
        UndoState current;
        const int maskId = m_maskList ? m_maskList->currentRow() : -1;
        if (maskId >= 0 && maskId < static_cast<int>(m_compMasks.size())) {
            current.mask = m_compMasks[static_cast<std::size_t>(maskId)].clone();
            current.mask_kind = MaskKind::Comparison;
            current.mask_index = maskId;
        }
        if (!current.mask.empty()) {
            stack.undo.push_back(std::move(current));
        }
        if (stack.undo.size() > m_maxUndoDepth) {
            stack.undo.erase(stack.undo.begin());
        }
        UndoState next = stack.redo.back();
        stack.redo.pop_back();
        if (next.mask_kind == MaskKind::Comparison &&
            next.mask_index >= 0 && next.mask_index < static_cast<int>(m_compMasks.size())) {
            m_compMasks[static_cast<std::size_t>(next.mask_index)] = next.mask.clone();
            updateMaskPreviewIcons();
            updatePreviewsForMaskId(next.mask_index);
        }
        updateMaskPreviewForFrame(index);
        updateUndoActions();
        return true;
    }
    const std::vector<int> targets = targetFrameIndices();
    bool handled = false;
    for (int frameIndex : targets) {
        if (frameIndex < 0 || frameIndex >= static_cast<int>(stacks->size())) {
            continue;
        }
        UndoStack& stack = (*stacks)[static_cast<std::size_t>(frameIndex)];
        if (stack.redo.empty()) {
            continue;
        }
        UndoState current;
        if (frameIndex < static_cast<int>(m_frameDynamicMaskMaps.size())) {
            current.mask = m_frameDynamicMaskMaps[static_cast<std::size_t>(frameIndex)].clone();
            current.mask_kind = MaskKind::Dynamic;
            current.mask_index = frameIndex;
        }
        if (!current.mask.empty()) {
            stack.undo.push_back(std::move(current));
        }
        if (stack.undo.size() > m_maxUndoDepth) {
            stack.undo.erase(stack.undo.begin());
        }
        UndoState next = stack.redo.back();
        stack.redo.pop_back();
        if (next.mask_kind == MaskKind::Dynamic &&
            next.mask_index >= 0 && next.mask_index < static_cast<int>(m_frameDynamicMaskMaps.size())) {
            m_frameDynamicMaskMaps[static_cast<std::size_t>(next.mask_index)] = next.mask.clone();
            handled = true;
        }
    }
    if (handled) {
        updateDynamicMaskPreviewIcons();
        updateMaskPreviewForFrame(index);
        const std::vector<int> targets = targetFrameIndices();
        for (int frameIndex : targets) {
            updateFramePreviewAt(frameIndex);
        }
    }
    updateUndoActions();
    return handled;
}

void MainWindow::updateUndoActions()
{
    const UndoTarget target = currentUndoTarget();
    int index = -1;
    const std::vector<UndoStack>* stacks = nullptr;
    bool canUndo = false;
    bool canRedo = false;
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
        case UndoTarget::Palette:
            canUndo = !m_paletteUndo.empty();
            canRedo = !m_paletteRedo.empty();
            break;
    }
    if (stacks && index >= 0 && index < static_cast<int>(stacks->size())) {
        if (target == UndoTarget::Frame || target == UndoTarget::DynMask || target == UndoTarget::BackgroundMask) {
            const std::vector<int> targets = targetFrameIndices();
            for (int frameIndex : targets) {
                if (frameIndex < 0 || frameIndex >= static_cast<int>(stacks->size())) {
                    continue;
                }
                const UndoStack& stack = (*stacks)[static_cast<std::size_t>(frameIndex)];
                canUndo = canUndo || !stack.undo.empty();
                canRedo = canRedo || !stack.redo.empty();
            }
        } else {
            const UndoStack& stack = (*stacks)[static_cast<std::size_t>(index)];
            canUndo = !stack.undo.empty();
            canRedo = !stack.redo.empty();
        }
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
    if (m_toolsTabs && m_colorsTab && m_toolsTabs->currentWidget() == m_colorsTab) {
        return UndoTarget::Palette;
    }
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
    refreshRotationEditor();
}

bool MainWindow::hasHdFrame(int index) const
{
    if (index < 0 || index >= static_cast<int>(m_frameExtraFrames.size())) {
        return false;
    }
    if (!m_frameExtraFrames[static_cast<std::size_t>(index)].empty()) {
        return true;
    }
    if (m_serumDataLoaded && index < static_cast<int>(m_serumData.nframes)) {
        const uint8_t* extraFlag = m_serumData.isextraframe[static_cast<uint32_t>(index)];
        return extraFlag && extraFlag[0] != 0;
    }
    return false;
}

cv::Mat* MainWindow::ensureHdFrameLocal(int index)
{
    if (index < 0 || index >= static_cast<int>(m_frameExtraFrames.size())) {
        return nullptr;
    }
    cv::Mat& frame = m_frameExtraFrames[static_cast<std::size_t>(index)];
    if (!frame.empty()) {
        return &frame;
    }
    if (m_serumDataLoaded && index < static_cast<int>(m_serumData.nframes)) {
        const uint8_t* extraFlag = m_serumData.isextraframe[static_cast<uint32_t>(index)];
        if (extraFlag && extraFlag[0] != 0 &&
            m_serumData.cframes_v2_extra.hasData(static_cast<uint32_t>(index))) {
            const uint16_t* data = m_serumData.cframes_v2_extra[static_cast<uint32_t>(index)];
            if (data && m_serumData.fwidth_extra > 0 && m_serumData.fheight_extra > 0) {
                frame = ConvertRgb565ToBgrMat(data,
                                              static_cast<int>(m_serumData.fwidth_extra),
                                              static_cast<int>(m_serumData.fheight_extra));
            }
        }
    }
    return frame.empty() ? nullptr : &frame;
}

cv::Mat* MainWindow::activeFrameImage(int index, bool forEdit)
{
    if (index < 0) {
        return nullptr;
    }
    if (m_useHdFrame && hasHdFrame(index)) {
        return ensureHdFrameLocal(index);
    }
    if (forEdit) {
        cv::Mat* image = m_frameStore->atMutable(index);
        if (image && image->empty()) {
            const cv::Mat base = renderFrameWithSerum(index, m_useHdFrame);
            if (!base.empty()) {
                *image = EnsureBgr(base);
                logLine(QString("Frame edit seed: index=%1 size=%2x%3")
                            .arg(index)
                            .arg(image->cols)
                            .arg(image->rows));
            }
        }
        return image;
    }
    const cv::Mat* image = m_frameStore->at(index);
    return image ? const_cast<cv::Mat*>(image) : nullptr;
}

cv::Mat* MainWindow::activeBackgroundImage(int index, bool forEdit)
{
    if (index < 0) {
        return nullptr;
    }
    if (m_useHdBackground && hasHdBackground(index)) {
        return ensureHdBackgroundLocal(index);
    }
    if (!m_backgroundStore) {
        return nullptr;
    }
    if (forEdit) {
        return m_backgroundStore->atMutable(index);
    }
    const cv::Mat* image = m_backgroundStore->at(index);
    return image ? const_cast<cv::Mat*>(image) : nullptr;
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
    if (m_paletteAssignButton) {
        m_paletteAssignButton->setEnabled(!m_paletteColors.isEmpty());
    }
}

void MainWindow::applyPaletteColorChanges(const std::vector<QColor>& before,
                                          const std::vector<QColor>& after)
{
    if (before.size() != after.size() || before.empty()) {
        return;
    }
    std::vector<uint16_t> mapping(65536);
    for (std::size_t i = 0; i < mapping.size(); ++i) {
        mapping[i] = static_cast<uint16_t>(i);
    }
    bool hasChange = false;
    for (std::size_t i = 0; i < before.size(); ++i) {
        if (before[i] == after[i]) {
            continue;
        }
        const cv::Vec3b oldBgr(static_cast<uint8_t>(before[i].blue()),
                               static_cast<uint8_t>(before[i].green()),
                               static_cast<uint8_t>(before[i].red()));
        const cv::Vec3b newBgr(static_cast<uint8_t>(after[i].blue()),
                               static_cast<uint8_t>(after[i].green()),
                               static_cast<uint8_t>(after[i].red()));
        const uint16_t old565 = BgrToRgb565(oldBgr);
        const uint16_t new565 = BgrToRgb565(newBgr);
        if (old565 != new565) {
            mapping[old565] = new565;
            hasChange = true;
        }
    }
    if (!hasChange) {
        return;
    }
    const std::vector<int> targets = targetFrameIndices();
    if (targets.empty()) {
        return;
    }
    for (int frameIndex : targets) {
        cv::Mat* image = activeFrameImage(frameIndex, true);
        if (!image || image->empty()) {
            continue;
        }
        pushUndoSnapshot(true, frameIndex);
        if (image->type() == CV_8UC3) {
            for (int y = 0; y < image->rows; ++y) {
                cv::Vec3b* row = image->ptr<cv::Vec3b>(y);
                for (int x = 0; x < image->cols; ++x) {
                    const uint16_t old565 = BgrToRgb565(row[x]);
                    const uint16_t new565 = mapping[old565];
                    if (new565 != old565) {
                        row[x] = Rgb565ToBgr(new565);
                    }
                }
            }
        } else if (image->type() == CV_8UC4) {
            for (int y = 0; y < image->rows; ++y) {
                cv::Vec4b* row = image->ptr<cv::Vec4b>(y);
                for (int x = 0; x < image->cols; ++x) {
                    const cv::Vec3b bgr(row[x][0], row[x][1], row[x][2]);
                    const uint16_t old565 = BgrToRgb565(bgr);
                    const uint16_t new565 = mapping[old565];
                    if (new565 != old565) {
                        const cv::Vec3b converted = Rgb565ToBgr(new565);
                        row[x][0] = converted[0];
                        row[x][1] = converted[1];
                        row[x][2] = converted[2];
                    }
                }
            }
        }
        commitFrameFromStore(frameIndex, m_useHdFrame);
        updateFramePreviewAt(frameIndex);
    }
    const int current = m_framesList ? m_framesList->currentRow() : -1;
    if (current >= 0) {
        updateFrameCanvasImage(current);
        updateMaskPreviewForFrame(current);
    }
}

void MainWindow::startPaletteBlink(int slot)
{
    if (!m_framesCanvas || slot < 0 || slot >= m_paletteColors.size()) {
        return;
    }
    if (m_paletteBlinkActive) {
        stopPaletteBlink();
    }
    const int frameIndex = m_framesList ? m_framesList->currentRow() : -1;
    if (frameIndex < 0) {
        return;
    }
    m_paletteBlinkActive = true;
    m_paletteBlinkOn = true;
    m_paletteBlinkSlot = slot;
    m_paletteBlinkFrameIndex = frameIndex;
    m_paletteBlinkUseHd = m_useHdFrame;
    updatePaletteBlinkPreview();
    if (m_paletteBlinkTimer) {
        m_paletteBlinkTimer->start();
    }
}

void MainWindow::stopPaletteBlink()
{
    if (!m_paletteBlinkActive) {
        return;
    }
    m_paletteBlinkActive = false;
    m_paletteBlinkOn = false;
    m_paletteBlinkSlot = -1;
    m_paletteBlinkFrameIndex = -1;
    if (m_paletteBlinkTimer) {
        m_paletteBlinkTimer->stop();
    }
    const int current = m_framesList ? m_framesList->currentRow() : -1;
    if (current >= 0) {
        updateFrameCanvasImage(current);
        updateMaskPreviewForFrame(current);
    }
}

void MainWindow::updatePaletteBlinkPreview()
{
    if (!m_paletteBlinkActive || !m_framesCanvas) {
        return;
    }
    if (!m_paletteBlinkOn) {
        const int current = m_framesList ? m_framesList->currentRow() : -1;
        if (current >= 0) {
            updateFrameCanvasImage(current);
        }
        return;
    }
    if (m_canvasRotateEnabled) {
        return;
    }
    const int frameIndex = m_paletteBlinkFrameIndex;
    if (frameIndex < 0) {
        return;
    }
    const cv::Mat* baseFrame = nullptr;
    if (m_paletteBlinkUseHd && hasHdFrame(frameIndex)) {
        if (frameIndex >= 0 && frameIndex < static_cast<int>(m_frameExtraFrames.size())) {
            baseFrame = &m_frameExtraFrames[static_cast<std::size_t>(frameIndex)];
        }
    }
    if (!baseFrame && m_frameStore) {
        baseFrame = m_frameStore->at(frameIndex);
    }
    if (!baseFrame || baseFrame->empty()) {
        return;
    }
    const QColor target = m_paletteColors[m_paletteBlinkSlot];
    const cv::Vec3b targetBgr(static_cast<uint8_t>(target.blue()),
                              static_cast<uint8_t>(target.green()),
                              static_cast<uint8_t>(target.red()));
    const uint16_t target565 = BgrToRgb565(targetBgr);
    const cv::Vec3b highlight(0, 220, 255);
    cv::Mat preview = baseFrame->clone();
    if (preview.type() == CV_8UC3) {
        for (int y = 0; y < preview.rows; ++y) {
            cv::Vec3b* row = preview.ptr<cv::Vec3b>(y);
            for (int x = 0; x < preview.cols; ++x) {
                if (BgrToRgb565(row[x]) == target565) {
                    row[x] = highlight;
                }
            }
        }
    } else if (preview.type() == CV_8UC4) {
        for (int y = 0; y < preview.rows; ++y) {
            cv::Vec4b* row = preview.ptr<cv::Vec4b>(y);
            for (int x = 0; x < preview.cols; ++x) {
                const cv::Vec3b bgr(row[x][0], row[x][1], row[x][2]);
                if (BgrToRgb565(bgr) == target565) {
                    row[x][0] = highlight[0];
                    row[x][1] = highlight[1];
                    row[x][2] = highlight[2];
                }
            }
        }
    }
    const cv::Mat composed = renderFrameWithSerum(frameIndex, m_paletteBlinkUseHd, preview);
    setFrameCanvasFromComposed(frameIndex, composed);
}

void MainWindow::pushPaletteUndoSnapshot()
{
    if (m_paletteSetIndex < 0 || m_paletteSetIndex >= m_fullPalettes.size()) {
        return;
    }
    markProjectDirty();
    PaletteUndoState state;
    state.kind = PaletteUndoState::Kind::Full;
    state.palette_index = m_paletteSetIndex;
    state.full_colors = m_fullPalettes[m_paletteSetIndex];
    m_paletteUndo.push_back(state);
    if (m_paletteUndo.size() > m_maxUndoDepth) {
        m_paletteUndo.erase(m_paletteUndo.begin());
    }
    m_paletteRedo.clear();
    updateUndoActions();
}

void MainWindow::pushReducedUndoSnapshot(int setIndex)
{
    if (setIndex < 0 || setIndex >= kReducedPaletteCount) {
        return;
    }
    if (m_reducedPaletteIndices.size() < static_cast<std::size_t>(kReducedPaletteCount * 16)) {
        return;
    }
    markProjectDirty();
    PaletteUndoState state;
    state.kind = PaletteUndoState::Kind::Reduced;
    state.set_index = setIndex;
    state.values.resize(16);
    const std::size_t offset = static_cast<std::size_t>(setIndex) * 16;
    std::copy_n(m_reducedPaletteIndices.begin() + offset, 16, state.values.begin());
    m_paletteUndo.push_back(state);
    if (m_paletteUndo.size() > m_maxUndoDepth) {
        m_paletteUndo.erase(m_paletteUndo.begin());
    }
    m_paletteRedo.clear();
    updateUndoActions();
}

void MainWindow::pushDynamicUndoSnapshot(int frameIndex, int setIndex)
{
    if (frameIndex < 0 || frameIndex >= static_cast<int>(m_frameDynamicColors.size()) ||
        setIndex < 0 || setIndex >= MAX_DYNA_SETS_PER_FRAMEN) {
        return;
    }
    const bool useHd = m_useHdFrame && hasHdFrame(frameIndex);
    std::vector<uint16_t>* colorsPtr = ensureFrameDynamicColorsLocal(frameIndex, useHd);
    if (!colorsPtr) {
        return;
    }
    markProjectDirty();
    const std::vector<uint16_t>& colors = *colorsPtr;
    const int stride = dynamicColorsPerSet(colors);
    if (stride <= 0) {
        return;
    }
    const std::size_t offset = static_cast<std::size_t>(setIndex) * stride;
    if (offset + static_cast<std::size_t>(stride) > colors.size()) {
        return;
    }
    PaletteUndoState state;
    state.kind = PaletteUndoState::Kind::Dynamic;
    state.frame_index = frameIndex;
    state.set_index = setIndex;
    state.values.resize(static_cast<std::size_t>(stride));
    std::copy_n(colors.begin() + offset, stride, state.values.begin());
    m_paletteUndo.push_back(state);
    if (m_paletteUndo.size() > m_maxUndoDepth) {
        m_paletteUndo.erase(m_paletteUndo.begin());
    }
    m_paletteRedo.clear();
    updateUndoActions();
}

void MainWindow::pushRotationUndoSnapshot(int frameIndex, int setIndex, bool useHd)
{
    if (setIndex < 0 || setIndex >= MAX_COLOR_ROTATIONN) {
        return;
    }
    if (frameIndex < 0) {
        return;
    }
    markProjectDirty();
    const uint16_t* rotations = rotationBlockForRead(frameIndex, useHd);
    if (!rotations) {
        rotations = nullptr;
    }
    const std::size_t base = rotationSetOffset(setIndex);
    PaletteUndoState state;
    state.kind = PaletteUndoState::Kind::Rotation;
    state.frame_index = frameIndex;
    state.set_index = setIndex;
    state.use_hd = useHd;
    state.values.resize(MAX_LENGTH_COLOR_ROTATION);
    if (rotations && base + MAX_LENGTH_COLOR_ROTATION <= rotationBlockSize()) {
        std::copy_n(rotations + base, MAX_LENGTH_COLOR_ROTATION, state.values.begin());
    } else {
        std::fill(state.values.begin(), state.values.end(), 0);
    }
    m_paletteUndo.push_back(state);
    if (m_paletteUndo.size() > m_maxUndoDepth) {
        m_paletteUndo.erase(m_paletteUndo.begin());
    }
    m_paletteRedo.clear();
    updateUndoActions();
}

bool MainWindow::undoPaletteEdit()
{
    if (m_paletteUndo.empty()) {
        return false;
    }
    PaletteUndoState previous = m_paletteUndo.back();
    m_paletteUndo.pop_back();

    PaletteUndoState current;
    if (previous.kind == PaletteUndoState::Kind::Full) {
        current.kind = PaletteUndoState::Kind::Full;
        current.palette_index = m_paletteSetIndex;
        if (m_paletteSetIndex >= 0 && m_paletteSetIndex < m_fullPalettes.size()) {
            current.full_colors = m_fullPalettes[m_paletteSetIndex];
        }
    } else if (previous.kind == PaletteUndoState::Kind::Reduced) {
        current.kind = PaletteUndoState::Kind::Reduced;
        current.set_index = previous.set_index;
        if (previous.set_index >= 0 && previous.set_index < kReducedPaletteCount) {
            current.values.resize(16);
            const std::size_t offset = static_cast<std::size_t>(previous.set_index) * 16;
            if (offset + 16 <= m_reducedPaletteIndices.size()) {
                std::copy_n(m_reducedPaletteIndices.begin() + offset, 16, current.values.begin());
            }
        }
    } else if (previous.kind == PaletteUndoState::Kind::Dynamic) {
        current.kind = PaletteUndoState::Kind::Dynamic;
        current.frame_index = previous.frame_index;
        current.set_index = previous.set_index;
        if (previous.frame_index >= 0 &&
            previous.frame_index < static_cast<int>(m_frameDynamicColors.size()) &&
            previous.set_index >= 0 && previous.set_index < MAX_DYNA_SETS_PER_FRAMEN) {
            const bool useHd = m_useHdFrame && hasHdFrame(previous.frame_index);
            int stride = 0;
            const uint16_t* colors = frameDynamicColorsData(previous.frame_index, useHd, &stride);
            const std::size_t offset = static_cast<std::size_t>(previous.set_index) * stride;
            const std::size_t size =
                static_cast<std::size_t>(MAX_DYNA_SETS_PER_FRAMEN) * static_cast<std::size_t>(stride);
            if (stride > 0 && colors && offset + static_cast<std::size_t>(stride) <= size) {
                current.values.resize(static_cast<std::size_t>(stride));
                std::copy_n(colors + offset, stride, current.values.begin());
            }
        }
    } else if (previous.kind == PaletteUndoState::Kind::Rotation) {
        current.kind = PaletteUndoState::Kind::Rotation;
        current.frame_index = previous.frame_index;
        current.set_index = previous.set_index;
        current.use_hd = previous.use_hd;
        const uint16_t* rotations = rotationBlockForRead(previous.frame_index, previous.use_hd);
        if (rotations) {
            const std::size_t base = static_cast<std::size_t>(previous.set_index) *
                MAX_LENGTH_COLOR_ROTATION;
            current.values.resize(MAX_LENGTH_COLOR_ROTATION);
            std::copy_n(rotations + base, MAX_LENGTH_COLOR_ROTATION, current.values.begin());
        }
    }
    m_paletteRedo.push_back(current);
    if (m_paletteRedo.size() > m_maxUndoDepth) {
        m_paletteRedo.erase(m_paletteRedo.begin());
    }

    if (previous.kind == PaletteUndoState::Kind::Full) {
        if (previous.palette_index >= 0 && previous.palette_index < m_fullPalettes.size()) {
            const std::vector<QColor> before(m_fullPalettes[previous.palette_index].begin(),
                                             m_fullPalettes[previous.palette_index].end());
            m_paletteSetIndex = previous.palette_index;
            m_fullPalettes[m_paletteSetIndex] = previous.full_colors;
            m_paletteColors = previous.full_colors;
            m_currentPaletteIndex = 0;
            refreshPaletteList();
            applyPaletteColorChanges(before,
                                     std::vector<QColor>(m_fullPalettes[m_paletteSetIndex].begin(),
                                                         m_fullPalettes[m_paletteSetIndex].end()));
        }
    } else if (previous.kind == PaletteUndoState::Kind::Reduced) {
        if (previous.set_index >= 0 && previous.set_index < kReducedPaletteCount &&
            previous.values.size() >= 16) {
            const std::size_t offset = static_cast<std::size_t>(previous.set_index) * 16;
            if (offset + 16 <= m_reducedPaletteIndices.size()) {
                std::copy_n(previous.values.begin(), 16, m_reducedPaletteIndices.begin() + offset);
            }
            refreshReducedPaletteButtons();
        }
    } else if (previous.kind == PaletteUndoState::Kind::Dynamic) {
        if (previous.frame_index >= 0 &&
            previous.frame_index < static_cast<int>(m_frameDynamicColors.size()) &&
            previous.set_index >= 0 && previous.set_index < MAX_DYNA_SETS_PER_FRAMEN &&
            !previous.values.empty()) {
            const bool useHd = m_useHdFrame && hasHdFrame(previous.frame_index);
            std::vector<uint16_t>* colorsPtr = ensureFrameDynamicColorsLocal(previous.frame_index, useHd);
            if (colorsPtr) {
                const int stride = dynamicColorsPerSet(*colorsPtr);
                const std::size_t offset = static_cast<std::size_t>(previous.set_index) * stride;
                if (stride > 0 && offset + previous.values.size() <= colorsPtr->size()) {
                    std::copy_n(previous.values.begin(), previous.values.size(), colorsPtr->begin() + offset);
                }
            }
            refreshDynamicPaletteButtons();
            updateDynamicMaskPreviewIcons();
            updateFramePreviewAt(previous.frame_index);
            if (m_framesList && previous.frame_index == m_framesList->currentRow()) {
                updateFrameCanvasImage(previous.frame_index);
            }
        }
    } else if (previous.kind == PaletteUndoState::Kind::Rotation) {
        std::vector<uint16_t>* rotations = ensureRotationBlockLocal(previous.frame_index, previous.use_hd);
        if (rotations && previous.values.size() >= MAX_LENGTH_COLOR_ROTATION) {
            const std::size_t base = static_cast<std::size_t>(previous.set_index) *
                MAX_LENGTH_COLOR_ROTATION;
            if (base + MAX_LENGTH_COLOR_ROTATION <= rotations->size()) {
                std::copy_n(previous.values.begin(), MAX_LENGTH_COLOR_ROTATION, rotations->begin() + base);
            }
            m_rotationSetIndex = previous.set_index;
            if (m_framesList && previous.frame_index == m_framesList->currentRow()) {
                refreshRotationEditor();
                resetCanvasRotationState();
            }
        }
    }
    updateUndoActions();
    return true;
}

bool MainWindow::redoPaletteEdit()
{
    if (m_paletteRedo.empty()) {
        return false;
    }
    PaletteUndoState next = m_paletteRedo.back();
    m_paletteRedo.pop_back();

    PaletteUndoState current;
    if (next.kind == PaletteUndoState::Kind::Full) {
        current.kind = PaletteUndoState::Kind::Full;
        current.palette_index = m_paletteSetIndex;
        if (m_paletteSetIndex >= 0 && m_paletteSetIndex < m_fullPalettes.size()) {
            current.full_colors = m_fullPalettes[m_paletteSetIndex];
        }
    } else if (next.kind == PaletteUndoState::Kind::Reduced) {
        current.kind = PaletteUndoState::Kind::Reduced;
        current.set_index = next.set_index;
        if (next.set_index >= 0 && next.set_index < kReducedPaletteCount) {
            current.values.resize(16);
            const std::size_t offset = static_cast<std::size_t>(next.set_index) * 16;
            if (offset + 16 <= m_reducedPaletteIndices.size()) {
                std::copy_n(m_reducedPaletteIndices.begin() + offset, 16, current.values.begin());
            }
        }
    } else if (next.kind == PaletteUndoState::Kind::Dynamic) {
        current.kind = PaletteUndoState::Kind::Dynamic;
        current.frame_index = next.frame_index;
        current.set_index = next.set_index;
        if (next.frame_index >= 0 &&
            next.frame_index < static_cast<int>(m_frameDynamicColors.size()) &&
            next.set_index >= 0 && next.set_index < MAX_DYNA_SETS_PER_FRAMEN) {
            const bool useHd = m_useHdFrame && hasHdFrame(next.frame_index);
            int stride = 0;
            const uint16_t* colors = frameDynamicColorsData(next.frame_index, useHd, &stride);
            const std::size_t offset = static_cast<std::size_t>(next.set_index) * stride;
            const std::size_t size =
                static_cast<std::size_t>(MAX_DYNA_SETS_PER_FRAMEN) * static_cast<std::size_t>(stride);
            if (stride > 0 && colors && offset + static_cast<std::size_t>(stride) <= size) {
                current.values.resize(static_cast<std::size_t>(stride));
                std::copy_n(colors + offset, stride, current.values.begin());
            }
        }
    } else if (next.kind == PaletteUndoState::Kind::Rotation) {
        current.kind = PaletteUndoState::Kind::Rotation;
        current.frame_index = next.frame_index;
        current.set_index = next.set_index;
        current.use_hd = next.use_hd;
        const uint16_t* rotations = rotationBlockForRead(next.frame_index, next.use_hd);
        if (rotations) {
            const std::size_t base = static_cast<std::size_t>(next.set_index) *
                MAX_LENGTH_COLOR_ROTATION;
            current.values.resize(MAX_LENGTH_COLOR_ROTATION);
            std::copy_n(rotations + base, MAX_LENGTH_COLOR_ROTATION, current.values.begin());
        }
    }
    m_paletteUndo.push_back(current);
    if (m_paletteUndo.size() > m_maxUndoDepth) {
        m_paletteUndo.erase(m_paletteUndo.begin());
    }

    if (next.kind == PaletteUndoState::Kind::Full) {
        if (next.palette_index >= 0 && next.palette_index < m_fullPalettes.size()) {
            const std::vector<QColor> before(m_fullPalettes[next.palette_index].begin(),
                                             m_fullPalettes[next.palette_index].end());
            m_paletteSetIndex = next.palette_index;
            m_fullPalettes[m_paletteSetIndex] = next.full_colors;
            m_paletteColors = next.full_colors;
            m_currentPaletteIndex = 0;
            refreshPaletteList();
            applyPaletteColorChanges(before,
                                     std::vector<QColor>(m_fullPalettes[m_paletteSetIndex].begin(),
                                                         m_fullPalettes[m_paletteSetIndex].end()));
        }
    } else if (next.kind == PaletteUndoState::Kind::Reduced) {
        if (next.set_index >= 0 && next.set_index < kReducedPaletteCount &&
            next.values.size() >= 16) {
            const std::size_t offset = static_cast<std::size_t>(next.set_index) * 16;
            if (offset + 16 <= m_reducedPaletteIndices.size()) {
                std::copy_n(next.values.begin(), 16, m_reducedPaletteIndices.begin() + offset);
            }
            refreshReducedPaletteButtons();
        }
    } else if (next.kind == PaletteUndoState::Kind::Dynamic) {
        if (next.frame_index >= 0 &&
            next.frame_index < static_cast<int>(m_frameDynamicColors.size()) &&
            next.set_index >= 0 && next.set_index < MAX_DYNA_SETS_PER_FRAMEN &&
            !next.values.empty()) {
            const bool useHd = m_useHdFrame && hasHdFrame(next.frame_index);
            std::vector<uint16_t>* colorsPtr = ensureFrameDynamicColorsLocal(next.frame_index, useHd);
            if (colorsPtr) {
                const int stride = dynamicColorsPerSet(*colorsPtr);
                const std::size_t offset = static_cast<std::size_t>(next.set_index) * stride;
                if (stride > 0 && offset + next.values.size() <= colorsPtr->size()) {
                    std::copy_n(next.values.begin(), next.values.size(), colorsPtr->begin() + offset);
                }
            }
            refreshDynamicPaletteButtons();
            updateDynamicMaskPreviewIcons();
            updateFramePreviewAt(next.frame_index);
            if (m_framesList && next.frame_index == m_framesList->currentRow()) {
                updateFrameCanvasImage(next.frame_index);
            }
        }
    } else if (next.kind == PaletteUndoState::Kind::Rotation) {
        std::vector<uint16_t>* rotations = ensureRotationBlockLocal(next.frame_index, next.use_hd);
        if (rotations && next.values.size() >= MAX_LENGTH_COLOR_ROTATION) {
            const std::size_t base = static_cast<std::size_t>(next.set_index) *
                MAX_LENGTH_COLOR_ROTATION;
            if (base + MAX_LENGTH_COLOR_ROTATION <= rotations->size()) {
                std::copy_n(next.values.begin(), MAX_LENGTH_COLOR_ROTATION, rotations->begin() + base);
            }
            m_rotationSetIndex = next.set_index;
            if (m_framesList && next.frame_index == m_framesList->currentRow()) {
                refreshRotationEditor();
                resetCanvasRotationState();
            }
        }
    }
    updateUndoActions();
    return true;
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

int MainWindow::serumDynamicStride() const
{
    int stride = m_noColors > 0 ? static_cast<int>(m_noColors) : 16;
    if (m_serumDataLoaded && m_serumData.nocolors > 0) {
        stride = static_cast<int>(m_serumData.nocolors);
    }
    stride = std::clamp(stride, 1, 16);
    return stride;
}

const uint16_t* MainWindow::frameDynamicColorsData(int frameIndex, bool useHd, int* strideOut) const
{
    if (strideOut) {
        *strideOut = 0;
    }
    if (frameIndex < 0 || frameIndex >= static_cast<int>(m_frameDynamicColors.size())) {
        return nullptr;
    }
    const std::vector<uint16_t>& local = m_frameDynamicColors[static_cast<std::size_t>(frameIndex)];
    if (!local.empty()) {
        const int stride = dynamicColorsPerSet(local);
        if (strideOut) {
            *strideOut = stride;
        }
        return local.data();
    }
    if (!m_serumDataLoaded || frameIndex >= static_cast<int>(m_serumData.nframes)) {
        return nullptr;
    }
    const uint32_t frameId = static_cast<uint32_t>(frameIndex);
    const uint16_t* data = nullptr;
    if (useHd && m_serumData.dyna4cols_v2_extra.hasData(frameId)) {
        data = m_serumData.dyna4cols_v2_extra[frameId];
    }
    if (!data && m_serumData.dyna4cols_v2.hasData(frameId)) {
        data = m_serumData.dyna4cols_v2[frameId];
    }
    if (!data) {
        return nullptr;
    }
    if (strideOut) {
        *strideOut = serumDynamicStride();
    }
    return data;
}

std::vector<uint16_t>* MainWindow::ensureFrameDynamicColorsLocal(int frameIndex, bool useHd)
{
    if (frameIndex < 0) {
        return nullptr;
    }
    if (m_frameDynamicColors.size() <= static_cast<std::size_t>(frameIndex)) {
        m_frameDynamicColors.resize(static_cast<std::size_t>(frameIndex) + 1);
    }
    std::vector<uint16_t>& colors = m_frameDynamicColors[static_cast<std::size_t>(frameIndex)];
    if (!colors.empty()) {
        return &colors;
    }
    const int stride = serumDynamicStride();
    const std::size_t size = static_cast<std::size_t>(MAX_DYNA_SETS_PER_FRAMEN) *
        static_cast<std::size_t>(stride);
    colors.assign(size, 0);
    if (!m_serumDataLoaded || frameIndex >= static_cast<int>(m_serumData.nframes)) {
        return &colors;
    }
    const uint32_t frameId = static_cast<uint32_t>(frameIndex);
    const uint16_t* data = nullptr;
    if (useHd && m_serumData.dyna4cols_v2_extra.hasData(frameId)) {
        data = m_serumData.dyna4cols_v2_extra[frameId];
    }
    if (!data && m_serumData.dyna4cols_v2.hasData(frameId)) {
        data = m_serumData.dyna4cols_v2[frameId];
    }
    if (data) {
        std::copy_n(data, size, colors.begin());
    }
    return &colors;
}

std::size_t MainWindow::rotationBlockSize() const
{
    return static_cast<std::size_t>(MAX_COLOR_ROTATIONN) * MAX_LENGTH_COLOR_ROTATION;
}

std::size_t MainWindow::rotationSetOffset(int setIndex) const
{
    return static_cast<std::size_t>(setIndex) * MAX_LENGTH_COLOR_ROTATION;
}

const uint16_t* MainWindow::rotationBlockForRead(int frameIndex, bool useHd) const
{
    if (frameIndex < 0) {
        return nullptr;
    }
    const auto& localMap = useHd ? m_frameRotationsLocalX : m_frameRotationsLocal;
    auto it = localMap.find(frameIndex);
    if (it != localMap.end() && !it->second.empty()) {
        return it->second.data();
    }
    if (!m_serumDataLoaded) {
        const std::vector<uint16_t>& rotations = useHd ? m_frameRotationsX : m_frameRotations;
        const std::size_t blockSize = rotationBlockSize();
        const std::size_t base = static_cast<std::size_t>(frameIndex) * blockSize;
        if (base + blockSize <= rotations.size()) {
            return rotations.data() + base;
        }
        return nullptr;
    }
    if (frameIndex >= static_cast<int>(m_serumData.nframes)) {
        return nullptr;
    }
    const uint32_t frameId = static_cast<uint32_t>(frameIndex);
    if (useHd && m_serumData.colorrotations_v2_extra.hasData(frameId)) {
        return m_serumData.colorrotations_v2_extra[frameId];
    }
    if (m_serumData.colorrotations_v2.hasData(frameId)) {
        return m_serumData.colorrotations_v2[frameId];
    }
    return nullptr;
}

std::vector<uint16_t>* MainWindow::ensureRotationBlockLocal(int frameIndex, bool useHd)
{
    if (frameIndex < 0) {
        return nullptr;
    }
    const std::size_t blockSize = rotationBlockSize();
    if (!m_serumDataLoaded) {
        std::vector<uint16_t>& rotations = useHd ? m_frameRotationsX : m_frameRotations;
        const std::size_t base = static_cast<std::size_t>(frameIndex) * blockSize;
        if (base + blockSize > rotations.size()) {
            rotations.resize(base + blockSize, 0);
        }
        return rotations.empty() ? nullptr : &rotations;
    }
    const uint16_t* src = rotationBlockForRead(frameIndex, useHd);
    auto& localMap = useHd ? m_frameRotationsLocalX : m_frameRotationsLocal;
    auto [it, inserted] = localMap.emplace(frameIndex, std::vector<uint16_t>());
    std::vector<uint16_t>& rotations = it->second;
    if (!rotations.empty()) {
        return &rotations;
    }
    rotations.assign(blockSize, 0);
    if (src) {
        std::copy_n(src, blockSize, rotations.begin());
    }
    return &rotations;
}

uint16_t* MainWindow::rotationBlockForEdit(int frameIndex, bool useHd)
{
    std::vector<uint16_t>* block = ensureRotationBlockLocal(frameIndex, useHd);
    return block ? block->data() : nullptr;
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
    const bool useHd = m_useHdFrame && hasHdFrame(frameIndex);
    int stride = 0;
    const uint16_t* colors = frameDynamicColorsData(frameIndex, useHd, &stride);
    if (stride <= 0 || slot >= stride) {
        return QColor(0, 0, 0);
    }
    const std::size_t offset = static_cast<std::size_t>(m_dynamicSetIndex) * stride + slot;
    const std::size_t maxSize =
        static_cast<std::size_t>(MAX_DYNA_SETS_PER_FRAMEN) * static_cast<std::size_t>(stride);
    if (!colors || offset >= maxSize) {
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
    const bool useHd = m_useHdFrame && hasHdFrame(frameIndex);
    std::vector<uint16_t>* colorsPtr = ensureFrameDynamicColorsLocal(frameIndex, useHd);
    if (!colorsPtr) {
        return;
    }
    std::vector<uint16_t>& colors = *colorsPtr;
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
    updateDynamicMaskPreviewIcons();
    updateFramePreviewAt(frameIndex);
    if (m_framesList && frameIndex == m_framesList->currentRow()) {
        updateFrameCanvasImage(frameIndex);
    }
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
        m_reducedAssignButton->setEnabled(m_reducedPaletteList->count() > 0);
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
        m_dynamicAssignButton->setEnabled(hasFrame);
    }
}

void MainWindow::refreshRotationEditor()
{
    if (!m_rotationSetCombo || !m_rotationDelaySpin || !m_rotationList) {
        return;
    }
    if (m_rotationSetIndex < 0 || m_rotationSetIndex >= MAX_COLOR_ROTATIONN) {
        m_rotationSetIndex = 0;
    }
    {
        QSignalBlocker blocker(m_rotationSetCombo);
        m_rotationSetCombo->setCurrentIndex(m_rotationSetIndex);
    }
    const int frameIndex = m_framesList ? m_framesList->currentRow() : -1;
    const bool hasFrame = frameIndex >= 0;
    m_rotationSetCombo->setEnabled(hasFrame);
    m_rotationDelaySpin->setEnabled(hasFrame);
    m_rotationAddButton->setEnabled(hasFrame);
    m_rotationRemoveButton->setEnabled(false);
    m_rotationUpButton->setEnabled(false);
    m_rotationDownButton->setEnabled(false);
    m_rotationClearButton->setEnabled(hasFrame);
    if (m_rotationAssignButton) {
        m_rotationAssignButton->setEnabled(hasFrame);
    }
    refreshRotationList();
}

void MainWindow::refreshRotationList()
{
    if (!m_rotationList || !m_rotationDelaySpin) {
        return;
    }
    const int frameIndex = m_framesList ? m_framesList->currentRow() : -1;
    const bool useHd = frameIndex >= 0 && m_useHdFrame && hasHdFrame(frameIndex);
    const uint16_t* rotations = (frameIndex >= 0) ? rotationBlockForRead(frameIndex, useHd) : nullptr;
    QSignalBlocker delayBlocker(m_rotationDelaySpin);
    QSignalBlocker listBlocker(m_rotationList);
    m_rotationList->clear();
    if (!rotations) {
        m_rotationDelaySpin->setValue(0);
        return;
    }
    const std::size_t base = rotationSetOffset(m_rotationSetIndex);
    if (base + MAX_LENGTH_COLOR_ROTATION > rotationBlockSize()) {
        m_rotationDelaySpin->setValue(0);
        return;
    }
    const uint16_t length = rotations[base];
    const uint16_t delay = rotations[base + 1];
    m_rotationDelaySpin->setValue(delay);
    for (uint16_t i = 0; i < length; ++i) {
        const uint16_t value = rotations[base + 2 + i];
        const cv::Vec3b bgr = Rgb565ToBgr(value);
        QPixmap pixmap(kPaletteSwatchSize, kPaletteSwatchSize);
        pixmap.fill(QColor::fromRgb(bgr[2], bgr[1], bgr[0]));
        auto* item = new QListWidgetItem(QIcon(pixmap), QString());
        item->setData(Qt::UserRole, static_cast<int>(value));
        item->setSizeHint(m_rotationList->gridSize());
        m_rotationList->addItem(item);
    }
    const bool hasSelection = m_rotationList->currentRow() >= 0;
    m_rotationRemoveButton->setEnabled(hasSelection);
    m_rotationUpButton->setEnabled(hasSelection && m_rotationList->currentRow() > 0);
    m_rotationDownButton->setEnabled(hasSelection &&
                                     m_rotationList->currentRow() + 1 < m_rotationList->count());
    if (m_rotationAssignButton) {
        m_rotationAssignButton->setEnabled(m_rotationList->count() > 0);
    }
}

void MainWindow::updateRotationDelay(int delayMs)
{
    const int frameIndex = m_framesList ? m_framesList->currentRow() : -1;
    if (frameIndex < 0) {
        return;
    }
    const bool useHd = m_useHdFrame && hasHdFrame(frameIndex);
    uint16_t* rotations = rotationBlockForEdit(frameIndex, useHd);
    const std::size_t base = rotationSetOffset(m_rotationSetIndex);
    if (!rotations || base + MAX_LENGTH_COLOR_ROTATION > rotationBlockSize()) {
        return;
    }
    const uint16_t clamped = static_cast<uint16_t>(std::clamp(delayMs, 0, 60000));
    if (rotations[base + 1] == clamped) {
        return;
    }
    pushRotationUndoSnapshot(frameIndex, m_rotationSetIndex, useHd);
    rotations[base + 1] = clamped;
    resetCanvasRotationState();
}

void MainWindow::updateRotationDataFromList()
{
    if (!m_rotationList) {
        return;
    }
    const int frameIndex = m_framesList ? m_framesList->currentRow() : -1;
    if (frameIndex < 0) {
        return;
    }
    const bool useHd = m_useHdFrame && hasHdFrame(frameIndex);
    uint16_t* rotations = rotationBlockForEdit(frameIndex, useHd);
    const std::size_t base = rotationSetOffset(m_rotationSetIndex);
    if (!rotations || base + MAX_LENGTH_COLOR_ROTATION > rotationBlockSize()) {
        return;
    }
    const int count = std::min(m_rotationList->count(), MAX_LENGTH_COLOR_ROTATION - 2);
    rotations[base] = static_cast<uint16_t>(count);
    for (int i = 0; i < count; ++i) {
        const QListWidgetItem* item = m_rotationList->item(i);
        const int value = item ? item->data(Qt::UserRole).toInt() : 0;
        rotations[base + 2 + i] = static_cast<uint16_t>(value);
    }
    for (int i = count; i < MAX_LENGTH_COLOR_ROTATION - 2; ++i) {
        rotations[base + 2 + i] = 0;
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

void MainWindow::startPaletteGradient()
{
    if (m_paletteGradientActive) {
        cancelPaletteGradient();
        return;
    }
    if (m_paletteSetSlotActive || m_reducedSetSlotActive || m_dynamicSetSlotActive || m_rotationSetSlotActive) {
        cancelPaletteSetSlot();
    }
    if (!m_paletteList || m_currentPaletteIndex < 0 || m_currentPaletteIndex >= m_paletteColors.size()) {
        return;
    }
    m_paletteGradientActive = true;
    m_paletteGradientStartIndex = m_currentPaletteIndex;
    if (m_paletteGradientButton) {
        m_paletteGradientButton->setText("Pick End");
    }
    statusBar()->showMessage("Select the end palette slot for the gradient (Esc to cancel).", 4000);
}

void MainWindow::startPaletteSetSlot()
{
    if (m_paletteSetSlotActive) {
        cancelPaletteSetSlot();
        return;
    }
    if (!m_paletteList || m_paletteColors.isEmpty()) {
        return;
    }
    m_paletteSelectionIsReference = false;
    m_paletteSetSlotActive = true;
    m_reducedSetSlotActive = false;
    m_dynamicSetSlotActive = false;
    m_rotationSetSlotActive = false;
    if (m_paletteGradientActive) {
        cancelPaletteGradient();
    }
    if (m_paletteList) {
        m_paletteList->setProperty("selectionColor", QColor(255, 140, 0));
        m_paletteList->viewport()->update();
    }
    if (m_paletteGradientButton) {
        m_paletteGradientButton->setEnabled(false);
    }
    if (m_paletteAssignButton) {
        m_paletteAssignButton->setText("Pick Slot");
    }
    statusBar()->showMessage("Select a full palette slot to set (Esc to cancel).", 4000);
}

void MainWindow::startReducedSetSlot()
{
    if (m_reducedSetSlotActive) {
        cancelPaletteSetSlot();
        return;
    }
    if (!m_reducedPaletteList) {
        return;
    }
    m_paletteSelectionIsReference = true;
    m_reducedSetSlotActive = true;
    m_paletteSetSlotActive = false;
    m_dynamicSetSlotActive = false;
    m_rotationSetSlotActive = false;
    if (m_paletteGradientActive) {
        cancelPaletteGradient();
    }
    if (m_paletteGradientButton) {
        m_paletteGradientButton->setEnabled(false);
    }
    if (m_reducedAssignButton) {
        m_reducedAssignButton->setText("Pick Slot");
    }
    if (m_paletteList) {
        m_paletteList->setProperty("selectionColor", QColor(70, 150, 255));
        m_paletteList->viewport()->update();
    }
    statusBar()->showMessage("Select a reduced slot to set (Esc to cancel).", 4000);
}

void MainWindow::startDynamicSetSlot()
{
    if (m_dynamicSetSlotActive) {
        cancelPaletteSetSlot();
        return;
    }
    if (!m_dynamicPaletteList) {
        return;
    }
    m_paletteSelectionIsReference = true;
    m_dynamicSetSlotActive = true;
    m_paletteSetSlotActive = false;
    m_reducedSetSlotActive = false;
    m_rotationSetSlotActive = false;
    if (m_paletteGradientActive) {
        cancelPaletteGradient();
    }
    if (m_paletteGradientButton) {
        m_paletteGradientButton->setEnabled(false);
    }
    if (m_dynamicAssignButton) {
        m_dynamicAssignButton->setText("Pick Slot");
    }
    if (m_paletteList) {
        m_paletteList->setProperty("selectionColor", QColor(70, 150, 255));
        m_paletteList->viewport()->update();
    }
    statusBar()->showMessage("Select a dynamic slot to set (Esc to cancel).", 4000);
}

void MainWindow::startRotationSetSlot()
{
    if (m_rotationSetSlotActive) {
        cancelPaletteSetSlot();
        return;
    }
    if (!m_rotationList || m_rotationList->count() == 0) {
        return;
    }
    m_rotationSetSlotActive = true;
    m_paletteSetSlotActive = false;
    m_reducedSetSlotActive = false;
    m_dynamicSetSlotActive = false;
    if (m_paletteGradientActive) {
        cancelPaletteGradient();
    }
    if (m_rotationAssignButton) {
        m_rotationAssignButton->setText("Pick Slot");
    }
    statusBar()->showMessage("Select a rotation slot to set (Esc to cancel).", 4000);
}

void MainWindow::cancelPaletteSetSlot()
{
    m_paletteSetSlotActive = false;
    m_reducedSetSlotActive = false;
    m_dynamicSetSlotActive = false;
    m_rotationSetSlotActive = false;
    if (m_paletteAssignButton) {
        m_paletteAssignButton->setText("Set Slot");
    }
    if (m_reducedAssignButton) {
        m_reducedAssignButton->setText("Set Slot");
    }
    if (m_dynamicAssignButton) {
        m_dynamicAssignButton->setText("Set Slot");
    }
    if (m_rotationAssignButton) {
        m_rotationAssignButton->setText("Set Slot");
    }
    if (m_paletteGradientButton) {
        const bool hasSelection = m_paletteList && m_paletteList->currentRow() >= 0;
        const bool enable = hasSelection && !m_paletteSelectionIsReference;
        m_paletteGradientButton->setEnabled(enable);
    }
    if (m_paletteList) {
        const QColor color = m_paletteSelectionIsReference ? QColor(70, 150, 255) : QColor(255, 140, 0);
        m_paletteList->setProperty("selectionColor", color);
        m_paletteList->viewport()->update();
    }
    statusBar()->showMessage("Set slot canceled.", 1500);
}

void MainWindow::cancelPaletteGradient()
{
    m_paletteGradientActive = false;
    m_paletteGradientStartIndex = -1;
    if (m_paletteGradientButton) {
        m_paletteGradientButton->setText("Gradient");
    }
    statusBar()->showMessage("Gradient canceled.", 2000);
}

void MainWindow::applyPaletteGradient(int startIndex, int endIndex)
{
    if (startIndex < 0 || endIndex < 0 || startIndex >= m_paletteColors.size() ||
        endIndex >= m_paletteColors.size() || startIndex == endIndex) {
        return;
    }
    const int stepCount = std::abs(endIndex - startIndex);
    const QColor startColor = m_paletteColors[startIndex];
    const QColor endColor = m_paletteColors[endIndex];
    for (int step = 0; step <= stepCount; ++step) {
        const double t = stepCount == 0 ? 0.0 : static_cast<double>(step) / stepCount;
        const int r = static_cast<int>(std::round(startColor.red() + (endColor.red() - startColor.red()) * t));
        const int g = static_cast<int>(std::round(startColor.green() + (endColor.green() - startColor.green()) * t));
        const int b = static_cast<int>(std::round(startColor.blue() + (endColor.blue() - startColor.blue()) * t));
        const cv::Vec3b quant = Rgb565ToBgr(BgrToRgb565(cv::Vec3b(
            static_cast<uint8_t>(b),
            static_cast<uint8_t>(g),
            static_cast<uint8_t>(r))));
        const int index = startIndex < endIndex ? (startIndex + step) : (startIndex - step);
        m_paletteColors[index] = QColor(quant[2], quant[1], quant[0]);
    }
    if (m_paletteSetIndex >= 0 && m_paletteSetIndex < m_fullPalettes.size()) {
        m_fullPalettes[m_paletteSetIndex] = m_paletteColors;
    }
    refreshPaletteList();
}

void MainWindow::keyPressEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_Escape) {
        if (m_framePreviewList) {
            const QPoint globalPos = QCursor::pos();
            const QPoint viewportPos = m_framePreviewList->viewport()->mapFromGlobal(globalPos);
            const bool overPreview = m_framePreviewList->viewport()->rect().contains(viewportPos);
            const bool previewFocused = m_framePreviewList->hasFocus() ||
                m_framePreviewList->viewport()->hasFocus();
            if (overPreview || previewFocused) {
                clearPreviewSelection();
                event->accept();
                return;
            }
        }
        if (m_paletteGradientActive) {
            cancelPaletteGradient();
            event->accept();
            return;
        }
    }
    QMainWindow::keyPressEvent(event);
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
        const cv::Mat image = m_backgroundStore->loadCopy(i);
        if (image.empty()) {
            continue;
        }
        cv::Mat rgb;
        const bool hasHd = hasHdBackground(i);
        cv::Mat hd;
        if (hasHd) {
            if (cv::Mat* local = ensureHdBackgroundLocal(i)) {
                hd = *local;
            }
        }
        const QColor gap = m_backgroundList->palette().color(QPalette::Window);
        cv::Mat previewMat = BuildBackgroundPreview(image,
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
        item->setData(kPreviewIconSizeRole, pixmap.size());
        {
            const int padding = 6;
            const int textHeight = m_backgroundList->fontMetrics().height() + 4;
            const int gap = 2;
            const int width = pixmap.width() + padding * 2;
            const int height = pixmap.height() + textHeight + padding * 2 + gap;
            item->setSizeHint(QSize(width, height));
        }
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
    updateFrameUsageHighlights(m_framesList ? m_framesList->currentRow() : -1);
}

std::vector<MainWindow::SpriteZoneGroup> MainWindow::buildSpriteZonesForFrame(int frameIndex) const
{
    std::vector<SpriteZoneGroup> zones;
    if (frameIndex < 0 || frameIndex >= static_cast<int>(m_frameSpriteAssignments.size() / MAX_SPRITES_PER_FRAME)) {
        return zones;
    }
    const cv::Mat* frame = m_frameStore ? m_frameStore->at(frameIndex) : nullptr;
    const int frameWidth = frame ? frame->cols : kDefaultFrameWidth;
    const int frameHeight = frame ? frame->rows : kDefaultFrameHeight;
    const std::size_t baseSlot = static_cast<std::size_t>(frameIndex) * MAX_SPRITES_PER_FRAME;
    const std::size_t bboxBase = static_cast<std::size_t>(frameIndex) * MAX_SPRITES_PER_FRAME * 4;
    std::map<std::tuple<int, int, int, int>, std::size_t> zoneMap;
    for (int slot = 0; slot < MAX_SPRITES_PER_FRAME; ++slot) {
        const std::size_t slotIndex = baseSlot + static_cast<std::size_t>(slot);
        if (slotIndex >= m_frameSpriteAssignments.size()) {
            break;
        }
        const uint8_t spriteIndex = m_frameSpriteAssignments[slotIndex];
        int minx = 0;
        int miny = 0;
        int maxx = frameWidth > 0 ? frameWidth - 1 : 0;
        int maxy = frameHeight > 0 ? frameHeight - 1 : 0;
        const std::size_t bboxIndex = bboxBase + static_cast<std::size_t>(slot) * 4;
        if (bboxIndex + 3 < m_frameSpriteBBoxes.size()) {
            minx = static_cast<int>(m_frameSpriteBBoxes[bboxIndex]);
            miny = static_cast<int>(m_frameSpriteBBoxes[bboxIndex + 1]);
            maxx = static_cast<int>(m_frameSpriteBBoxes[bboxIndex + 2]);
            maxy = static_cast<int>(m_frameSpriteBBoxes[bboxIndex + 3]);
        }
        const bool hasBBox = !(minx == 0 && miny == 0 && maxx == 0 && maxy == 0);
        const bool hasSprite = (spriteIndex != 255);
        const bool zoneFlagged = slotIndex < m_frameSpriteZoneFlags.size() &&
            m_frameSpriteZoneFlags[slotIndex] != 0;
        if (!hasSprite && (!hasBBox || !zoneFlagged)) {
            continue;
        }
        if (maxx < minx || maxy < miny) {
            continue;
        }
        minx = std::clamp(minx, 0, frameWidth > 0 ? frameWidth - 1 : 0);
        miny = std::clamp(miny, 0, frameHeight > 0 ? frameHeight - 1 : 0);
        maxx = std::clamp(maxx, minx, frameWidth > 0 ? frameWidth - 1 : minx);
        maxy = std::clamp(maxy, miny, frameHeight > 0 ? frameHeight - 1 : miny);
        const QRect rect(minx, miny, maxx - minx + 1, maxy - miny + 1);
        const auto key = std::make_tuple(rect.x(), rect.y(), rect.width(), rect.height());
        auto it = zoneMap.find(key);
        if (it == zoneMap.end()) {
            SpriteZoneGroup zone;
            zone.rect = rect;
            zone.slotIndices.push_back(slot);
            if (hasSprite) {
                zone.sprites.push_back(static_cast<int>(spriteIndex));
            }
            zones.push_back(std::move(zone));
            zoneMap[key] = zones.size() - 1;
        } else {
            SpriteZoneGroup& zone = zones[it->second];
            zone.slotIndices.push_back(slot);
            if (hasSprite) {
                zone.sprites.push_back(static_cast<int>(spriteIndex));
            }
        }
    }
    return zones;
}

void MainWindow::refreshSpriteZoneList()
{
    if (!m_spriteZoneList) {
        return;
    }
    const int frameIndex = m_framesList ? m_framesList->currentRow() : -1;
    QSignalBlocker blocker(m_spriteZoneList);
    m_spriteZoneList->clear();
    if (frameIndex < 0 || !m_frameStore || m_frameStore->count() <= 0) {
        auto* item = new QListWidgetItem("No frames");
        item->setFlags(item->flags() & ~Qt::ItemIsEnabled);
        m_spriteZoneList->addItem(item);
        m_spriteZoneList->setEnabled(false);
        m_selectedSpriteZoneIndex = -1;
        if (m_spriteZoneAddButton) {
            m_spriteZoneAddButton->setEnabled(false);
        }
        if (m_spriteZoneRemoveButton) {
            m_spriteZoneRemoveButton->setEnabled(false);
        }
        refreshSpriteZoneSpritesList();
        return;
    }
    m_spriteZones = buildSpriteZonesForFrame(frameIndex);
    if (m_spriteZones.empty()) {
        auto* item = new QListWidgetItem("No sprite zones");
        item->setFlags(item->flags() & ~Qt::ItemIsEnabled);
        m_spriteZoneList->addItem(item);
        m_spriteZoneList->setEnabled(false);
        m_selectedSpriteZoneIndex = -1;
        if (m_spriteZoneAddButton) {
            m_spriteZoneAddButton->setEnabled(true);
        }
        if (m_spriteZoneRemoveButton) {
            m_spriteZoneRemoveButton->setEnabled(false);
        }
        refreshSpriteZoneSpritesList();
        return;
    }
    m_spriteZoneList->setEnabled(true);
    int preferredZone = -1;
    if (m_selectedSpriteSlot >= 0 && m_selectedSpriteSlot < MAX_SPRITES_PER_FRAME) {
        for (std::size_t i = 0; i < m_spriteZones.size(); ++i) {
            const SpriteZoneGroup& zone = m_spriteZones[i];
            if (std::find(zone.slotIndices.begin(), zone.slotIndices.end(), m_selectedSpriteSlot) != zone.slotIndices.end()) {
                preferredZone = static_cast<int>(i);
                break;
            }
        }
    }
    const QSize iconSize(kPreviewIconWidth, kPreviewIconHeight);
    const cv::Mat reference = buildOriginalPreviewForIndex(frameIndex);
    cv::Mat original = buildOriginalFrame(reference);
    if (original.empty()) {
        original = MakePlaceholderImage(kDefaultFrameWidth, kDefaultFrameHeight,
                                        cv::Scalar(18, 18, 18),
                                        cv::Scalar(55, 55, 55));
    }
    cv::Mat previewBase = EnsureBgr(original);
    const double scale = std::min(static_cast<double>(iconSize.width()) / previewBase.cols,
                                  static_cast<double>(iconSize.height()) / previewBase.rows);
    const int targetW = std::max(1, static_cast<int>(previewBase.cols * scale));
    const int targetH = std::max(1, static_cast<int>(previewBase.rows * scale));
    cv::Mat resized;
    cv::resize(previewBase, resized, cv::Size(targetW, targetH), 0.0, 0.0, cv::INTER_NEAREST);
    const QColor baseColor = m_spriteZoneList
        ? m_spriteZoneList->palette().color(QPalette::Window)
        : QApplication::palette().color(QPalette::Window);
    cv::Mat fitted(iconSize.height(), iconSize.width(), CV_8UC3,
                   cv::Scalar(baseColor.blue(), baseColor.green(), baseColor.red()));
    const int offsetX = (iconSize.width() - targetW) / 2;
    const int offsetY = (iconSize.height() - targetH) / 2;
    resized.copyTo(fitted(cv::Rect(offsetX, offsetY, targetW, targetH)));
    cv::Mat previewRgb;
    cv::cvtColor(fitted, previewRgb, cv::COLOR_BGR2RGB);
    QImage baseImage(previewRgb.data, previewRgb.cols, previewRgb.rows, previewRgb.step, QImage::Format_RGB888);
    for (std::size_t i = 0; i < m_spriteZones.size(); ++i) {
        const SpriteZoneGroup& zone = m_spriteZones[i];
        QPixmap pixmap = QPixmap::fromImage(baseImage.copy());
        QPainter painter(&pixmap);
        painter.setRenderHint(QPainter::Antialiasing, false);
        const double scaleX = previewBase.cols > 0 ? static_cast<double>(targetW) / previewBase.cols : 1.0;
        const double scaleY = previewBase.rows > 0 ? static_cast<double>(targetH) / previewBase.rows : 1.0;
        const QRectF rect(offsetX + zone.rect.x() * scaleX,
                          offsetY + zone.rect.y() * scaleY,
                          zone.rect.width() * scaleX,
                          zone.rect.height() * scaleY);
        QPen pen(QColor(255, 200, 0));
        pen.setWidth(2);
        painter.setPen(pen);
        painter.drawRect(rect);
        if (m_spriteStore && !zone.sprites.empty()) {
            const int thumbSize = 28;
            const int padding = 2;
            int thumbX = pixmap.width() - padding - thumbSize;
            int thumbY = pixmap.height() - padding - thumbSize;
            for (int spriteIndex : zone.sprites) {
                if (thumbX < padding) {
                    break;
                }
                const cv::Mat sprite = (spriteIndex >= 0 && spriteIndex < m_spriteStore->count())
                    ? m_spriteStore->loadCopy(spriteIndex)
                    : cv::Mat();
                if (sprite.empty()) {
                    continue;
                }
                cv::Mat spriteRgb;
                cv::Mat spriteBgr = EnsureBgr(sprite);
                cv::cvtColor(spriteBgr, spriteRgb, cv::COLOR_BGR2RGB);
                QImage spriteImage(spriteRgb.data,
                                   spriteRgb.cols,
                                   spriteRgb.rows,
                                   spriteRgb.step,
                                   QImage::Format_RGB888);
                QPixmap spritePixmap = QPixmap::fromImage(spriteImage.copy());
                spritePixmap = spritePixmap.scaled(thumbSize,
                                                   thumbSize,
                                                   Qt::KeepAspectRatio,
                                                   Qt::FastTransformation);
                painter.drawPixmap(QRect(thumbX, thumbY, thumbSize, thumbSize), spritePixmap);
                thumbX -= thumbSize + padding;
            }
        }
        painter.end();

        QString label = QString("Zone %1").arg(static_cast<int>(i) + 1);
        auto* item = new QListWidgetItem();
        item->setIcon(QIcon(pixmap));
        item->setText(label);
        item->setData(kSpriteZoneIndexRole, static_cast<int>(i));
        item->setData(kPreviewIconSizeRole, pixmap.size());
        const int padding = 6;
        const int textHeight = m_spriteZoneList->fontMetrics().height() + 4;
        const int gap = 2;
        const int width = pixmap.width() + padding * 2;
        const int height = pixmap.height() + textHeight + padding * 2 + gap;
        item->setSizeHint(QSize(width, height));
        m_spriteZoneList->addItem(item);
    }
    if (m_selectedSpriteZoneIndex >= 0 && m_selectedSpriteZoneIndex < m_spriteZoneList->count()) {
        m_spriteZoneList->setCurrentRow(m_selectedSpriteZoneIndex);
    } else if (preferredZone >= 0 && preferredZone < m_spriteZoneList->count()) {
        m_spriteZoneList->setCurrentRow(preferredZone);
        m_selectedSpriteZoneIndex = preferredZone;
    } else if (m_spriteZoneList->count() > 0) {
        m_spriteZoneList->setCurrentRow(0);
        m_selectedSpriteZoneIndex = 0;
    }
    if (m_spriteZoneAddButton) {
        m_spriteZoneAddButton->setEnabled(frameIndex >= 0);
    }
    if (m_spriteZoneRemoveButton) {
        m_spriteZoneRemoveButton->setEnabled(m_selectedSpriteZoneIndex >= 0);
    }
    refreshSpriteZoneSpritesList();
}

void MainWindow::refreshSpriteZoneSpritesList()
{
    if (!m_spriteZoneSpritesList) {
        return;
    }
    const int frameIndex = m_framesList ? m_framesList->currentRow() : -1;
    int restoreSlot = m_spriteZoneSpritesList->currentItem()
        ? m_spriteZoneSpritesList->currentItem()->data(kSpriteZoneSlotRole).toInt()
        : -1;
    if (m_spriteZonePreferredSlot >= 0) {
        restoreSlot = m_spriteZonePreferredSlot;
        m_spriteZonePreferredSlot = -1;
    }
    QSignalBlocker blocker(m_spriteZoneSpritesList);
    m_spriteZoneSpritesList->clear();
        m_spriteZoneSpritesList->setEnabled(false);
    if (m_spriteZoneSpriteUp) {
        m_spriteZoneSpriteUp->setEnabled(false);
    }
    if (m_spriteZoneSpriteDown) {
        m_spriteZoneSpriteDown->setEnabled(false);
    }
    if (m_spriteZoneSpriteRemove) {
        m_spriteZoneSpriteRemove->setEnabled(false);
    }
    if (frameIndex < 0 ||
        m_selectedSpriteZoneIndex < 0 ||
        m_selectedSpriteZoneIndex >= static_cast<int>(m_spriteZones.size())) {
        auto* item = new QListWidgetItem("No sprite zone selected");
        item->setFlags(item->flags() & ~Qt::ItemIsEnabled);
        m_spriteZoneSpritesList->addItem(item);
        return;
    }
    const SpriteZoneGroup& zone = m_spriteZones[static_cast<std::size_t>(m_selectedSpriteZoneIndex)];
    if (zone.slotIndices.empty()) {
        auto* item = new QListWidgetItem("No sprites assigned");
        item->setFlags(item->flags() & ~Qt::ItemIsEnabled);
        m_spriteZoneSpritesList->addItem(item);
        return;
    }
    m_spriteZoneSpritesList->setEnabled(true);
    const QColor gap = m_spriteZoneSpritesList->palette().color(QPalette::Window);
    const QStringList spriteNames = m_state->sprites();
    bool added = false;
    for (int slot : zone.slotIndices) {
        const std::size_t slotIndex = static_cast<std::size_t>(frameIndex) * MAX_SPRITES_PER_FRAME +
            static_cast<std::size_t>(slot);
        if (slotIndex >= m_frameSpriteAssignments.size()) {
            continue;
        }
        const int spriteIndex = static_cast<int>(m_frameSpriteAssignments[slotIndex]);
        if (spriteIndex < 0 || spriteIndex == 255) {
            continue;
        }
        const cv::Mat sprite = (spriteIndex >= 0 && m_spriteStore &&
                                 spriteIndex < m_spriteStore->count())
            ? m_spriteStore->loadCopy(spriteIndex)
            : cv::Mat();
        if (sprite.empty()) {
            continue;
        }
        cv::Mat hd;
        if (spriteIndex >= 0) {
            if (const cv::Mat* hdSprite = const_cast<MainWindow*>(this)->ensureHdSpriteLocal(spriteIndex)) {
                hd = *hdSprite;
            }
        }
        cv::Mat previewMat = BuildBackgroundPreview(sprite,
                                                    hd,
                                                    cv::Scalar(gap.blue(), gap.green(), gap.red()));
        if (previewMat.empty()) {
            continue;
        }
        cv::Mat rgb;
        cv::cvtColor(previewMat, rgb, cv::COLOR_BGR2RGB);
        QImage previewImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
        const bool hasHd = !hd.empty();
        const QSize iconSize = hasHd
            ? QSize(kPreviewIconWidthHd, kPreviewIconHeightHd)
            : QSize(kPreviewIconWidth, kPreviewIconHeight);
        QPixmap pixmap = QPixmap::fromImage(previewImage.copy());
        pixmap = pixmap.scaled(iconSize, Qt::KeepAspectRatio, Qt::FastTransformation);

        auto* item = new QListWidgetItem();
        item->setIcon(QIcon(pixmap));
        const QString spriteLabel = (spriteIndex >= 0 && spriteIndex < spriteNames.size() &&
                                     !spriteNames[spriteIndex].isEmpty())
            ? spriteNames[spriteIndex]
            : QString("Sprite %1").arg(spriteIndex + 1);
        item->setText(QString("Slot %1 - %2").arg(slot + 1).arg(spriteLabel));
        item->setData(kPreviewIconSizeRole, pixmap.size());
        item->setData(kSpriteZoneSlotRole, slot);
        item->setData(kSpriteZoneSpriteIndexRole, spriteIndex);
        {
            const int padding = 6;
            const int textHeight = m_spriteZoneSpritesList->fontMetrics().height() + 4;
            const int gap = 2;
            const int width = pixmap.width() + padding * 2;
            const int height = pixmap.height() + textHeight + padding * 2 + gap;
            item->setSizeHint(QSize(width, height));
        }
        m_spriteZoneSpritesList->addItem(item);
        added = true;
    }
    if (!added) {
        auto* item = new QListWidgetItem("No sprites assigned");
        item->setFlags(item->flags() & ~Qt::ItemIsEnabled);
        m_spriteZoneSpritesList->addItem(item);
        m_spriteZoneSpritesList->setEnabled(false);
        return;
    }
    if (restoreSlot >= 0) {
        for (int i = 0; i < m_spriteZoneSpritesList->count(); ++i) {
            QListWidgetItem* item = m_spriteZoneSpritesList->item(i);
            if (item && item->data(kSpriteZoneSlotRole).toInt() == restoreSlot) {
                m_spriteZoneSpritesList->setCurrentRow(i);
                break;
            }
        }
    }
    if (m_spriteZoneSpritesList->currentRow() < 0 && m_spriteZoneSpritesList->count() > 0) {
        m_spriteZoneSpritesList->setCurrentRow(0);
    }
    if (m_spriteZoneSpriteUp) {
        m_spriteZoneSpriteUp->setEnabled(m_spriteZoneSpritesList->count() > 0);
    }
    if (m_spriteZoneSpriteDown) {
        m_spriteZoneSpriteDown->setEnabled(m_spriteZoneSpritesList->count() > 0);
    }
    if (m_spriteZoneSpriteRemove) {
        m_spriteZoneSpriteRemove->setEnabled(m_spriteZoneSpritesList->count() > 0);
    }
}

void MainWindow::refreshFrameSpriteSlotCombo()
{
    if (m_selectedSpriteSlot < 0) {
        m_selectedSpriteSlot = 0;
    }
}

void MainWindow::refreshFrameSpriteLists()
{
    if (m_isLoadingProject) {
        return;
    }
    const int frameSelection = m_framesList ? m_framesList->currentRow() : -1;
    const int spriteSelection = m_spritesList ? m_spritesList->currentRow() : -1;
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
        if (m_serumDataLoaded) {
            m_frameStore->setCount(m_state->frames().size());
            m_serumData.nframes = static_cast<uint32_t>(m_frameStore->count());
        } else {
            while (m_frameStore->count() < m_state->frames().size()) {
                m_frameStore->add(MakePlaceholderImage(kDefaultFrameWidth, kDefaultFrameHeight,
                                                       cv::Scalar(18, 18, 18),
                                                       cv::Scalar(55, 55, 55)));
            }
            while (m_frameStore->count() > m_state->frames().size()) {
                m_frameStore->removeAt(m_frameStore->count() - 1);
            }
        }
        const QColor gap = m_framesList->palette().color(QPalette::Window);
        const QStringList frameNames = m_state->frames();
        std::unordered_map<int, QString> bookmarkLabels;
        if (!m_sectionStarts.empty() && !m_sectionNames.empty()) {
            const std::size_t count = std::min(m_sectionStarts.size(), m_sectionNames.size());
            for (std::size_t i = 0; i < count; ++i) {
                if (m_sectionNames[i].empty()) {
                    continue;
                }
                bookmarkLabels[static_cast<int>(m_sectionStarts[i])] =
                    QString::fromStdString(m_sectionNames[i]);
            }
        }
        for (int i = 0; i < frameNames.size(); ++i) {
            if (!m_serumDataLoaded) {
                const cv::Mat* image = m_frameStore->at(i);
                if (!image || image->empty()) {
                    continue;
                }
            }
            cv::Mat composed = renderFrameWithSerum(i, false);
            cv::Mat hdComposed;
            if (hasHdFrame(i) && i < static_cast<int>(m_frameExtraFrames.size())) {
                const cv::Mat& hdFrame = m_frameExtraFrames[static_cast<std::size_t>(i)];
                if (!hdFrame.empty()) {
                    hdComposed = renderFrameWithSerum(i, true);
                }
            }
            cv::Mat previewMat = BuildBackgroundPreview(composed,
                                                        hdComposed,
                                                        cv::Scalar(gap.blue(), gap.green(), gap.red()));
            if (previewMat.empty()) {
                continue;
            }
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
            QString label = frameNames[i];
            const auto bookmarkIt = bookmarkLabels.find(i);
            if (bookmarkIt != bookmarkLabels.end()) {
                label = QString("%1 - %2").arg(label, bookmarkIt->second);
            }
            item->setText(label);
            item->setData(kPreviewIconSizeRole, pixmap.size());
            {
                const int padding = 6;
                const int textHeight = m_framesList->fontMetrics().height() + 4;
                const int gap = 2;
                const int width = pixmap.width() + padding * 2;
                const int height = pixmap.height() + textHeight + padding * 2 + gap;
                item->setSizeHint(QSize(width, height));
            }
            item->setData(Qt::UserRole, i);
            item->setData(Qt::UserRole + 1, QStringLiteral("frame"));
            m_framesList->addItem(item);
        }
        if (frameSelection >= 0 && frameSelection < m_framesList->count()) {
            m_framesList->setCurrentRow(frameSelection);
        } else if (m_framesList->currentRow() < 0 && m_framesList->count() > 0) {
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
    if (!m_serumDataLoaded) {
        const std::size_t rotationBlockSize =
            static_cast<std::size_t>(MAX_COLOR_ROTATIONN) * MAX_LENGTH_COLOR_ROTATION;
        const std::size_t rotationSize = static_cast<std::size_t>(frameCount) * rotationBlockSize;
        if (m_frameRotations.size() != rotationSize) {
            m_frameRotations.resize(rotationSize, 0);
        }
        if (m_frameRotationsX.size() != rotationSize) {
            m_frameRotationsX.resize(rotationSize, 0);
        }
    }
    }

    if (m_state->sprites().isEmpty()) {
        auto* item = new QListWidgetItem("No sprites loaded");
        item->setFlags(item->flags() & ~Qt::ItemIsEnabled);
        m_spritesList->addItem(item);
        m_spritesCanvas->setTitle("Sprites canvas (placeholder)");
        m_spritesCanvas->setStatusText("No sprites loaded");
        m_spritesCanvas->setImage(cv::Mat());
        m_spriteStore->clear();
        updateMetadataForSprite(-1);
        m_spriteDetAreaMode = false;
        if (m_spritesCanvas) {
            m_spritesCanvas->setMaskButtonsChecked(false, false);
        }
        if (m_spriteDetAreaCombo) {
            QSignalBlocker blocker(m_spriteDetAreaCombo);
            m_spriteDetAreaCombo->setCurrentIndex(0);
            m_spriteDetAreaCombo->setEnabled(false);
        }
        if (m_spriteDetAreaClearButton) {
            m_spriteDetAreaClearButton->setEnabled(false);
        }
    } else {
        while (m_spriteStore->count() < m_state->sprites().size()) {
            m_spriteStore->add(MakePlaceholderImage(kDefaultSpriteWidth, kDefaultSpriteHeight,
                                                    cv::Scalar(24, 24, 24),
                                                    cv::Scalar(70, 70, 70)));
        }
        while (m_spriteStore->count() > m_state->sprites().size()) {
            m_spriteStore->removeAt(m_spriteStore->count() - 1);
        }
        const QColor gap = m_spritesList->palette().color(QPalette::Window);
        const QStringList spriteNames = m_state->sprites();
        for (int i = 0; i < spriteNames.size(); ++i) {
            const cv::Mat image = m_spriteStore->loadCopy(i);
            if (image.empty()) {
                continue;
            }
            cv::Mat hd;
            if (i >= 0) {
                if (const cv::Mat* hdSprite = ensureHdSpriteLocal(i)) {
                    hd = *hdSprite;
                }
            }
            cv::Mat previewMat = BuildBackgroundPreview(image,
                                                        hd,
                                                        cv::Scalar(gap.blue(), gap.green(), gap.red()));
            if (previewMat.empty()) {
                continue;
            }
            cv::Mat rgb;
            cv::cvtColor(previewMat, rgb, cv::COLOR_BGR2RGB);
            QImage previewImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
            const bool hasHd = !hd.empty();
            const QSize iconSize = hasHd
                ? QSize(kPreviewIconWidthHd, kPreviewIconHeightHd)
                : QSize(kPreviewIconWidth, kPreviewIconHeight);
            QPixmap pixmap = QPixmap::fromImage(previewImage.copy());
            pixmap = pixmap.scaled(iconSize, Qt::KeepAspectRatio, Qt::FastTransformation);
            auto* item = new QListWidgetItem();
            item->setIcon(QIcon(pixmap));
            const QString label = spriteNames[i].isEmpty()
                ? QString("Sprite %1").arg(i + 1)
                : spriteNames[i];
            item->setText(label);
            item->setData(kPreviewIconSizeRole, pixmap.size());
            {
                const int padding = 6;
                const int textHeight = m_spritesList->fontMetrics().height() + 4;
                const int gap = 2;
                const int width = pixmap.width() + padding * 2;
                const int height = pixmap.height() + textHeight + padding * 2 + gap;
                item->setSizeHint(QSize(width, height));
            }
            item->setData(Qt::UserRole, i);
            item->setData(Qt::UserRole + 1, QStringLiteral("sprite"));
            m_spritesList->addItem(item);
        }
        if (spriteSelection >= 0 && spriteSelection < m_spritesList->count()) {
            m_spritesList->setCurrentRow(spriteSelection);
        } else if (m_spritesList->currentRow() < 0 && m_spritesList->count() > 0) {
            m_spritesList->setCurrentRow(0);
        } else if (m_spritesList->currentRow() >= 0) {
            showSpriteAtIndex(m_spritesList->currentRow());
        }
    }

    updateFrameJumpRange();
    refreshFramePreviews();
    ensureUndoStacksSize();
    ensureSpriteDataSize();
    ensureMaskDataSize();
    refreshFrameSpriteSlotCombo();
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

bool MainWindow::eventFilter(QObject* obj, QEvent* event)
{
    if (!m_uiReady) {
        return QMainWindow::eventFilter(obj, event);
    }
    if (m_projectLabel && obj == m_projectLabel && event->type() == QEvent::Resize) {
        updateProjectLabelHeight();
    }
    if (m_paletteList &&
        (obj == m_paletteList || obj == m_paletteList->viewport())) {
        if (event->type() == QEvent::MouseButtonPress) {
            auto* mouseEvent = static_cast<QMouseEvent*>(event);
            QPoint pos = mouseEvent->pos();
            if (obj == m_paletteList) {
                pos = m_paletteList->viewport()->mapFrom(m_paletteList, pos);
            }
            QListWidgetItem* item = m_paletteList->itemAt(pos);
            const int row = item ? item->data(Qt::UserRole).toInt() : -1;
            if (mouseEvent->button() == Qt::RightButton) {
                if (row >= 0 && row < m_paletteColors.size()) {
                    const QColor picked = QColorDialog::getColor(m_paletteColors[row], this, "Select palette color");
                    if (picked.isValid()) {
                        const std::vector<QColor> before(m_paletteColors.begin(), m_paletteColors.end());
                        pushPaletteUndoSnapshot();
                        const cv::Vec3b quant =
                            Rgb565ToBgr(BgrToRgb565(cv::Vec3b(picked.blue(), picked.green(), picked.red())));
                        m_paletteColors[row] = QColor(quant[2], quant[1], quant[0]);
                        if (m_paletteSetIndex >= 0 && m_paletteSetIndex < m_fullPalettes.size()) {
                            m_fullPalettes[m_paletteSetIndex] = m_paletteColors;
                        }
                        refreshPaletteList();
                        applyPaletteColorChanges(before, std::vector<QColor>(m_paletteColors.begin(), m_paletteColors.end()));
                        if (m_paletteList) {
                            m_paletteList->setCurrentRow(row);
                        }
                    }
                }
                return true;
            }
            if (mouseEvent->button() == Qt::LeftButton) {
                m_paletteDragPending = row >= 0;
                m_paletteDragSlot = row;
                m_paletteDragStart = pos;
                if (row >= 0 && row < m_paletteColors.size()) {
                    startPaletteBlink(row);
                }
            }
        }
        if (event->type() == QEvent::MouseMove) {
            auto* mouseEvent = static_cast<QMouseEvent*>(event);
            QPoint pos = mouseEvent->pos();
            if (obj == m_paletteList) {
                pos = m_paletteList->viewport()->mapFrom(m_paletteList, pos);
            }
            if (m_paletteDragPending &&
                (mouseEvent->buttons() & Qt::LeftButton) &&
                (pos - m_paletteDragStart).manhattanLength() >= QApplication::startDragDistance()) {
                if (m_paletteDragSlot >= 0 && m_paletteDragSlot < m_paletteColors.size()) {
                    stopPaletteBlink();
                    m_paletteDragPending = false;
                    QDrag* drag = new QDrag(m_paletteList);
                    QMimeData* mime = new QMimeData();
                    const QColor color = m_paletteColors[m_paletteDragSlot];
                    mime->setData(kPaletteColorMime, EncodePaletteColor(color));
                    drag->setMimeData(mime);
                    QPixmap pixmap(kPaletteSwatchSize, kPaletteSwatchSize);
                    pixmap.fill(color);
                    drag->setPixmap(pixmap);
                    drag->setHotSpot(QPoint(pixmap.width() / 2, pixmap.height() / 2));
                    drag->exec(Qt::CopyAction);
                }
                return true;
            }
        }
        if (event->type() == QEvent::MouseButtonRelease) {
            auto* mouseEvent = static_cast<QMouseEvent*>(event);
            if (mouseEvent->button() == Qt::LeftButton) {
                stopPaletteBlink();
                m_paletteDragPending = false;
                m_paletteDragSlot = -1;
            }
        }
        if (event->type() == QEvent::Leave) {
            stopPaletteBlink();
            m_paletteDragPending = false;
            m_paletteDragSlot = -1;
        }
        if (event->type() == QEvent::DragEnter || event->type() == QEvent::DragMove) {
            auto* dragEvent = static_cast<QDragMoveEvent*>(event);
            QColor color;
            if (DecodePaletteColor(dragEvent->mimeData(), color)) {
                dragEvent->setDropAction(Qt::CopyAction);
                dragEvent->accept();
                return true;
            }
            dragEvent->ignore();
            return true;
        }
        if (event->type() == QEvent::Drop) {
            auto* dropEvent = static_cast<QDropEvent*>(event);
            QColor color;
            if (!DecodePaletteColor(dropEvent->mimeData(), color)) {
                dropEvent->ignore();
                return true;
            }
            QPoint pos = dropEvent->position().toPoint();
            if (obj == m_paletteList) {
                pos = m_paletteList->viewport()->mapFrom(m_paletteList, pos);
            }
            QListWidgetItem* item = m_paletteList->itemAt(pos);
            const int row = item ? item->data(Qt::UserRole).toInt() : -1;
            if (row >= 0 && row < m_paletteColors.size()) {
                const std::vector<QColor> before(m_paletteColors.begin(), m_paletteColors.end());
                pushPaletteUndoSnapshot();
                const cv::Vec3b quant = Rgb565ToBgr(BgrToRgb565(cv::Vec3b(color.blue(), color.green(), color.red())));
                m_paletteColors[row] = QColor(quant[2], quant[1], quant[0]);
                if (m_paletteSetIndex >= 0 && m_paletteSetIndex < m_fullPalettes.size()) {
                    m_fullPalettes[m_paletteSetIndex] = m_paletteColors;
                }
                refreshPaletteList();
                applyPaletteColorChanges(before, std::vector<QColor>(m_paletteColors.begin(), m_paletteColors.end()));
                m_paletteList->setCurrentRow(row);
                dropEvent->setDropAction(Qt::CopyAction);
                dropEvent->accept();
                return true;
            }
            dropEvent->ignore();
            return true;
        }
    }
    if (m_currentColorButton && obj == m_currentColorButton) {
        if (event->type() == QEvent::MouseButtonPress) {
            auto* mouseEvent = static_cast<QMouseEvent*>(event);
            if (mouseEvent->button() == Qt::LeftButton) {
                m_currentColorDragPending = true;
                m_currentColorDragStart = mouseEvent->pos();
            }
        }
        if (event->type() == QEvent::MouseMove) {
            auto* mouseEvent = static_cast<QMouseEvent*>(event);
            if (m_currentColorDragPending &&
                (mouseEvent->buttons() & Qt::LeftButton) &&
                (mouseEvent->pos() - m_currentColorDragStart).manhattanLength() >= QApplication::startDragDistance()) {
                m_currentColorDragPending = false;
                const QColor color(static_cast<int>(m_drawColor[2]),
                                   static_cast<int>(m_drawColor[1]),
                                   static_cast<int>(m_drawColor[0]));
                QDrag* drag = new QDrag(m_currentColorButton);
                QMimeData* mime = new QMimeData();
                mime->setData(kPaletteColorMime, EncodePaletteColor(color));
                drag->setMimeData(mime);
                QPixmap pixmap(kPaletteSwatchSize, kPaletteSwatchSize);
                pixmap.fill(color);
                drag->setPixmap(pixmap);
                drag->setHotSpot(QPoint(pixmap.width() / 2, pixmap.height() / 2));
                drag->exec(Qt::CopyAction);
                return true;
            }
        }
        if (event->type() == QEvent::MouseButtonRelease) {
            auto* mouseEvent = static_cast<QMouseEvent*>(event);
            if (mouseEvent->button() == Qt::LeftButton) {
                m_currentColorDragPending = false;
            }
        }
    }
    auto handlePaletteDrop = [this, obj](QListWidget* list,
                                    QEvent* event,
                                    auto applyFn) -> bool {
        if (!list || (event->type() != QEvent::DragEnter &&
                      event->type() != QEvent::DragMove &&
                      event->type() != QEvent::Drop)) {
            return false;
        }
        if (event->type() == QEvent::DragEnter || event->type() == QEvent::DragMove) {
            auto* dragEvent = static_cast<QDragMoveEvent*>(event);
            QColor color;
            if (DecodePaletteColor(dragEvent->mimeData(), color)) {
                dragEvent->setDropAction(Qt::CopyAction);
                dragEvent->accept();
                return true;
            }
            dragEvent->ignore();
            return true;
        }
        auto* dropEvent = static_cast<QDropEvent*>(event);
        QColor color;
        if (!DecodePaletteColor(dropEvent->mimeData(), color)) {
            dropEvent->ignore();
            return true;
        }
        QPoint pos = dropEvent->position().toPoint();
        if (obj == list) {
            pos = list->viewport()->mapFrom(list, pos);
        }
        QListWidgetItem* item = list->itemAt(pos);
        const int row = item ? item->listWidget()->row(item) : -1;
        if (row >= 0) {
            applyFn(row, color);
            dropEvent->setDropAction(Qt::CopyAction);
            dropEvent->accept();
            return true;
        }
        dropEvent->ignore();
        return true;
    };
    if (m_reducedPaletteList &&
        (obj == m_reducedPaletteList || obj == m_reducedPaletteList->viewport())) {
        return handlePaletteDrop(m_reducedPaletteList, event, [this](int row, const QColor& color) {
            pushReducedUndoSnapshot(m_reducedPaletteIndex);
            setReducedSlotColor(m_reducedPaletteIndex, row, color);
            refreshReducedPaletteButtons();
            if (m_reducedPaletteList) {
                m_reducedPaletteList->setCurrentRow(row);
            }
        });
    }
    if (m_dynamicPaletteList &&
        (obj == m_dynamicPaletteList || obj == m_dynamicPaletteList->viewport())) {
        return handlePaletteDrop(m_dynamicPaletteList, event, [this](int row, const QColor& color) {
            const std::vector<int> targets = targetFrameIndices();
            for (int frameIndex : targets) {
                if (frameIndex < 0 || frameIndex >= static_cast<int>(m_frameDynamicColors.size())) {
                    continue;
                }
                pushDynamicUndoSnapshot(frameIndex, m_dynamicSetIndex);
                setDynamicSlotColor(frameIndex, m_dynamicSetIndex, row, color);
            }
            refreshDynamicPaletteButtons();
            if (m_dynamicPaletteList) {
                m_dynamicPaletteList->setCurrentRow(row);
            }
        });
    }
    if (m_rotationList &&
        (obj == m_rotationList || obj == m_rotationList->viewport())) {
        return handlePaletteDrop(m_rotationList, event, [this](int row, const QColor& color) {
            const int frameIndex = m_framesList ? m_framesList->currentRow() : -1;
            if (frameIndex < 0) {
                return;
            }
            const bool useHd = m_useHdFrame && hasHdFrame(frameIndex);
            pushRotationUndoSnapshot(frameIndex, m_rotationSetIndex, useHd);
            uint16_t* rotations = rotationBlockForEdit(frameIndex, useHd);
            const std::size_t base = rotationSetOffset(m_rotationSetIndex);
            if (!rotations || base + MAX_LENGTH_COLOR_ROTATION > rotationBlockSize()) {
                return;
            }
            const uint16_t length = rotations[base];
            if (row < 0 || row >= static_cast<int>(length)) {
                return;
            }
            const cv::Vec3b bgr(color.blue(), color.green(), color.red());
            const uint16_t value = BgrToRgb565(bgr);
            rotations[base + 2 + static_cast<std::size_t>(row)] = value;
            refreshRotationList();
            if (m_rotationList) {
                m_rotationList->setCurrentRow(row);
            }
            resetCanvasRotationState();
        });
    }
    if (m_framePreviewList &&
        (obj == m_framePreviewList || obj == m_framePreviewList->viewport())) {
        if (event->type() == QEvent::KeyPress) {
            auto* keyEvent = static_cast<QKeyEvent*>(event);
            if (keyEvent->key() == Qt::Key_Escape) {
                clearPreviewSelection();
                return true;
            }
        }
        if (event->type() == QEvent::MouseButtonPress) {
            auto* mouseEvent = static_cast<QMouseEvent*>(event);
            if (mouseEvent->button() == Qt::LeftButton &&
                mouseEvent->modifiers() == Qt::NoModifier) {
                QPoint pos = mouseEvent->position().toPoint();
                if (obj == m_framePreviewList) {
                    pos = m_framePreviewList->viewport()->mapFrom(m_framePreviewList, pos);
                }
                QListWidgetItem* hit = m_framePreviewList->itemAt(pos);
                bool hitContent = false;
                if (hit) {
                    const QRect itemRect = m_framePreviewList->visualItemRect(hit);
                    const QFontMetrics metrics(m_framePreviewList->font());
                    const int padding = 6;
                    const int textHeight = metrics.height() + 4;
                    QRect contentRect = itemRect.adjusted(padding, padding, -padding, -padding);
                    const QVariant sizeData = hit->data(kPreviewIconSizeRole);
                    const QSize iconSize = sizeData.isValid() && sizeData.toSize().isValid()
                        ? sizeData.toSize()
                        : m_framePreviewList->iconSize().isValid()
                            ? m_framePreviewList->iconSize()
                            : QSize(kPreviewIconWidth, kPreviewIconHeight);
                    const int iconX = contentRect.left() + (contentRect.width() - iconSize.width()) / 2;
                    QRect iconRect(iconX,
                                   contentRect.top() + textHeight + 2,
                                   iconSize.width(),
                                   contentRect.height() - textHeight - 2);
                    QRect textRect(iconRect.left(), contentRect.top(), iconRect.width(), textHeight);
                    hitContent = iconRect.contains(pos) || textRect.contains(pos);
                }
                if (!hit || !hitContent) {
                    clearPreviewSelection();
                    return true;
                }
            }
        }
        if (event->type() == QEvent::DragEnter) {
            auto* dragEvent = static_cast<QDragEnterEvent*>(event);
            QString kind;
            int index = -1;
            if (m_previewSelectedOnly && ExtractListDrop(dragEvent->mimeData(), kind, index) &&
                kind == "frame") {
                dragEvent->setDropAction(Qt::CopyAction);
                dragEvent->acceptProposedAction();
                return true;
            }
            dragEvent->ignore();
            return true;
        }
        if (event->type() == QEvent::DragMove) {
            auto* dragEvent = static_cast<QDragMoveEvent*>(event);
            QString kind;
            int index = -1;
            if (m_previewSelectedOnly && ExtractListDrop(dragEvent->mimeData(), kind, index) &&
                kind == "frame") {
                dragEvent->setDropAction(Qt::CopyAction);
                dragEvent->acceptProposedAction();
                return true;
            }
            dragEvent->ignore();
            return true;
        }
        if (event->type() == QEvent::Drop) {
            auto* dropEvent = static_cast<QDropEvent*>(event);
            QString kind;
            int index = -1;
            if (!m_previewSelectedOnly || !ExtractListDrop(dropEvent->mimeData(), kind, index) ||
                kind != "frame") {
                dropEvent->ignore();
                return true;
            }
            const int count = m_frameStore ? m_frameStore->count() : 0;
            if (index < 0 || index >= count) {
                dropEvent->ignore();
                return true;
            }
            if (m_framePreviewList) {
                QSignalBlocker blocker(m_framePreviewList);
                m_framePreviewList->clearSelection();
            }
            if (std::find(m_previewSelectedFrames.begin(),
                          m_previewSelectedFrames.end(),
                          index) == m_previewSelectedFrames.end()) {
                m_previewSelectedFrames.push_back(index);
                std::sort(m_previewSelectedFrames.begin(), m_previewSelectedFrames.end());
                m_previewSelectedFrames.erase(std::unique(m_previewSelectedFrames.begin(),
                                                          m_previewSelectedFrames.end()),
                                              m_previewSelectedFrames.end());
            }
            refreshFramePreviews();
            dropEvent->setDropAction(Qt::CopyAction);
            dropEvent->acceptProposedAction();
            return true;
        }
    }
    if (m_spriteZoneSpritesList &&
        (obj == m_spriteZoneSpritesList || obj == m_spriteZoneSpritesList->viewport())) {
        if (event->type() == QEvent::DragEnter) {
            auto* dragEvent = static_cast<QDragEnterEvent*>(event);
            QString kind;
            int index = -1;
            if (ExtractListDrop(dragEvent->mimeData(), kind, index) && kind == "sprite") {
                dragEvent->setDropAction(Qt::CopyAction);
                dragEvent->acceptProposedAction();
                return true;
            }
            dragEvent->ignore();
            return true;
        }
        if (event->type() == QEvent::DragMove) {
            auto* dragEvent = static_cast<QDragMoveEvent*>(event);
            QString kind;
            int index = -1;
            if (ExtractListDrop(dragEvent->mimeData(), kind, index) && kind == "sprite") {
                dragEvent->setDropAction(Qt::CopyAction);
                dragEvent->acceptProposedAction();
                return true;
            }
            dragEvent->ignore();
            return true;
        }
        if (event->type() == QEvent::Drop) {
            auto* dropEvent = static_cast<QDropEvent*>(event);
            QString kind;
            int index = -1;
            if (!ExtractListDrop(dropEvent->mimeData(), kind, index) || kind != "sprite") {
                dropEvent->ignore();
                return true;
            }
            const int frameIndex = m_framesList ? m_framesList->currentRow() : -1;
            if (frameIndex < 0) {
                dropEvent->ignore();
                return true;
            }
            if (m_selectedSpriteZoneIndex < 0 ||
                m_selectedSpriteZoneIndex >= static_cast<int>(m_spriteZones.size())) {
                statusBar()->showMessage("Select a sprite zone to assign a sprite.", 2000);
                dropEvent->ignore();
                return true;
            }
            if (index < 0 || !m_spritesList || index >= m_spritesList->count()) {
                dropEvent->ignore();
                return true;
            }
            SpriteZoneGroup zone = m_spriteZones[static_cast<std::size_t>(m_selectedSpriteZoneIndex)];
            int slotToUse = -1;
            for (int slot : zone.slotIndices) {
                const std::size_t slotIndex = static_cast<std::size_t>(frameIndex) * MAX_SPRITES_PER_FRAME +
                    static_cast<std::size_t>(slot);
                if (slotIndex < m_frameSpriteAssignments.size() &&
                    m_frameSpriteAssignments[slotIndex] == 255) {
                    slotToUse = slot;
                    break;
                }
            }
            if (slotToUse < 0) {
                for (int slot = 0; slot < MAX_SPRITES_PER_FRAME; ++slot) {
                    const std::size_t slotIndex = static_cast<std::size_t>(frameIndex) * MAX_SPRITES_PER_FRAME +
                        static_cast<std::size_t>(slot);
                    if (slotIndex < m_frameSpriteAssignments.size() &&
                        m_frameSpriteAssignments[slotIndex] == 255) {
                        slotToUse = slot;
                        break;
                    }
                }
            }
            if (slotToUse < 0) {
                statusBar()->showMessage("No empty sprite slots available for this frame.", 2000);
                dropEvent->ignore();
                return true;
            }
            const std::vector<int> targets = targetFrameIndices();
            for (int row : targets) {
                const std::size_t slotIndex = static_cast<std::size_t>(row) * MAX_SPRITES_PER_FRAME +
                    static_cast<std::size_t>(slotToUse);
                if (slotIndex >= m_frameSpriteAssignments.size()) {
                    continue;
                }
                m_frameSpriteAssignments[slotIndex] = static_cast<uint8_t>(index);
                if (slotIndex < m_frameSpriteZoneFlags.size()) {
                    m_frameSpriteZoneFlags[slotIndex] = 1;
                }
                const std::size_t bboxIndex = static_cast<std::size_t>(row) * MAX_SPRITES_PER_FRAME * 4 +
                    static_cast<std::size_t>(slotToUse) * 4;
                if (bboxIndex + 3 < m_frameSpriteBBoxes.size()) {
                    m_frameSpriteBBoxes[bboxIndex] = static_cast<uint16_t>(zone.rect.x());
                    m_frameSpriteBBoxes[bboxIndex + 1] = static_cast<uint16_t>(zone.rect.y());
                    m_frameSpriteBBoxes[bboxIndex + 2] =
                        static_cast<uint16_t>(zone.rect.x() + zone.rect.width() - 1);
                    m_frameSpriteBBoxes[bboxIndex + 3] =
                        static_cast<uint16_t>(zone.rect.y() + zone.rect.height() - 1);
                }
                updateFramePreviewAt(row);
            }
            m_selectedSpriteSlot = slotToUse;
            m_spriteZonePreferredSlot = slotToUse;
            refreshSpriteZoneList();
            refreshSpriteZoneSpritesList();
            if (m_previewFilterEnabled && currentPreviewFilterKind() == PreviewFilterKind::Sprite) {
                updatePreviewFilterState();
            }
            updateMaskPreviewForFrame(m_framesList ? m_framesList->currentRow() : -1);
            updateFrameUsageHighlights(m_framesList ? m_framesList->currentRow() : -1);
            dropEvent->setDropAction(Qt::CopyAction);
            dropEvent->acceptProposedAction();
            return true;
        }
    }
    return QMainWindow::eventFilter(obj, event);
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
    if (m_isLoadingProject) {
        return;
    }
    if (m_frameCanvasOverrideIndex >= 0 && m_frameCanvasOverrideIndex != index) {
        m_frameCanvasOverrideImage.release();
        m_frameCanvasOverrideIndex = -1;
    }
    const cv::Mat* image = m_frameStore->at(index);
    if (image && !image->empty()) {
        m_framesCanvas->canvas()->clearPreviewImage();
        updateFrameCanvasImage(index);
        updatePlaybackIdleFrame(index);
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
        if (m_frameDynamicMaskAssign) {
            QSignalBlocker blockAssign(m_frameDynamicMaskAssign);
            const int selected = currentFrameDynamicMaskId();
            m_frameDynamicMaskAssign->setCurrentIndex(selected >= 0 ? selected + 1 : 0);
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
        if (m_triggerIdSpin && m_triggerMonochromeCheck) {
            QSignalBlocker blockSpin(m_triggerIdSpin);
            QSignalBlocker blockCheck(m_triggerMonochromeCheck);
            if (index >= 0 && index < static_cast<int>(m_frameTriggerIds.size())) {
                const uint32_t trigger = m_frameTriggerIds[static_cast<std::size_t>(index)];
                const bool isMono = trigger == static_cast<uint32_t>(kMonochromeTriggerId);
                m_triggerMonochromeCheck->setChecked(isMono);
                m_triggerMonochromeCheck->setEnabled(true);
                m_triggerIdSpin->setEnabled(true);
                m_triggerIdSpin->setReadOnly(isMono);
                if (isMono) {
                    m_triggerIdSpin->setValue(kMonochromeTriggerId);
                } else if (trigger == 0xffffffffu) {
                    m_triggerIdSpin->setValue(-1);
                } else {
                    m_triggerIdSpin->setValue(static_cast<int>(trigger));
                }
            } else {
                m_triggerMonochromeCheck->setChecked(false);
                m_triggerMonochromeCheck->setEnabled(false);
                m_triggerIdSpin->setEnabled(false);
                m_triggerIdSpin->setReadOnly(false);
                m_triggerIdSpin->setValue(-1);
            }
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
        refreshSpriteZoneList();
        refreshFrameSpriteSlotCombo();
        updateDynamicMaskPreviewIcons();
        syncDynamicSetSelection();
        refreshDynamicPaletteButtons();
        refreshRotationEditor();
        updateFrameUsageHighlights(index);
        updatePlaybackButtons();
    } else {
        updateFrameCanvasImage(-1);
        updatePlaybackIdleFrame(-1);
        if (m_triggerIdSpin && m_triggerMonochromeCheck) {
            QSignalBlocker blockSpin(m_triggerIdSpin);
            QSignalBlocker blockCheck(m_triggerMonochromeCheck);
            m_triggerMonochromeCheck->setChecked(false);
            m_triggerMonochromeCheck->setEnabled(false);
            m_triggerIdSpin->setEnabled(false);
            m_triggerIdSpin->setReadOnly(false);
            m_triggerIdSpin->setValue(-1);
        }
        refreshFrameSpriteSlotCombo();
        refreshRotationEditor();
        updateFrameUsageHighlights(-1);
        updatePlaybackButtons();
    }
}

void MainWindow::showSpriteAtIndex(int index)
{
    if (m_isLoadingProject) {
        return;
    }
    const bool hasHd = hasHdSprite(index);
    if (!hasHd && m_useHdSprite) {
        m_useHdSprite = false;
    }
    if (m_spritesCanvas) {
        m_spritesCanvas->setHdButtonEnabled(hasHd);
        m_spritesCanvas->setHdButtonChecked(m_useHdSprite);
        m_spritesCanvas->setOriginalVisible(m_showSpriteOriginal);
    }
    if (index >= 0) {
        m_spritesCanvas->canvas()->clearPreviewImage();
        updateSpriteCanvasImage(index);
        if (!m_drawPointEnabled) {
            QTimer::singleShot(0, this, [this]() {
                m_spritesCanvas->canvas()->requestFitOnResize(true);
            });
        }
    } else {
        m_spritesCanvas->setImage(cv::Mat());
    }
    if (m_spriteDynamicSetCombo) {
        QSignalBlocker blocker(m_spriteDynamicSetCombo);
        m_spriteDynamicSetCombo->setCurrentIndex(m_spriteDynamicSetIndex);
    }
    if (m_spriteDetAreaCombo) {
        m_spriteDetAreaIndex = std::clamp(m_spriteDetAreaIndex, 0, MAX_SPRITE_DETECT_AREAS - 1);
        QSignalBlocker blocker(m_spriteDetAreaCombo);
        m_spriteDetAreaCombo->setCurrentIndex(m_spriteDetAreaIndex);
        m_spriteDetAreaCombo->setEnabled(index >= 0);
    }
    if (m_spriteDetAreaClearButton) {
        m_spriteDetAreaClearButton->setEnabled(index >= 0);
    }
    updateHdControlsForContext();
}

void MainWindow::showBackgroundAtIndex(int index)
{
    if (m_isLoadingProject) {
        return;
    }
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

void MainWindow::updateProjectLabelHeight()
{
    if (!m_projectLabel) {
        return;
    }
    const int width = std::max(0, m_projectLabel->width());
    if (width == 0) {
        m_projectLabel->adjustSize();
        return;
    }
    const QFontMetrics metrics(m_projectLabel->font());
    const QRect bounds = metrics.boundingRect(QRect(0, 0, width, 100000),
                                              Qt::TextWordWrap,
                                              m_projectLabel->text());
    const int height = std::max(bounds.height(), metrics.lineSpacing());
    m_projectLabel->setFixedHeight(height + 2);
    m_projectLabel->updateGeometry();
}

bool MainWindow::setupSerumData(const QString& cromcPath,
                                const LegacyProject& legacy,
                                std::string* error)
{
    m_serumData.Clear();
    m_serumDataLoaded = false;
    if (!cromcPath.isEmpty() && QFileInfo::exists(cromcPath)) {
        const uint8_t flags = FLAG_REQUEST_32P_FRAMES | FLAG_REQUEST_64P_FRAMES;
        if (m_serumData.LoadFromFile(cromcPath.toStdString().c_str(), flags)) {
            m_serumDataLoaded = true;
            return true;
        }
    }
    if (!BuildConcentrateData(legacy, m_serumData, error)) {
        return false;
    }
    m_serumDataLoaded = true;
    return true;
}

bool MainWindow::ensureSerumRuntime()
{
    if (m_serumRuntimeFrame) {
        return true;
    }
    if (m_serumRuntimeAltDir.isEmpty() || m_serumRuntimeRomName.isEmpty()) {
        return false;
    }
    Serum_SetGenerateCRomC(false);
    const uint8_t flags = FLAG_REQUEST_32P_FRAMES |
                          FLAG_REQUEST_64P_FRAMES |
                          FLAG_REQUEST_FILL_MODIFIED_ELEMENTS;
    m_serumRuntimeFrame = Serum_Load(m_serumRuntimeAltDir.toUtf8().constData(),
                                     m_serumRuntimeRomName.toUtf8().constData(),
                                     flags);
    if (!m_serumRuntimeFrame) {
        logLine(QString("Playback: Serum_Load failed (altcolor=%1 rom=%2)")
                    .arg(m_serumRuntimeAltDir, m_serumRuntimeRomName));
        return false;
    }
    return true;
}

void MainWindow::disposeSerumRuntime()
{
    if (!m_serumRuntimeFrame) {
        return;
    }
    Serum_Dispose();
    m_serumRuntimeFrame = nullptr;
}

void MainWindow::configureFrameStoreAdapter()
{
    if (!m_frameStore) {
        return;
    }
    if (!m_serumDataLoaded) {
        m_frameStore->setAdapter({}, {}, {});
        m_frameStore->setCount(-1);
        return;
    }
    m_frameStore->setCacheLimit(m_frameCacheLimit);
    m_frameStore->setAdapter({}, [this](int index) -> cv::Mat {
        if (!m_serumDataLoaded || index < 0 || index >= static_cast<int>(m_serumData.nframes)) {
            return cv::Mat();
        }
        if (!m_serumData.cframes_v2.hasData(static_cast<uint32_t>(index))) {
            return cv::Mat();
        }
        const uint16_t* data = m_serumData.cframes_v2[static_cast<uint32_t>(index)];
        if (!data) {
            return cv::Mat();
        }
        if (m_serumData.fwidth == 0 || m_serumData.fheight == 0) {
            return cv::Mat();
        }
        return ConvertRgb565ToBgrMat(data,
                                     static_cast<int>(m_serumData.fwidth),
                                     static_cast<int>(m_serumData.fheight));
    }, [this](int index, const cv::Mat& image) {
        commitFrameToSerum(index, image, false);
    });
    m_frameStore->setCount(static_cast<int>(m_serumData.nframes));
}

void MainWindow::configureSpriteStoreAdapter()
{
    if (!m_spriteStore) {
        return;
    }
    if (!m_serumDataLoaded) {
        m_spriteStore->setAdapter({}, {}, {});
        m_spriteStore->setCount(-1);
        return;
    }
    m_spriteStore->setCacheLimit(m_spriteCacheLimit);
    m_spriteStore->setAdapter({}, [this](int index) -> cv::Mat {
        if (!m_serumDataLoaded || index < 0 || index >= static_cast<int>(m_serumData.nsprites)) {
            return cv::Mat();
        }
        const uint16_t* data = m_serumData.spritecolored[static_cast<uint32_t>(index)];
        if (!data) {
            return cv::Mat();
        }
        const QSize spriteSize = serumSpriteBaseSize();
        if (spriteSize.isEmpty()) {
            return cv::Mat();
        }
        return ConvertRgb565ToBgrMat(data, spriteSize.width(), spriteSize.height());
    }, [this](int index, const cv::Mat& image) {
        if (!m_serumDataLoaded || index < 0 || image.empty()) {
            return;
        }
        const std::vector<uint16_t> sprite565 = ConvertBgrMatToRgb565(image);
        m_serumData.spritecolored.set(static_cast<uint32_t>(index),
                                      sprite565.data(),
                                      sprite565.size());
    });
    m_spriteStore->setCount(static_cast<int>(m_serumData.nsprites));
}

void MainWindow::configureBackgroundStoreAdapter()
{
    if (!m_backgroundStore) {
        return;
    }
    if (!m_serumDataLoaded) {
        m_backgroundStore->setAdapter({}, {}, {});
        m_backgroundStore->setCount(-1);
        return;
    }
    m_backgroundStore->setCacheLimit(m_backgroundCacheLimit);
    m_backgroundStore->setAdapter({}, [this](int index) -> cv::Mat {
        if (!m_serumDataLoaded || index < 0 || index >= static_cast<int>(m_serumData.nbackgrounds)) {
            return cv::Mat();
        }
        const uint16_t* data = m_serumData.backgroundframes_v2[static_cast<uint32_t>(index)];
        if (!data || m_serumData.fwidth == 0 || m_serumData.fheight == 0) {
            return cv::Mat();
        }
        return ConvertRgb565ToBgrMat(data,
                                     static_cast<int>(m_serumData.fwidth),
                                     static_cast<int>(m_serumData.fheight));
    }, [this](int index, const cv::Mat& image) {
        if (!m_serumDataLoaded || index < 0 || image.empty()) {
            return;
        }
        const std::vector<uint16_t> bg565 = ConvertBgrMatToRgb565(image);
        m_serumData.backgroundframes_v2.set(static_cast<uint32_t>(index),
                                            bg565.data(),
                                            bg565.size());
    });
    m_backgroundStore->setCount(static_cast<int>(m_serumData.nbackgrounds));
}

void MainWindow::commitFrameEdits(const std::vector<int>& indices, bool useHd)
{
    if (!m_serumDataLoaded) {
        return;
    }
    for (int index : indices) {
        commitFrameFromStore(index, useHd);
    }
}

void MainWindow::commitFrameToSerum(int index, const cv::Mat& image, bool useHd)
{
    if (!m_serumDataLoaded || index < 0 || image.empty()) {
        return;
    }
    const int width = image.cols;
    const int height = image.rows;
    if (width <= 0 || height <= 0) {
        return;
    }
    std::vector<uint16_t> frame565 = ConvertBgrMatToRgb565(image);
    if (useHd) {
        const uint8_t extra = 1;
        m_serumData.isextraframe.setIndex(static_cast<uint32_t>(index), &extra, 1);
        m_serumData.cframes_v2_extra.set(static_cast<uint32_t>(index),
                                         frame565.data(),
                                         frame565.size(),
                                         &m_serumData.isextraframe);
    } else {
        m_serumData.cframes_v2.set(static_cast<uint32_t>(index), frame565.data(), frame565.size());
    }
}

void MainWindow::commitFrameFromStore(int index, bool useHd)
{
    if (!m_serumDataLoaded || index < 0) {
        return;
    }
    if (!useHd) {
        if (m_frameStore) {
            m_frameStore->flushIndex(index);
        }
        return;
    }
    if (index >= 0 && index < static_cast<int>(m_frameExtraFrames.size())) {
        const cv::Mat& image = m_frameExtraFrames[static_cast<std::size_t>(index)];
        if (!image.empty()) {
            commitFrameToSerum(index, image, true);
        }
    }
}

void MainWindow::handleToolPress(bool isFrame,
                                 int x,
                                 int y,
                                 Qt::MouseButton button,
                                 Qt::KeyboardModifiers modifiers)
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
        const int gap = FrameGapForWidth(layout.topWidth);
        int localX = x;
        int localY = y;
        m_frameDrawOnMask = false;
        m_frameDrawOnZone = false;
        if (m_spriteZoneMode) {
            if (m_selectedSpriteZoneIndex < 0 ||
                m_selectedSpriteZoneIndex >= static_cast<int>(m_spriteZones.size())) {
                statusBar()->showMessage("Select a sprite zone before editing.", 2000);
                return;
            }
            const SpriteZoneGroup& zone = m_spriteZones[static_cast<std::size_t>(m_selectedSpriteZoneIndex)];
            if (zone.slotIndices.empty()) {
                statusBar()->showMessage("Select a sprite zone before editing.", 2000);
                return;
            }
            if (std::find(zone.slotIndices.begin(), zone.slotIndices.end(), m_selectedSpriteSlot) == zone.slotIndices.end()) {
                m_selectedSpriteSlot = zone.slotIndices.front();
            }
            const bool useBottom = m_showOriginalFrame && !reference.empty();
            if (useBottom) {
                if (y < layout.topHeight + gap || y >= layout.topHeight + gap + layout.bottomHeight ||
                    x < layout.bottomX || x >= layout.bottomX + layout.bottomWidth) {
                    return;
                }
                localX = x - layout.bottomX;
                localY = y - (layout.topHeight + gap);
            } else {
                if (y < 0 || y >= layout.topHeight ||
                    x < layout.topX || x >= layout.topX + layout.topWidth) {
                    return;
                }
                localX = x - layout.topX;
                localY = y;
            }
            const int baseWidth = reference.empty() ? topFrame->cols : reference.cols;
            const int baseHeight = reference.empty() ? topFrame->rows : reference.rows;
            const int scaleX = (useBottom && baseWidth > 0) ? layout.bottomWidth / baseWidth : 1;
            const int scaleY = (useBottom && baseHeight > 0) ? layout.bottomHeight / baseHeight : 1;
            const int mappedX = (scaleX > 1) ? localX / scaleX : localX;
            const int mappedY = (scaleY > 1) ? localY / scaleY : localY;
            if (mappedX < 0 || mappedY < 0 || mappedX >= baseWidth || mappedY >= baseHeight) {
                return;
            }
            m_frameDrawOnZone = true;
            m_frameStart = QPoint(mappedX, mappedY);
            m_frameHasStart = true;
            m_frameStartButton = button;
            return;
        }
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
            cv::Mat* baseMask = activeBackgroundMask(index);
            if (!baseMask || baseMask->empty()) {
                return;
            }
            if (m_drawTool == DrawTool::MagicFill || m_drawTool == DrawTool::Point) {
                if (localX < 0 || localY < 0 || localX >= baseMask->cols || localY >= baseMask->rows) {
                    return;
                }
                if (!m_frameUndoActive) {
                    const std::vector<int> targets = targetFrameIndices();
                    for (int frameIndex : targets) {
                        pushBackgroundMaskUndoSnapshot(frameIndex);
                    }
                    m_frameUndoActive = true;
                }
                const QPoint basePoint(localX, localY);
                const QSize baseSize(baseMask->cols, baseMask->rows);
                const std::vector<int> targets = targetFrameIndices();
                for (int frameIndex : targets) {
                    cv::Mat* mask = activeBackgroundMask(frameIndex);
                    if (!mask || mask->empty()) {
                        continue;
                    }
                    const QPoint mapped = ScalePointToSize(basePoint, baseSize, QSize(mask->cols, mask->rows));
                    const bool erase = (button == Qt::RightButton) || modifiers.testFlag(Qt::ShiftModifier);
                    if (m_drawTool == DrawTool::MagicFill) {
                        applyMaskFill(*mask, mapped.x(), mapped.y(), erase);
                    } else {
                        applyToolToMask(*mask,
                                        DrawTool::Point,
                                        mapped,
                                        mapped,
                                        erase);
                    }
                    updateFramePreviewAt(frameIndex);
                }
                updateFrameCanvasImage(index);
                updateMaskPreviewForFrame(index);
                return;
            }
            if (m_drawTool == DrawTool::ColorPicker) {
                return;
            }
            if (!m_frameUndoActive) {
                const std::vector<int> targets = targetFrameIndices();
                for (int frameIndex : targets) {
                    pushBackgroundMaskUndoSnapshot(frameIndex);
                }
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
            int dynamicSetId = -1;
            if (m_maskMode == MaskMode::Comparison) {
                const int assigned = currentFrameMaskId();
                const int selected = m_maskList->currentRow();
                if (assigned < 0 || assigned != selected) {
                    statusBar()->showMessage("Assign the selected mask to this frame before editing.", 2000);
                    return;
                }
                mask = activeComparisonMask();
            } else if (m_maskMode == MaskMode::Dynamic) {
                dynamicSetId = currentFrameDynamicMaskId();
                if (dynamicSetId < 0) {
                    statusBar()->showMessage("Select a dynamic mask to edit.", 2000);
                    return;
                }
                mask = activeDynamicMaskMap(index);
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
                    if (m_maskMode == MaskMode::Dynamic) {
                        const std::vector<int> targets = targetFrameIndices();
                        for (int frameIndex : targets) {
                            pushMaskUndoSnapshot(m_maskMode, frameIndex);
                        }
                    } else {
                        pushMaskUndoSnapshot(m_maskMode, index);
                    }
                    m_frameUndoActive = true;
                }
                const QPoint basePoint(mappedX, mappedY);
                const QSize baseSize(mask->cols, mask->rows);
                if (m_maskMode == MaskMode::Dynamic) {
                    const bool erase = (button == Qt::RightButton) || modifiers.testFlag(Qt::ShiftModifier);
                    const std::vector<int> targets = targetFrameIndices();
                    for (int frameIndex : targets) {
                        cv::Mat* targetMask = activeDynamicMaskMap(frameIndex);
                        if (!targetMask || targetMask->empty()) {
                            continue;
                        }
                        const QPoint mapped = ScalePointToSize(basePoint, baseSize,
                                                               QSize(targetMask->cols, targetMask->rows));
                        if (m_drawTool == DrawTool::MagicFill) {
                            applyDynamicMaskFill(*targetMask, dynamicSetId, mapped.x(), mapped.y(), erase);
                        } else {
                            applyToolToDynamicMask(*targetMask,
                                                   dynamicSetId,
                                                   DrawTool::Point,
                                                   mapped,
                                                   mapped,
                                                   erase);
                        }
                        updateFramePreviewAt(frameIndex);
                    }
                } else {
                    const bool erase = (button == Qt::RightButton) || modifiers.testFlag(Qt::ShiftModifier);
                    if (m_drawTool == DrawTool::MagicFill) {
                        applyMaskFill(*mask, mappedX, mappedY, erase);
                    } else {
                        applyToolToMask(*mask,
                                        DrawTool::Point,
                                        basePoint,
                                        basePoint,
                                        erase);
                    }
                    const std::vector<int> targets = targetFrameIndices();
                    for (int frameIndex : targets) {
                        updateFramePreviewAt(frameIndex);
                    }
                }
                if (m_maskMode == MaskMode::Comparison) {
                    updateMaskPreviewIcons();
                    updatePreviewsForMaskId(m_maskList ? m_maskList->currentRow() : -1);
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
                if (m_maskMode == MaskMode::Dynamic) {
                    const std::vector<int> targets = targetFrameIndices();
                    for (int frameIndex : targets) {
                        pushMaskUndoSnapshot(m_maskMode, frameIndex);
                    }
                } else {
                    pushMaskUndoSnapshot(m_maskMode, index);
                }
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
    if (!isFrame && m_spriteDetAreaMode) {
        const int index = m_spritesList ? m_spritesList->currentRow() : -1;
        if (index < 0) {
            return;
        }
        ensureSpriteDataSize();
        const cv::Mat* spriteImage = m_spriteStore ? m_spriteStore->at(index) : nullptr;
        if (!spriteImage || spriteImage->empty()) {
            return;
        }
        const QRect contentRect = spriteContentRect(index);
        const int offsetX = contentRect.isValid() ? contentRect.x() : 0;
        const int offsetY = contentRect.isValid() ? contentRect.y() : 0;
        const int baseWidth = contentRect.isValid() ? contentRect.width() : spriteImage->cols;
        const int baseHeight = contentRect.isValid() ? contentRect.height() : spriteImage->rows;
        cv::Mat originalRef;
        if (const cv::Mat* originalSource = spriteOriginalForDisplay(index)) {
            originalRef = (*originalSource)(cv::Rect(offsetX, offsetY, baseWidth, baseHeight));
        }
        if (originalRef.empty()) {
            return;
        }
        cv::Mat cleaned = originalRef.clone();
        for (int yy = 0; yy < cleaned.rows; ++yy) {
            uint8_t* row = cleaned.ptr<uint8_t>(yy);
            for (int xx = 0; xx < cleaned.cols; ++xx) {
                if (row[xx] == 255) {
                    row[xx] = 0;
                }
            }
        }
        cv::Mat original = buildOriginalFrame(cleaned);
        cv::Mat displayOriginal = BuildDisplayOriginal(original, cv::Size(baseWidth, baseHeight));
        FrameLayout layout = BuildFrameLayout(baseWidth, baseHeight,
                                              displayOriginal.cols, displayOriginal.rows);
        const int gap = FrameGapForWidth(layout.topWidth);
        if (y < layout.topHeight + gap || y >= layout.topHeight + gap + layout.bottomHeight ||
            x < layout.bottomX || x >= layout.bottomX + layout.bottomWidth) {
            return;
        }
        int localX = x - layout.bottomX;
        int localY = y - (layout.topHeight + gap);
        const int scaleX = (layout.bottomWidth > 0 && original.cols > 0) ? layout.bottomWidth / original.cols : 1;
        const int scaleY = (layout.bottomHeight > 0 && original.rows > 0) ? layout.bottomHeight / original.rows : 1;
        const int mappedX = (scaleX > 1) ? localX / scaleX : localX;
        const int mappedY = (scaleY > 1) ? localY / scaleY : localY;
        if (mappedX < 0 || mappedY < 0 || mappedX >= original.cols || mappedY >= original.rows) {
            return;
        }
        if (m_drawTool == DrawTool::ColorPicker) {
            return;
        }
        if (!m_spriteUndoActive) {
            pushSpriteDetAreaUndoSnapshot(index);
            m_spriteUndoActive = true;
        }
        m_spriteStart = QPoint(mappedX, mappedY);
        m_spriteHasStart = true;
        m_spriteStartButton = button;
        return;
    }
    if (!isFrame && m_spriteDynamicMaskMode) {
        const int index = m_spritesList ? m_spritesList->currentRow() : -1;
        if (index < 0) {
            return;
        }
        ensureSpriteDataSize();
        cv::Mat* map = activeSpriteDynamicMask(index);
        if (!map || map->empty()) {
            return;
        }
        const cv::Mat* spriteImage = activeSpriteImage(index);
        const QRect contentRect = spriteContentRect(index);
        const QRect displayRect = spriteImage ? spriteDisplayRect(index, *spriteImage) : QRect();
        const int offsetX = displayRect.isValid() ? displayRect.x() : 0;
        const int offsetY = displayRect.isValid() ? displayRect.y() : 0;
        const QSize baseSize = displayRect.isValid()
            ? QSize(displayRect.width(), displayRect.height())
            : QSize(spriteImage && !spriteImage->empty() ? spriteImage->cols : map->cols,
                    spriteImage && !spriteImage->empty() ? spriteImage->rows : map->rows);
        cv::Mat originalRef;
        if (const cv::Mat* originalSource = spriteOriginalForDisplay(index)) {
            originalRef = contentRect.isValid()
                ? (*originalSource)(cv::Rect(contentRect.x(), contentRect.y(), contentRect.width(), contentRect.height()))
                : *originalSource;
        }
        cv::Mat original = originalRef.empty() ? cv::Mat() : buildOriginalFrame(originalRef);
        cv::Mat displayOriginal = BuildDisplayOriginal(original, cv::Size(baseSize.width(), baseSize.height()));
        FrameLayout layout = BuildFrameLayout(baseSize.width(), baseSize.height(),
                                              displayOriginal.cols, displayOriginal.rows);
        if (x < layout.topX || x >= layout.topX + layout.topWidth ||
            y < 0 || y >= layout.topHeight) {
            return;
        }
        const QPoint mappedLocal = ScalePointToSize(QPoint(x - layout.topX, y),
                                                    QSize(layout.topWidth, layout.topHeight),
                                                    QSize(baseSize.width(), baseSize.height()));
        const QPoint mapped(mappedLocal.x() + offsetX, mappedLocal.y() + offsetY);
        if (mapped.x() < 0 || mapped.y() < 0 || mapped.x() >= map->cols || mapped.y() >= map->rows) {
            return;
        }
        const cv::Mat* spriteMask = spriteOriginalForDisplay(index);
        cv::Mat maskScaled;
        if (spriteMask && !spriteMask->empty()) {
            if (spriteMask->size() != map->size()) {
                cv::resize(*spriteMask, maskScaled, map->size(), 0.0, 0.0, cv::INTER_NEAREST);
            } else {
                maskScaled = *spriteMask;
            }
            if (maskScaled.at<uint8_t>(mapped.y(), mapped.x()) == 255) {
                return;
            }
        }
        if (m_drawTool == DrawTool::MagicFill || m_drawTool == DrawTool::Point) {
            if (!m_spriteUndoActive) {
                pushSpriteDynamicMaskUndoSnapshot(index);
                m_spriteUndoActive = true;
            }
            const bool erase = (button == Qt::RightButton) || modifiers.testFlag(Qt::ShiftModifier);
            if (m_drawTool == DrawTool::MagicFill) {
                applyDynamicMaskFill(*map, m_spriteDynamicSetIndex, mapped.x(), mapped.y(), erase);
            } else {
                applyToolToDynamicMask(*map, m_spriteDynamicSetIndex, DrawTool::Point, mapped, mapped, erase);
            }
            if (!maskScaled.empty()) {
                ApplySpriteMaskConstraints(*map, maskScaled);
            }
            updateSpriteCanvasImage(index);
            return;
        }
        if (m_drawTool == DrawTool::ColorPicker) {
            return;
        }
        if (!m_spriteUndoActive) {
            pushSpriteDynamicMaskUndoSnapshot(index);
            m_spriteUndoActive = true;
        }
        m_spriteStart = mapped;
        m_spriteHasStart = true;
        m_spriteStartButton = button;
        return;
    }
    cv::Mat* image = isFrame ? activeFrameImage(m_framesList->currentRow(), true)
                             : activeSpriteImageMutable(m_spritesList ? m_spritesList->currentRow() : -1);
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
    } else {
        const int index = m_spritesList ? m_spritesList->currentRow() : -1;
        const QRect contentRect = spriteContentRect(index);
        const QRect displayRect = spriteDisplayRect(index, *image);
        const int offsetX = displayRect.isValid() ? displayRect.x() : 0;
        const int offsetY = displayRect.isValid() ? displayRect.y() : 0;
        const int baseWidth = displayRect.isValid() ? displayRect.width() : image->cols;
        const int baseHeight = displayRect.isValid() ? displayRect.height() : image->rows;
        cv::Mat originalRef;
        if (const cv::Mat* originalSource = spriteOriginalForDisplay(index)) {
            originalRef = contentRect.isValid()
                ? (*originalSource)(cv::Rect(contentRect.x(), contentRect.y(),
                                             contentRect.width(), contentRect.height()))
                : *originalSource;
        }
        cv::Mat original = originalRef.empty() ? cv::Mat() : buildOriginalFrame(originalRef);
        cv::Mat displayOriginal = BuildDisplayOriginal(original, cv::Size(baseWidth, baseHeight));
        FrameLayout layout = BuildFrameLayout(baseWidth, baseHeight,
                                              displayOriginal.cols, displayOriginal.rows);
        if (y < 0 || y >= layout.topHeight ||
            x < layout.topX || x >= layout.topX + layout.topWidth) {
            statusBar()->showMessage("Sprite edits apply to the top sprite only.", 2000);
            return;
        }
        x -= layout.topX;
        x += offsetX;
        y += offsetY;
    }
    const int currentIndex = isFrame ? m_framesList->currentRow() : m_spritesList->currentRow();
    bool& undoActive = isFrame ? m_frameUndoActive : m_spriteUndoActive;
    if (m_drawTool != DrawTool::ColorPicker) {
        if (!undoActive) {
            if (isFrame) {
                const std::vector<int> targets = targetFrameIndices();
                for (int frameIndex : targets) {
                    pushUndoSnapshot(true, frameIndex);
                }
            } else {
                pushUndoSnapshot(false, currentIndex);
            }
            undoActive = true;
        }
    }
        if (m_drawTool == DrawTool::Point) {
            if (isFrame) {
                const QPoint basePoint(x, y);
                const QSize baseSize(image->cols, image->rows);
                const std::vector<int> targets = targetFrameIndices();
                for (int frameIndex : targets) {
                    cv::Mat* target = activeFrameImage(frameIndex, true);
                    if (!target || target->empty()) {
                        continue;
                    }
                    const QPoint mapped = ScalePointToSize(basePoint, baseSize,
                                                           QSize(target->cols, target->rows));
                    const cv::Mat spriteMask = buildSpriteCoverageMask(frameIndex, m_useHdFrame);
                    if (!spriteMask.empty() &&
                        mapped.y() >= 0 && mapped.y() < spriteMask.rows &&
                        mapped.x() >= 0 && mapped.x() < spriteMask.cols &&
                        spriteMask.at<uint8_t>(mapped.y(), mapped.x())) {
                        continue;
                    }
                    applyToolToImage(*target, DrawTool::Point, mapped, mapped, button == Qt::RightButton);
                    commitFrameFromStore(frameIndex, m_useHdFrame);
                    updateFramePreviewAt(frameIndex);
                }
            } else {
                applyToolToImage(*image, DrawTool::Point, QPoint(x, y), QPoint(x, y), button == Qt::RightButton);
            }
    } else if (m_drawTool == DrawTool::ColorPicker) {
        pickColorFromImage(*image, x, y);
        } else if (m_drawTool == DrawTool::MagicFill) {
            if (isFrame) {
                const QPoint basePoint(x, y);
                const QSize baseSize(image->cols, image->rows);
                const std::vector<int> targets = targetFrameIndices();
                for (int frameIndex : targets) {
                    cv::Mat* target = activeFrameImage(frameIndex, true);
                    if (!target || target->empty()) {
                        continue;
                    }
                    const QPoint mapped = ScalePointToSize(basePoint, baseSize,
                                                           QSize(target->cols, target->rows));
                    cv::Mat backup = target->clone();
                    applyMagicFill(*target, mapped.x(), mapped.y());
                    const cv::Mat spriteMask = buildSpriteCoverageMask(frameIndex, m_useHdFrame);
                    if (!spriteMask.empty()) {
                        restoreSpriteCoverage(*target, backup, spriteMask);
                    }
                    commitFrameFromStore(frameIndex, m_useHdFrame);
                    updateFramePreviewAt(frameIndex);
                }
            } else {
                applyMagicFill(*image, x, y);
            }
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
        updateSpriteCanvasImage(m_spritesList ? m_spritesList->currentRow() : -1);
    }
}

void MainWindow::handleToolDrag(bool isFrame,
                                int x,
                                int y,
                                Qt::MouseButtons buttons,
                                Qt::KeyboardModifiers modifiers)
{
    if (!m_drawPointEnabled) {
        return;
    }
    if (isFrame && m_frameDrawOnZone) {
        const int index = m_framesList ? m_framesList->currentRow() : -1;
        if (index < 0) {
            return;
        }
        const cv::Mat* topFrame = activeFrameImage(index, false);
        if (!topFrame || topFrame->empty()) {
            return;
        }
        cv::Mat reference = buildOriginalPreviewForIndex(index);
        FrameLayout layout = BuildFrameLayout(*topFrame, reference);
        const int gap = FrameGapForWidth(layout.topWidth);
        const bool useBottom = m_showOriginalFrame && !reference.empty();
        int localX = x;
        int localY = y;
        if (useBottom) {
            if (y < layout.topHeight + gap || y >= layout.topHeight + gap + layout.bottomHeight ||
                x < layout.bottomX || x >= layout.bottomX + layout.bottomWidth) {
                return;
            }
            localX = x - layout.bottomX;
            localY = y - (layout.topHeight + gap);
        } else {
            if (y < 0 || y >= layout.topHeight ||
                x < layout.topX || x >= layout.topX + layout.topWidth) {
                return;
            }
            localX = x - layout.topX;
            localY = y;
        }
        const int baseWidth = reference.empty() ? topFrame->cols : reference.cols;
        const int baseHeight = reference.empty() ? topFrame->rows : reference.rows;
        const int scaleX = (useBottom && baseWidth > 0) ? layout.bottomWidth / baseWidth : 1;
        const int scaleY = (useBottom && baseHeight > 0) ? layout.bottomHeight / baseHeight : 1;
        const int mappedX = (scaleX > 1) ? localX / scaleX : localX;
        const int mappedY = (scaleY > 1) ? localY / scaleY : localY;
        if (mappedX < 0 || mappedY < 0 || mappedX >= baseWidth || mappedY >= baseHeight) {
            return;
        }
        if (!m_frameHasStart) {
            return;
        }
        const int x0 = std::clamp(m_frameStart.x(), 0, baseWidth - 1);
        const int y0 = std::clamp(m_frameStart.y(), 0, baseHeight - 1);
        const int x1 = std::clamp(mappedX, 0, baseWidth - 1);
        const int y1 = std::clamp(mappedY, 0, baseHeight - 1);
        const int left = std::min(x0, x1);
        const int top = std::min(y0, y1);
        const int right = std::max(x0, x1);
        const int bottom = std::max(y0, y1);
        cv::Mat previewMask(cv::Size(baseWidth, baseHeight), CV_8UC1, cv::Scalar(0));
        cv::rectangle(previewMask, cv::Rect(left, top, right - left + 1, bottom - top + 1), cv::Scalar(1), cv::FILLED);
        cv::Mat otherMask(cv::Size(baseWidth, baseHeight), CV_8UC1, cv::Scalar(0));
        const std::vector<SpriteZoneGroup> zones = buildSpriteZonesForFrame(index);
        for (std::size_t i = 0; i < zones.size(); ++i) {
            if (m_selectedSpriteZoneIndex >= 0 && static_cast<int>(i) == m_selectedSpriteZoneIndex) {
                continue;
            }
            const SpriteZoneGroup& zone = zones[i];
            cv::Rect rect(zone.rect.x(), zone.rect.y(), zone.rect.width(), zone.rect.height());
            rect &= cv::Rect(0, 0, baseWidth, baseHeight);
            if (rect.width <= 0 || rect.height <= 0) {
                continue;
            }
            cv::rectangle(otherMask, rect, cv::Scalar(1), cv::FILLED);
        }
        QRect region;
        if (useBottom) {
            region = QRect(layout.bottomX,
                           layout.topHeight + gap,
                           layout.bottomWidth,
                           layout.bottomHeight);
        } else {
            region = QRect(layout.topX, 0, layout.topWidth, layout.topHeight);
        }
        m_framesCanvas->canvas()->setMaskOutline(previewMask, QColor(255, 200, 0), region);
        if (MaskHasContent(otherMask)) {
            m_framesCanvas->canvas()->setSecondaryMaskOutline(otherMask, QColor(0, 200, 255), region);
        } else {
            m_framesCanvas->canvas()->setSecondaryMaskOutline(cv::Mat(), QColor());
        }
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
            const bool erase = buttons.testFlag(Qt::RightButton) || modifiers.testFlag(Qt::ShiftModifier);
            const QPoint basePoint(x, y);
            const QSize baseSize(mask->cols, mask->rows);
            const std::vector<int> targets = targetFrameIndices();
            for (int frameIndex : targets) {
                cv::Mat* targetMask = activeBackgroundMask(frameIndex);
                if (!targetMask || targetMask->empty()) {
                    continue;
                }
                const QPoint mapped = ScalePointToSize(basePoint, baseSize,
                                                       QSize(targetMask->cols, targetMask->rows));
                applyToolToMask(*targetMask, DrawTool::Point, mapped, mapped, erase);
                updateFramePreviewAt(frameIndex);
            }
            updateFrameCanvasImage(index);
            return;
        }
        if (!m_frameHasStart) {
            return;
        }
        cv::Mat previewMask = mask->clone();
        const bool erase = buttons.testFlag(Qt::RightButton) || modifiers.testFlag(Qt::ShiftModifier);
        applyToolToMask(previewMask, m_drawTool, m_frameStart, QPoint(x, y), erase);
        cv::Mat base = renderFrameWithSerum(index, m_useHdFrame);
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
        const int gap = FrameGapForWidth(layout.topWidth);
        if (y < layout.topHeight + gap || y >= layout.topHeight + gap + layout.bottomHeight ||
            x < layout.bottomX || x >= layout.bottomX + layout.bottomWidth) {
            return;
        }
        x -= layout.bottomX;
        y -= (layout.topHeight + gap);
        cv::Mat* mask = nullptr;
        cv::Vec3b color(200, 0, 200);
        int dynamicSetId = -1;
        if (m_maskMode == MaskMode::Comparison) {
            const int assigned = currentFrameMaskId();
            const int selected = m_maskList->currentRow();
            if (assigned < 0 || assigned != selected) {
                return;
            }
            mask = activeComparisonMask();
            color = cv::Vec3b(200, 0, 200);
        } else if (m_maskMode == MaskMode::Dynamic) {
            dynamicSetId = currentFrameDynamicMaskId();
            if (dynamicSetId < 0) {
                return;
            }
            mask = activeDynamicMaskMap(index);
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
            const bool erase = buttons.testFlag(Qt::RightButton) ||
                (m_maskMode == MaskMode::Dynamic && modifiers.testFlag(Qt::ShiftModifier)) ||
                (m_maskMode == MaskMode::Comparison && modifiers.testFlag(Qt::ShiftModifier));
            const QPoint basePoint(x, y);
            const QSize baseSize(mask->cols, mask->rows);
            if (m_maskMode == MaskMode::Dynamic) {
                const std::vector<int> targets = targetFrameIndices();
                for (int frameIndex : targets) {
                    cv::Mat* targetMask = activeDynamicMaskMap(frameIndex);
                    if (!targetMask || targetMask->empty()) {
                        continue;
                    }
                    const QPoint mapped = ScalePointToSize(basePoint, baseSize,
                                                           QSize(targetMask->cols, targetMask->rows));
                    applyToolToDynamicMask(*targetMask, dynamicSetId, DrawTool::Point, mapped, mapped, erase);
                    updateFramePreviewAt(frameIndex);
                }
            } else {
                applyToolToMask(*mask, DrawTool::Point, basePoint, basePoint, erase);
                updatePreviewsForMaskId(m_maskList ? m_maskList->currentRow() : -1);
            }
            if (m_maskMode == MaskMode::Comparison) {
                updateMaskPreviewIcons();
                m_framesCanvas->canvas()->setMaskOutline(*mask, QColor(200, 0, 200));
            } else {
                cv::Mat outlineMask = buildDynamicMaskFromMap(*mask, dynamicSetId);
                updateDynamicMaskPreviewIcons();
                m_framesCanvas->canvas()->setMaskOutline(outlineMask, QColor(255, 200, 0));
            }
            updateMaskPreviewForFrame(index);
            return;
        }
        if (!m_frameHasStart) {
            return;
        }
        cv::Mat previewMask;
        cv::Mat previewMap = mask->clone();
        const bool erase = buttons.testFlag(Qt::RightButton) ||
            (m_maskMode == MaskMode::Dynamic && modifiers.testFlag(Qt::ShiftModifier)) ||
            (m_maskMode == MaskMode::Comparison && modifiers.testFlag(Qt::ShiftModifier));
        if (m_maskMode == MaskMode::Dynamic) {
            applyToolToDynamicMask(previewMap, dynamicSetId, m_drawTool, m_frameStart, QPoint(x, y), erase);
            previewMask = buildDynamicMaskFromMap(previewMap, dynamicSetId);
        } else {
            previewMask = previewMap;
            applyToolToMask(previewMask, m_drawTool, m_frameStart, QPoint(x, y), erase);
        }
        if (const cv::Mat* frame = activeFrameImage(index, false)) {
            const QColor gapColorQt = m_framesCanvas->palette().color(QPalette::Window);
            const cv::Scalar gapColor(gapColorQt.blue(), gapColorQt.green(), gapColorQt.red());
            cv::Mat base = renderFrameWithSerum(index, m_useHdFrame);
            cv::Mat preview = buildCombinedMaskPreview(base, reference, previewMask, color, gapColor);
            m_framesCanvas->canvas()->setPreviewImage(preview);
            QRect outlineRegion;
            if (m_showOriginalFrame) {
                outlineRegion = QRect(layout.bottomX,
                                      layout.topHeight + gap,
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
    if (!isFrame && m_spriteDetAreaMode) {
        const int index = m_spritesList ? m_spritesList->currentRow() : -1;
        if (index < 0) {
            return;
        }
        if (!m_spriteHasStart) {
            return;
        }
        const cv::Mat* spriteImage = m_spriteStore ? m_spriteStore->at(index) : nullptr;
        if (!spriteImage || spriteImage->empty()) {
            return;
        }
        const QRect contentRect = spriteContentRect(index);
        const int offsetX = contentRect.isValid() ? contentRect.x() : 0;
        const int offsetY = contentRect.isValid() ? contentRect.y() : 0;
        const int baseWidth = contentRect.isValid() ? contentRect.width() : spriteImage->cols;
        const int baseHeight = contentRect.isValid() ? contentRect.height() : spriteImage->rows;
        cv::Mat originalRef;
        if (const cv::Mat* originalSource = spriteOriginalForDisplay(index)) {
            originalRef = (*originalSource)(cv::Rect(offsetX, offsetY, baseWidth, baseHeight));
        }
        if (originalRef.empty()) {
            return;
        }
        cv::Mat cleaned = originalRef.clone();
        for (int yy = 0; yy < cleaned.rows; ++yy) {
            uint8_t* row = cleaned.ptr<uint8_t>(yy);
            for (int xx = 0; xx < cleaned.cols; ++xx) {
                if (row[xx] == 255) {
                    row[xx] = 0;
                }
            }
        }
        cv::Mat original = buildOriginalFrame(cleaned);
        cv::Mat displayOriginal = BuildDisplayOriginal(original, cv::Size(baseWidth, baseHeight));
        FrameLayout layout = BuildFrameLayout(baseWidth, baseHeight,
                                              displayOriginal.cols, displayOriginal.rows);
        const int gap = FrameGapForWidth(layout.topWidth);
        if (y < layout.topHeight + gap || y >= layout.topHeight + gap + layout.bottomHeight ||
            x < layout.bottomX || x >= layout.bottomX + layout.bottomWidth) {
            return;
        }
        const int localX = x - layout.bottomX;
        const int localY = y - (layout.topHeight + gap);
        const int scaleX = (layout.bottomWidth > 0 && original.cols > 0) ? layout.bottomWidth / original.cols : 1;
        const int scaleY = (layout.bottomHeight > 0 && original.rows > 0) ? layout.bottomHeight / original.rows : 1;
        const int mappedX = (scaleX > 1) ? localX / scaleX : localX;
        const int mappedY = (scaleY > 1) ? localY / scaleY : localY;
        const int maxX = original.cols - 1;
        const int maxY = original.rows - 1;
        const int x0 = std::clamp(m_spriteStart.x(), 0, maxX);
        const int y0 = std::clamp(m_spriteStart.y(), 0, maxY);
        const int x1 = std::clamp(mappedX, 0, maxX);
        const int y1 = std::clamp(mappedY, 0, maxY);
        const int left = std::min(x0, x1);
        const int top = std::min(y0, y1);
        const int right = std::max(x0, x1);
        const int bottom = std::max(y0, y1);
        const cv::Rect rect(left, top, right - left + 1, bottom - top + 1);
        cv::Mat previewMask(original.size(), CV_8UC1, cv::Scalar(0));
        cv::rectangle(previewMask, rect, cv::Scalar(1), cv::FILLED);
        cv::Mat otherMask(original.size(), CV_8UC1, cv::Scalar(0));
        const std::size_t base = static_cast<std::size_t>(index) * MAX_SPRITE_DETECT_AREAS * 4;
        for (int area = 0; area < MAX_SPRITE_DETECT_AREAS; ++area) {
            if (area == m_spriteDetAreaIndex) {
                continue;
            }
            const std::size_t areaBase = base + static_cast<std::size_t>(area) * 4;
            if (areaBase + 3 >= m_spriteDetAreas.size()) {
                continue;
            }
            const uint16_t leftVal = m_spriteDetAreas[areaBase];
            if (leftVal == 0xffff) {
                continue;
            }
            const int fullX = static_cast<int>(leftVal);
            const int fullY = static_cast<int>(m_spriteDetAreas[areaBase + 1]);
            const int aw = static_cast<int>(m_spriteDetAreas[areaBase + 2]);
            const int ah = static_cast<int>(m_spriteDetAreas[areaBase + 3]);
            cv::Rect rect(fullX - offsetX, fullY - offsetY, aw, ah);
            rect &= cv::Rect(0, 0, original.cols, original.rows);
            if (rect.width <= 0 || rect.height <= 0) {
                continue;
            }
            cv::rectangle(otherMask, rect, cv::Scalar(1), cv::FILLED);
        }
        QRect region(layout.bottomX,
                     layout.topHeight + gap,
                     layout.bottomWidth,
                     layout.bottomHeight);
        m_spritesCanvas->canvas()->setMaskOutline(previewMask, QColor(255, 200, 0), region);
        if (MaskHasContent(otherMask)) {
            m_spritesCanvas->canvas()->setSecondaryMaskOutline(otherMask, QColor(0, 200, 255), region);
        } else {
            m_spritesCanvas->canvas()->setSecondaryMaskOutline(cv::Mat(), QColor());
        }
        return;
    }
    if (!isFrame && m_spriteDynamicMaskMode) {
        const int index = m_spritesList ? m_spritesList->currentRow() : -1;
        if (index < 0) {
            return;
        }
        ensureSpriteDataSize();
        cv::Mat* map = activeSpriteDynamicMask(index);
        if (!map || map->empty()) {
            return;
        }
        const cv::Mat* spriteImage = activeSpriteImage(index);
        const QRect contentRect = spriteContentRect(index);
        const QRect displayRect = spriteImage ? spriteDisplayRect(index, *spriteImage) : QRect();
        const int offsetX = displayRect.isValid() ? displayRect.x() : 0;
        const int offsetY = displayRect.isValid() ? displayRect.y() : 0;
        const QSize baseSize = displayRect.isValid()
            ? QSize(displayRect.width(), displayRect.height())
            : QSize(spriteImage && !spriteImage->empty() ? spriteImage->cols : map->cols,
                    spriteImage && !spriteImage->empty() ? spriteImage->rows : map->rows);
        cv::Mat originalRef;
        if (const cv::Mat* originalSource = spriteOriginalForDisplay(index)) {
            originalRef = contentRect.isValid()
                ? (*originalSource)(cv::Rect(contentRect.x(), contentRect.y(), contentRect.width(), contentRect.height()))
                : *originalSource;
        }
        cv::Mat original = originalRef.empty() ? cv::Mat() : buildOriginalFrame(originalRef);
        cv::Mat displayOriginal = BuildDisplayOriginal(original, cv::Size(baseSize.width(), baseSize.height()));
        FrameLayout layout = BuildFrameLayout(baseSize.width(), baseSize.height(),
                                              displayOriginal.cols, displayOriginal.rows);
        if (x < layout.topX || x >= layout.topX + layout.topWidth ||
            y < 0 || y >= layout.topHeight) {
            return;
        }
        const QPoint mappedLocal = ScalePointToSize(QPoint(x - layout.topX, y),
                                                    QSize(layout.topWidth, layout.topHeight),
                                                    QSize(baseSize.width(), baseSize.height()));
        const QPoint mapped(mappedLocal.x() + offsetX, mappedLocal.y() + offsetY);
        if (mapped.x() < 0 || mapped.y() < 0 || mapped.x() >= map->cols || mapped.y() >= map->rows) {
            return;
        }
        const cv::Mat* spriteMask = spriteOriginalForDisplay(index);
        cv::Mat maskScaled;
        if (spriteMask && !spriteMask->empty()) {
            if (spriteMask->size() != map->size()) {
                cv::resize(*spriteMask, maskScaled, map->size(), 0.0, 0.0, cv::INTER_NEAREST);
            } else {
                maskScaled = *spriteMask;
            }
            if (maskScaled.at<uint8_t>(mapped.y(), mapped.x()) == 255) {
                return;
            }
        }
        if (m_drawTool == DrawTool::Point) {
            const bool erase = buttons.testFlag(Qt::RightButton) || modifiers.testFlag(Qt::ShiftModifier);
            applyToolToDynamicMask(*map, m_spriteDynamicSetIndex, DrawTool::Point, mapped, mapped, erase);
            if (!maskScaled.empty()) {
                ApplySpriteMaskConstraints(*map, maskScaled);
            }
            updateSpriteCanvasImage(index);
            return;
        }
        if (!m_spriteHasStart) {
            return;
        }
        cv::Mat previewMap = map->clone();
        const bool erase = buttons.testFlag(Qt::RightButton) || modifiers.testFlag(Qt::ShiftModifier);
        applyToolToDynamicMask(previewMap, m_spriteDynamicSetIndex, m_drawTool, m_spriteStart, mapped, erase);
        if (!maskScaled.empty()) {
            ApplySpriteMaskConstraints(previewMap, maskScaled);
        }
        if (spriteImage && !spriteImage->empty()) {
            cv::Mat baseFull = applySpriteDynamicColors(index, *spriteImage);
            cv::Mat previewMaskFull = buildDynamicMaskFromMap(previewMap, m_spriteDynamicSetIndex);
            cv::Rect roi(offsetX, offsetY, baseSize.width(), baseSize.height());
            cv::Mat base = baseFull(roi).clone();
            cv::Mat previewMask = previewMaskFull(roi).clone();
            cv::Mat preview = buildMaskPreview(base, previewMask, cv::Vec3b(0, 200, 255));
            cv::Mat originalRef;
            if (const cv::Mat* originalSource = spriteOriginalForDisplay(index)) {
                originalRef = contentRect.isValid()
                    ? (*originalSource)(cv::Rect(contentRect.x(), contentRect.y(), contentRect.width(), contentRect.height())).clone()
                    : originalSource->clone();
            }
            cv::Mat original;
            if (!originalRef.empty()) {
                cv::Mat cleaned = originalRef.clone();
                for (int yy = 0; yy < cleaned.rows; ++yy) {
                    uint8_t* row = cleaned.ptr<uint8_t>(yy);
                    for (int xx = 0; xx < cleaned.cols; ++xx) {
                        if (row[xx] == 255) {
                            row[xx] = 0;
                        }
                    }
                }
                original = buildOriginalFrame(cleaned);
            }
            if (!original.empty()) {
                cv::Mat displayOriginal = BuildDisplayOriginal(original, preview.size());
                const QColor gap = m_spritesCanvas->palette().color(QPalette::Window);
                cv::Mat combined = buildCombinedFrame(preview,
                                                      displayOriginal,
                                                      cv::Scalar(gap.blue(), gap.green(), gap.red()));
                m_spritesCanvas->canvas()->setPreviewImage(combined);
            } else {
                m_spritesCanvas->canvas()->setPreviewImage(preview);
            }
        }
        return;
    }
    cv::Mat* image = isFrame ? activeFrameImage(m_framesList->currentRow(), true)
                             : activeSpriteImageMutable(m_spritesList ? m_spritesList->currentRow() : -1);
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
    } else {
        const int index = m_spritesList ? m_spritesList->currentRow() : -1;
        const QRect contentRect = spriteContentRect(index);
        const QRect displayRect = spriteDisplayRect(index, *image);
        const int offsetX = displayRect.isValid() ? displayRect.x() : 0;
        const int offsetY = displayRect.isValid() ? displayRect.y() : 0;
        const int baseWidth = displayRect.isValid() ? displayRect.width() : image->cols;
        const int baseHeight = displayRect.isValid() ? displayRect.height() : image->rows;
        cv::Mat originalRef;
        if (const cv::Mat* originalSource = spriteOriginalForDisplay(index)) {
            originalRef = contentRect.isValid()
                ? (*originalSource)(cv::Rect(contentRect.x(), contentRect.y(),
                                             contentRect.width(), contentRect.height()))
                : *originalSource;
        }
        cv::Mat original = originalRef.empty() ? cv::Mat() : buildOriginalFrame(originalRef);
        cv::Mat displayOriginal = BuildDisplayOriginal(original, cv::Size(baseWidth, baseHeight));
        FrameLayout layout = BuildFrameLayout(baseWidth, baseHeight,
                                              displayOriginal.cols, displayOriginal.rows);
        if (y < 0 || y >= layout.topHeight ||
            x < layout.topX || x >= layout.topX + layout.topWidth) {
            return;
        }
        x -= layout.topX;
        x += offsetX;
        y += offsetY;
    }
    if (m_drawTool == DrawTool::Point) {
        const bool erase = buttons.testFlag(Qt::RightButton);
        if (isFrame) {
            const QPoint basePoint(x, y);
            const QSize baseSize(image->cols, image->rows);
            const std::vector<int> targets = targetFrameIndices();
            for (int frameIndex : targets) {
                cv::Mat* target = activeFrameImage(frameIndex, true);
                if (!target || target->empty()) {
                    continue;
                }
                const QPoint mapped = ScalePointToSize(basePoint, baseSize,
                                                       QSize(target->cols, target->rows));
                const cv::Mat spriteMask = buildSpriteCoverageMask(frameIndex, m_useHdFrame);
                if (!spriteMask.empty() &&
                    mapped.y() >= 0 && mapped.y() < spriteMask.rows &&
                    mapped.x() >= 0 && mapped.x() < spriteMask.cols &&
                    spriteMask.at<uint8_t>(mapped.y(), mapped.x())) {
                    continue;
                }
                applyToolToImage(*target, DrawTool::Point, mapped, mapped, erase);
                updateFramePreviewAt(frameIndex);
            }
            updateFrameCanvasImage(m_framesList->currentRow());
            updateMaskPreviewForFrame(m_framesList->currentRow());
        } else {
            applyToolToImage(*image, DrawTool::Point, QPoint(x, y), QPoint(x, y), erase);
            updateSpriteCanvasImage(m_spritesList ? m_spritesList->currentRow() : -1);
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
        const int frameIndex = m_framesList->currentRow();
        const cv::Mat spriteMask = buildSpriteCoverageMask(frameIndex, m_useHdFrame);
        if (!spriteMask.empty()) {
            restoreSpriteCoverage(preview, *image, spriteMask);
        }
        cv::Mat reference = buildOriginalPreviewForIndex(m_framesList->currentRow());
        cv::Mat original = m_showOriginalFrame ? buildOriginalFrame(reference) : cv::Mat();
        cv::Mat composed = renderFrameWithSerum(frameIndex, m_useHdFrame, preview);
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
        const int spriteIndex = m_spritesList ? m_spritesList->currentRow() : -1;
        const QRect contentRect = spriteContentRect(spriteIndex);
        const QRect displayRect = spriteDisplayRect(spriteIndex, preview);
        const int offsetX = displayRect.isValid() ? displayRect.x() : 0;
        const int offsetY = displayRect.isValid() ? displayRect.y() : 0;
        const int baseWidth = displayRect.isValid() ? displayRect.width() : preview.cols;
        const int baseHeight = displayRect.isValid() ? displayRect.height() : preview.rows;
        cv::Mat previewDisplay = applySpriteDynamicColors(spriteIndex, preview);
        if (displayRect.isValid()) {
            previewDisplay = previewDisplay(cv::Rect(offsetX, offsetY, baseWidth, baseHeight)).clone();
        }
        cv::Mat originalRef;
        if (const cv::Mat* originalSource = spriteOriginalForDisplay(spriteIndex)) {
            originalRef = contentRect.isValid()
                ? (*originalSource)(cv::Rect(contentRect.x(), contentRect.y(),
                                             contentRect.width(), contentRect.height()))
                : *originalSource;
        }
        cv::Mat original;
        if (!originalRef.empty()) {
            cv::Mat cleaned = originalRef.clone();
            for (int yy = 0; yy < cleaned.rows; ++yy) {
                uint8_t* row = cleaned.ptr<uint8_t>(yy);
                for (int xx = 0; xx < cleaned.cols; ++xx) {
                    if (row[xx] == 255) {
                        row[xx] = 0;
                    }
                }
            }
            original = buildOriginalFrame(cleaned);
        }
        if (!original.empty()) {
            cv::Mat displayOriginal = BuildDisplayOriginal(original, previewDisplay.size());
            const QColor gap = m_spritesCanvas->palette().color(QPalette::Window);
            cv::Mat combined = buildCombinedFrame(previewDisplay,
                                                  displayOriginal,
                                                  cv::Scalar(gap.blue(), gap.green(), gap.red()));
            m_spritesCanvas->canvas()->setPreviewImage(combined);
        } else {
            m_spritesCanvas->canvas()->setPreviewImage(previewDisplay);
        }
    }
}

void MainWindow::handleToolRelease(bool isFrame,
                                   int x,
                                   int y,
                                   Qt::MouseButton button,
                                   Qt::KeyboardModifiers modifiers)
{
    if (!m_drawPointEnabled) {
        return;
    }
    if (isFrame && m_frameDrawOnZone) {
        if (!m_frameHasStart) {
            return;
        }
        const int index = m_framesList ? m_framesList->currentRow() : -1;
        if (index < 0) {
            m_frameHasStart = false;
            m_frameDrawOnZone = false;
            return;
        }
        ensureMaskDataSize();
        const cv::Mat* topFrame = activeFrameImage(index, false);
        if (!topFrame || topFrame->empty()) {
            m_frameHasStart = false;
            m_frameDrawOnZone = false;
            return;
        }
        cv::Mat reference = buildOriginalPreviewForIndex(index);
        FrameLayout layout = BuildFrameLayout(*topFrame, reference);
        const int gap = FrameGapForWidth(layout.topWidth);
        const bool useBottom = m_showOriginalFrame && !reference.empty();
        int localX = x;
        int localY = y;
        if (useBottom) {
            if (y < layout.topHeight + gap || y >= layout.topHeight + gap + layout.bottomHeight ||
                x < layout.bottomX || x >= layout.bottomX + layout.bottomWidth) {
                m_frameHasStart = false;
                m_frameDrawOnZone = false;
                return;
            }
            localX = x - layout.bottomX;
            localY = y - (layout.topHeight + gap);
        } else {
            if (y < 0 || y >= layout.topHeight ||
                x < layout.topX || x >= layout.topX + layout.topWidth) {
                m_frameHasStart = false;
                m_frameDrawOnZone = false;
                return;
            }
            localX = x - layout.topX;
            localY = y;
        }
        const int baseWidth = reference.empty() ? topFrame->cols : reference.cols;
        const int baseHeight = reference.empty() ? topFrame->rows : reference.rows;
        const int scaleX = (useBottom && baseWidth > 0) ? layout.bottomWidth / baseWidth : 1;
        const int scaleY = (useBottom && baseHeight > 0) ? layout.bottomHeight / baseHeight : 1;
        const int mappedX = (scaleX > 1) ? localX / scaleX : localX;
        const int mappedY = (scaleY > 1) ? localY / scaleY : localY;
        if (mappedX < 0 || mappedY < 0 || mappedX >= baseWidth || mappedY >= baseHeight) {
            m_frameHasStart = false;
            m_frameDrawOnZone = false;
            return;
        }
        const QPoint baseStart = m_frameStart;
        const QPoint baseEnd(mappedX, mappedY);
        const QSize baseSize(baseWidth, baseHeight);
        if (m_selectedSpriteZoneIndex < 0 ||
            m_selectedSpriteZoneIndex >= static_cast<int>(m_spriteZones.size())) {
            statusBar()->showMessage("Select a sprite zone before editing.", 2000);
            m_frameHasStart = false;
            m_frameDrawOnZone = false;
            return;
        }
        const SpriteZoneGroup zone = m_spriteZones[static_cast<std::size_t>(m_selectedSpriteZoneIndex)];
        if (zone.slotIndices.empty()) {
            m_frameHasStart = false;
            m_frameDrawOnZone = false;
            return;
        }
        const std::vector<int> targets = targetFrameIndices();
        for (int frameIndex : targets) {
            if (frameIndex < 0) {
                continue;
            }
            cv::Mat targetReference = buildOriginalPreviewForIndex(frameIndex);
            const QSize targetSize = targetReference.empty()
                ? baseSize
                : QSize(targetReference.cols, targetReference.rows);
            const QPoint mappedStart = ScalePointToSize(baseStart, baseSize, targetSize);
            const QPoint mappedEnd = ScalePointToSize(baseEnd, baseSize, targetSize);
            const int left = std::min(mappedStart.x(), mappedEnd.x());
            const int top = std::min(mappedStart.y(), mappedEnd.y());
            const int right = std::max(mappedStart.x(), mappedEnd.x());
            const int bottom = std::max(mappedStart.y(), mappedEnd.y());
            for (int slot : zone.slotIndices) {
                const std::size_t bboxIndex = static_cast<std::size_t>(frameIndex) * MAX_SPRITES_PER_FRAME * 4 +
                    static_cast<std::size_t>(slot) * 4;
                if (bboxIndex + 3 >= m_frameSpriteBBoxes.size()) {
                    continue;
                }
                m_frameSpriteBBoxes[bboxIndex] = static_cast<uint16_t>(std::clamp(left, 0, targetSize.width() - 1));
                m_frameSpriteBBoxes[bboxIndex + 1] = static_cast<uint16_t>(std::clamp(top, 0, targetSize.height() - 1));
                m_frameSpriteBBoxes[bboxIndex + 2] = static_cast<uint16_t>(std::clamp(right, 0, targetSize.width() - 1));
                m_frameSpriteBBoxes[bboxIndex + 3] = static_cast<uint16_t>(std::clamp(bottom, 0, targetSize.height() - 1));
            }
            updateFramePreviewAt(frameIndex);
        }
        m_frameHasStart = false;
        m_frameDrawOnZone = false;
        refreshSpriteZoneList();
        refreshFrameSpriteSlotCombo();
        updateMaskPreviewForFrame(index);
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
        const bool erase = (m_frameStartButton == Qt::RightButton || button == Qt::RightButton ||
                            modifiers.testFlag(Qt::ShiftModifier));
        const QPoint baseStart = m_frameStart;
        const QPoint baseEnd(x, y);
        const QSize baseSize(mask->cols, mask->rows);
        const std::vector<int> targets = targetFrameIndices();
        for (int frameIndex : targets) {
            cv::Mat* targetMask = activeBackgroundMask(frameIndex);
            if (!targetMask || targetMask->empty()) {
                continue;
            }
            const QPoint mappedStart = ScalePointToSize(baseStart, baseSize,
                                                        QSize(targetMask->cols, targetMask->rows));
            const QPoint mappedEnd = ScalePointToSize(baseEnd, baseSize,
                                                      QSize(targetMask->cols, targetMask->rows));
            applyToolToMask(*targetMask, m_drawTool, mappedStart, mappedEnd, erase);
            updateFramePreviewAt(frameIndex);
        }
        m_frameHasStart = false;
        m_framesCanvas->canvas()->clearPreviewImage();
        updateFrameCanvasImage(index);
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
        const int gap = FrameGapForWidth(layout.topWidth);
        if (y < layout.topHeight + gap || y >= layout.topHeight + gap + layout.bottomHeight ||
            x < layout.bottomX || x >= layout.bottomX + layout.bottomWidth) {
            m_frameHasStart = false;
            return;
        }
        x -= layout.bottomX;
        y -= (layout.topHeight + gap);
        cv::Mat* mask = nullptr;
        int dynamicSetId = -1;
        if (m_maskMode == MaskMode::Comparison) {
            const int assigned = currentFrameMaskId();
            const int selected = m_maskList->currentRow();
            if (assigned < 0 || assigned != selected) {
                m_frameHasStart = false;
                return;
            }
            mask = activeComparisonMask();
        } else if (m_maskMode == MaskMode::Dynamic) {
            dynamicSetId = currentFrameDynamicMaskId();
            if (dynamicSetId < 0) {
                m_frameHasStart = false;
                return;
            }
            mask = activeDynamicMaskMap(index);
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
        const bool erase = (m_frameStartButton == Qt::RightButton || button == Qt::RightButton ||
                            (m_maskMode == MaskMode::Dynamic && modifiers.testFlag(Qt::ShiftModifier)) ||
                            (m_maskMode == MaskMode::Comparison && modifiers.testFlag(Qt::ShiftModifier)));
        const QPoint baseStart = m_frameStart;
        const QPoint baseEnd(x, y);
        const QSize baseSize(mask->cols, mask->rows);
        if (m_maskMode == MaskMode::Dynamic) {
            const std::vector<int> targets = targetFrameIndices();
            for (int frameIndex : targets) {
                cv::Mat* targetMask = activeDynamicMaskMap(frameIndex);
                if (!targetMask || targetMask->empty()) {
                    continue;
                }
                const QPoint mappedStart = ScalePointToSize(baseStart, baseSize,
                                                            QSize(targetMask->cols, targetMask->rows));
                const QPoint mappedEnd = ScalePointToSize(baseEnd, baseSize,
                                                          QSize(targetMask->cols, targetMask->rows));
                applyToolToDynamicMask(*targetMask, dynamicSetId, m_drawTool, mappedStart, mappedEnd, erase);
                updateFramePreviewAt(frameIndex);
            }
        } else {
            applyToolToMask(*mask, m_drawTool, baseStart, baseEnd, erase);
            updatePreviewsForMaskId(m_maskList ? m_maskList->currentRow() : -1);
        }
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
    if (!isFrame && m_spriteDetAreaMode) {
        if (m_drawTool == DrawTool::ColorPicker) {
            m_spriteUndoActive = false;
            return;
        }
        if (!m_spriteHasStart) {
            return;
        }
        const int index = m_spritesList ? m_spritesList->currentRow() : -1;
        if (index < 0) {
            m_spriteHasStart = false;
            return;
        }
        ensureSpriteDataSize();
        const cv::Mat* spriteImage = m_spriteStore ? m_spriteStore->at(index) : nullptr;
        if (!spriteImage || spriteImage->empty()) {
            m_spriteHasStart = false;
            return;
        }
        const QRect contentRect = spriteContentRect(index);
        const int offsetX = contentRect.isValid() ? contentRect.x() : 0;
        const int offsetY = contentRect.isValid() ? contentRect.y() : 0;
        const int baseWidth = contentRect.isValid() ? contentRect.width() : spriteImage->cols;
        const int baseHeight = contentRect.isValid() ? contentRect.height() : spriteImage->rows;
        cv::Mat originalRef;
        if (const cv::Mat* originalSource = spriteOriginalForDisplay(index)) {
            originalRef = (*originalSource)(cv::Rect(offsetX, offsetY, baseWidth, baseHeight));
        }
        if (originalRef.empty()) {
            m_spriteHasStart = false;
            return;
        }
        cv::Mat cleaned = originalRef.clone();
        for (int yy = 0; yy < cleaned.rows; ++yy) {
            uint8_t* row = cleaned.ptr<uint8_t>(yy);
            for (int xx = 0; xx < cleaned.cols; ++xx) {
                if (row[xx] == 255) {
                    row[xx] = 0;
                }
            }
        }
        cv::Mat original = buildOriginalFrame(cleaned);
        cv::Mat displayOriginal = BuildDisplayOriginal(original, cv::Size(baseWidth, baseHeight));
        FrameLayout layout = BuildFrameLayout(baseWidth, baseHeight,
                                              displayOriginal.cols, displayOriginal.rows);
        const int gap = FrameGapForWidth(layout.topWidth);
        if (y < layout.topHeight + gap || y >= layout.topHeight + gap + layout.bottomHeight ||
            x < layout.bottomX || x >= layout.bottomX + layout.bottomWidth) {
            m_spriteHasStart = false;
            return;
        }
        const int localX = x - layout.bottomX;
        const int localY = y - (layout.topHeight + gap);
        const int scaleX = (layout.bottomWidth > 0 && original.cols > 0) ? layout.bottomWidth / original.cols : 1;
        const int scaleY = (layout.bottomHeight > 0 && original.rows > 0) ? layout.bottomHeight / original.rows : 1;
        const int mappedX = (scaleX > 1) ? localX / scaleX : localX;
        const int mappedY = (scaleY > 1) ? localY / scaleY : localY;
        const int maxX = original.cols - 1;
        const int maxY = original.rows - 1;
        const int x0 = std::clamp(m_spriteStart.x(), 0, maxX);
        const int y0 = std::clamp(m_spriteStart.y(), 0, maxY);
        const int x1 = std::clamp(mappedX, 0, maxX);
        const int y1 = std::clamp(mappedY, 0, maxY);
        const int left = std::min(x0, x1);
        const int top = std::min(y0, y1);
        const int right = std::max(x0, x1);
        const int bottom = std::max(y0, y1);
        const int width = right - left + 1;
        const int height = bottom - top + 1;
        const int fullLeft = left + offsetX;
        const int fullTop = top + offsetY;
        const cv::Mat* originalSource = spriteOriginalForDisplay(index);
        const int fullMaxX = originalSource ? originalSource->cols - 1 : fullLeft + width - 1;
        const int fullMaxY = originalSource ? originalSource->rows - 1 : fullTop + height - 1;
        const int clippedLeft = std::clamp(fullLeft, 0, fullMaxX);
        const int clippedTop = std::clamp(fullTop, 0, fullMaxY);
        const int clippedWidth = std::clamp(width, 1, fullMaxX - clippedLeft + 1);
        const int clippedHeight = std::clamp(height, 1, fullMaxY - clippedTop + 1);
        const std::size_t base = static_cast<std::size_t>(index) * MAX_SPRITE_DETECT_AREAS * 4 +
            static_cast<std::size_t>(m_spriteDetAreaIndex) * 4;
        if (base + 3 < m_spriteDetAreas.size()) {
            m_spriteDetAreas[base] = static_cast<uint16_t>(clippedLeft);
            m_spriteDetAreas[base + 1] = static_cast<uint16_t>(clippedTop);
            m_spriteDetAreas[base + 2] = static_cast<uint16_t>(clippedWidth);
            m_spriteDetAreas[base + 3] = static_cast<uint16_t>(clippedHeight);
        }
        m_spriteHasStart = false;
        m_spriteUndoActive = false;
        updateSpriteCanvasImage(index);
        return;
    }
    if (!isFrame && m_spriteDynamicMaskMode) {
        if (m_drawTool == DrawTool::Point ||
            m_drawTool == DrawTool::ColorPicker ||
            m_drawTool == DrawTool::MagicFill) {
            m_spriteUndoActive = false;
            return;
        }
        if (!m_spriteHasStart) {
            return;
        }
        const int index = m_spritesList ? m_spritesList->currentRow() : -1;
        if (index < 0) {
            m_spriteHasStart = false;
            return;
        }
        ensureSpriteDataSize();
        cv::Mat* map = activeSpriteDynamicMask(index);
        if (!map || map->empty()) {
            m_spriteHasStart = false;
            return;
        }
        const cv::Mat* spriteImage = activeSpriteImage(index);
        const QRect contentRect = spriteContentRect(index);
        const QRect displayRect = spriteImage ? spriteDisplayRect(index, *spriteImage) : QRect();
        const int offsetX = displayRect.isValid() ? displayRect.x() : 0;
        const int offsetY = displayRect.isValid() ? displayRect.y() : 0;
        const QSize baseSize = displayRect.isValid()
            ? QSize(displayRect.width(), displayRect.height())
            : QSize(spriteImage && !spriteImage->empty() ? spriteImage->cols : map->cols,
                    spriteImage && !spriteImage->empty() ? spriteImage->rows : map->rows);
        cv::Mat originalRef;
        if (const cv::Mat* originalSource = spriteOriginalForDisplay(index)) {
            originalRef = contentRect.isValid()
                ? (*originalSource)(cv::Rect(contentRect.x(), contentRect.y(), contentRect.width(), contentRect.height()))
                : *originalSource;
        }
        cv::Mat original = originalRef.empty() ? cv::Mat() : buildOriginalFrame(originalRef);
        cv::Mat displayOriginal = BuildDisplayOriginal(original, cv::Size(baseSize.width(), baseSize.height()));
        FrameLayout layout = BuildFrameLayout(baseSize.width(), baseSize.height(),
                                              displayOriginal.cols, displayOriginal.rows);
        if (x < layout.topX || x >= layout.topX + layout.topWidth ||
            y < 0 || y >= layout.topHeight) {
            m_spriteHasStart = false;
            return;
        }
        const QPoint mappedLocal = ScalePointToSize(QPoint(x - layout.topX, y),
                                                    QSize(layout.topWidth, layout.topHeight),
                                                    QSize(baseSize.width(), baseSize.height()));
        const QPoint mapped(mappedLocal.x() + offsetX, mappedLocal.y() + offsetY);
        if (mapped.x() < 0 || mapped.y() < 0 || mapped.x() >= map->cols || mapped.y() >= map->rows) {
            m_spriteHasStart = false;
            return;
        }
        const cv::Mat* spriteMask = spriteOriginalForDisplay(index);
        cv::Mat maskScaled;
        if (spriteMask && !spriteMask->empty()) {
            if (spriteMask->size() != map->size()) {
                cv::resize(*spriteMask, maskScaled, map->size(), 0.0, 0.0, cv::INTER_NEAREST);
            } else {
                maskScaled = *spriteMask;
            }
        }
        const bool erase = (m_spriteStartButton == Qt::RightButton || button == Qt::RightButton ||
                            modifiers.testFlag(Qt::ShiftModifier));
        applyToolToDynamicMask(*map, m_spriteDynamicSetIndex, m_drawTool, m_spriteStart, mapped, erase);
        if (!maskScaled.empty()) {
            ApplySpriteMaskConstraints(*map, maskScaled);
        }
        m_spriteHasStart = false;
        m_spritesCanvas->canvas()->clearPreviewImage();
        updateSpriteCanvasImage(index);
        m_spriteUndoActive = false;
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
                             : activeSpriteImageMutable(m_spritesList ? m_spritesList->currentRow() : -1);
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
    } else {
        const int index = m_spritesList ? m_spritesList->currentRow() : -1;
        const QRect contentRect = spriteContentRect(index);
        const QRect displayRect = spriteDisplayRect(index, *image);
        const int offsetX = displayRect.isValid() ? displayRect.x() : 0;
        const int offsetY = displayRect.isValid() ? displayRect.y() : 0;
        const int baseWidth = displayRect.isValid() ? displayRect.width() : image->cols;
        const int baseHeight = displayRect.isValid() ? displayRect.height() : image->rows;
        cv::Mat originalRef;
        if (const cv::Mat* originalSource = spriteOriginalForDisplay(index)) {
            originalRef = contentRect.isValid()
                ? (*originalSource)(cv::Rect(contentRect.x(), contentRect.y(),
                                             contentRect.width(), contentRect.height()))
                : *originalSource;
        }
        cv::Mat original = originalRef.empty() ? cv::Mat() : buildOriginalFrame(originalRef);
        cv::Mat displayOriginal = BuildDisplayOriginal(original, cv::Size(baseWidth, baseHeight));
        FrameLayout layout = BuildFrameLayout(baseWidth, baseHeight,
                                              displayOriginal.cols, displayOriginal.rows);
        if (y < 0 || y >= layout.topHeight ||
            x < layout.topX || x >= layout.topX + layout.topWidth) {
            m_spriteHasStart = false;
            return;
        }
        x -= layout.topX;
        x += offsetX;
        y += offsetY;
    }
    const bool erase = (startButton == Qt::RightButton || button == Qt::RightButton);
    if (isFrame) {
        const QPoint baseStart = start;
        const QPoint baseEnd(x, y);
        const QSize baseSize(image->cols, image->rows);
        const std::vector<int> targets = targetFrameIndices();
        for (int frameIndex : targets) {
            cv::Mat* target = activeFrameImage(frameIndex, true);
            if (!target || target->empty()) {
                continue;
            }
            cv::Mat backup = target->clone();
            const QPoint mappedStart = ScalePointToSize(baseStart, baseSize,
                                                        QSize(target->cols, target->rows));
            const QPoint mappedEnd = ScalePointToSize(baseEnd, baseSize,
                                                      QSize(target->cols, target->rows));
            applyToolToImage(*target, m_drawTool, mappedStart, mappedEnd, erase);
            const cv::Mat spriteMask = buildSpriteCoverageMask(frameIndex, m_useHdFrame);
            if (!spriteMask.empty()) {
                restoreSpriteCoverage(*target, backup, spriteMask);
            }
            commitFrameFromStore(frameIndex, m_useHdFrame);
            updateFramePreviewAt(frameIndex);
        }
        if (targets.size() == 1) {
            const int editedIndex = targets.front();
            const cv::Mat* edited = activeFrameImage(editedIndex, false);
            if (edited && !edited->empty()) {
                m_frameCanvasOverrideIndex = editedIndex;
                m_frameCanvasOverrideImage = edited->clone();
                int nonzero = 0;
                for (int y = 0; y < m_frameCanvasOverrideImage.rows; ++y) {
                    const cv::Vec3b* row = m_frameCanvasOverrideImage.ptr<cv::Vec3b>(y);
                    for (int x = 0; x < m_frameCanvasOverrideImage.cols; ++x) {
                        const cv::Vec3b px = row[x];
                        if (px[0] || px[1] || px[2]) {
                            ++nonzero;
                        }
                    }
                }
                logLine(QString("Draw commit: override stats index=%1 size=%2x%3 nonzero=%4")
                            .arg(editedIndex)
                            .arg(m_frameCanvasOverrideImage.cols)
                            .arg(m_frameCanvasOverrideImage.rows)
                            .arg(nonzero));
            }
        }
        m_framesCanvas->canvas()->clearPreviewImage();
        updateFrameCanvasImage(m_framesList->currentRow());
        updateMaskPreviewForFrame(m_framesList->currentRow());
        logLine(QString("Draw commit: frame index=%1 tool=%2 targets=%3 hd=%4")
                    .arg(m_framesList->currentRow())
                    .arg(static_cast<int>(m_drawTool))
                    .arg(targets.size())
                    .arg(m_useHdFrame ? "true" : "false"));
        m_frameUndoActive = false;
    } else {
        applyToolToImage(*image, m_drawTool, start, QPoint(x, y), erase);
        m_spritesCanvas->canvas()->clearPreviewImage();
        updateSpriteCanvasImage(m_spritesList ? m_spritesList->currentRow() : -1);
        m_spriteUndoActive = false;
    }
}

void MainWindow::handleBackgroundToolPress(int x,
                                           int y,
                                           Qt::MouseButton button,
                                           Qt::KeyboardModifiers /*modifiers*/)
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

void MainWindow::handleBackgroundToolDrag(int x,
                                          int y,
                                          Qt::MouseButtons buttons,
                                          Qt::KeyboardModifiers /*modifiers*/)
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

void MainWindow::handleBackgroundToolRelease(int x,
                                             int y,
                                             Qt::MouseButton button,
                                             Qt::KeyboardModifiers /*modifiers*/)
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
    if (m_paletteGradientActive) {
        cancelPaletteGradient();
        return;
    }
    if (m_paletteSetSlotActive || m_reducedSetSlotActive || m_dynamicSetSlotActive || m_rotationSetSlotActive) {
        cancelPaletteSetSlot();
        return;
    }
    m_frameHasStart = false;
    m_spriteHasStart = false;
    m_frameUndoActive = false;
    m_spriteUndoActive = false;
    m_frameDrawOnMask = false;
    m_frameDrawOnZone = false;
    m_framesCanvas->canvas()->clearPreviewImage();
    m_spritesCanvas->canvas()->clearPreviewImage();
    updateMaskPreviewForFrame(m_framesList->currentRow());
    updateSpriteCanvasImage(m_spritesList ? m_spritesList->currentRow() : -1);
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
        if (m_spriteDynamicSetCombo) {
            m_spriteDynamicSetCombo->setEnabled(false);
        }
        return;
    }
    const std::string& name = m_spriteNames[index];
    if (name.empty()) {
        m_spriteMetaLabel->setText("-");
    } else {
        m_spriteMetaLabel->setText(QString::fromStdString(name));
    }
    if (m_spriteDynamicSetCombo) {
        m_spriteDynamicSetCombo->setEnabled(true);
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
    const QListWidgetItem* firstItem = (m_framesList && m_framesList->count() > 0)
        ? m_framesList->item(0)
        : nullptr;
    if (count <= 0 || (count == 1 && firstItem && firstItem->text().startsWith("No frames"))) {
        m_frameJump->setEnabled(false);
        m_frameJump->setRange(0, 0);
        m_frameJump->setValue(0);
        return;
    }
    m_frameJump->setEnabled(true);
    m_frameJump->setRange(0, std::max(0, count - 1));
    m_frameJump->setValue(std::min(m_frameJump->value(), m_frameJump->maximum()));
}
namespace {
bool EnsureConcentrateExists(const QString& cromPath,
                             const QString& cromcPath,
                             QString* errorMessage)
{
    if (QFileInfo::exists(cromcPath)) {
        return true;
    }
    if (cromPath.isEmpty() || !QFileInfo::exists(cromPath)) {
        if (errorMessage) {
            *errorMessage = "Missing .cROM to generate .cROMc";
        }
        return false;
    }

    const QFileInfo cromInfo(cromPath);
    const QString romName = cromInfo.completeBaseName();
    const QString altcolorDir = cromPath;

    Serum_SetGenerateCRomC(true);
    const uint8_t flags = FLAG_REQUEST_32P_FRAMES | FLAG_REQUEST_64P_FRAMES;
    Serum_Frame_Struc* frame = Serum_Load(altcolorDir.toUtf8().constData(),
                                          romName.toUtf8().constData(),
                                          flags);
    (void)frame;
    Serum_Dispose();
    Serum_SetGenerateCRomC(false);

    if (!QFileInfo::exists(cromcPath)) {
        if (errorMessage) {
            *errorMessage = QString("Failed to generate .cROMc from .cROM. "
                                    "altcolor='%1' rom='%2' frame=0x%3")
                                .arg(altcolorDir,
                                     romName,
                                     QString::number(reinterpret_cast<quintptr>(frame), 16));
        }
        return false;
    }
    return true;
}
}  // namespace
