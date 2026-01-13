#include "GLCanvasWidget.h"

#include <algorithm>
#include <cmath>

#include <QMouseEvent>
#include <QPainter>
#include <QPen>
#include <QResizeEvent>
#include <QWheelEvent>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMimeData>
#include <QDataStream>
#include <QMap>
#include <QEvent>

GLCanvasWidget::GLCanvasWidget(QWidget* parent)
    : QOpenGLWidget(parent)
    , m_overlayText("Canvas")
    , m_zoom(1.0)
    , m_pan(0.0, 0.0)
    , m_panning(false)
    , m_panningEnabled(true)
    , m_fitOnResize(false)
    , m_gridEnabled(true)
{
    setFocusPolicy(Qt::StrongFocus);
    setAcceptDrops(true);
    setMouseTracking(true);
}

void GLCanvasWidget::setOverlayText(const QString& text)
{
    m_overlayText = text;
    update();
}

namespace {
cv::Mat ConvertToDisplayMat(const cv::Mat& image)
{
    if (image.empty()) {
        return cv::Mat();
    }
    cv::Mat converted;
    if (image.channels() == 3) {
        cv::cvtColor(image, converted, cv::COLOR_BGR2RGB);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, converted, cv::COLOR_BGRA2RGBA);
    } else {
        converted = image.clone();
    }
    return converted;
}
}

void GLCanvasWidget::setImage(const cv::Mat& image)
{
    m_image = ConvertToDisplayMat(image);
    update();
}

void GLCanvasWidget::setPreviewImage(const cv::Mat& image)
{
    m_preview = ConvertToDisplayMat(image);
    update();
}

void GLCanvasWidget::clearPreviewImage()
{
    m_preview.release();
    update();
}

void GLCanvasWidget::setPanningEnabled(bool enabled)
{
    m_panningEnabled = enabled;
    if (!m_panningEnabled) {
        m_panning = false;
    }
}

void GLCanvasWidget::fitToImage()
{
    if (m_image.empty()) {
        return;
    }
    const double scaleX = width() > 0 ? static_cast<double>(width()) / m_image.cols : 1.0;
    const double scaleY = height() > 0 ? static_cast<double>(height()) / m_image.rows : 1.0;
    m_zoom = std::max(0.1, std::min(scaleX, scaleY));
    m_pan = QPointF(0.0, 0.0);
    update();
}

void GLCanvasWidget::requestFitOnResize(bool enabled)
{
    m_fitOnResize = enabled;
    if (m_fitOnResize) {
        fitToImage();
    }
}

void GLCanvasWidget::setGridEnabled(bool enabled)
{
    m_gridEnabled = enabled;
    update();
}

void GLCanvasWidget::setGridSegments(int topHeight, int gapHeight, int bottomHeight)
{
    m_gridTopHeight = topHeight;
    m_gridGap = gapHeight;
    m_gridBottomHeight = bottomHeight;
    update();
}

void GLCanvasWidget::setGridScales(int topScale, int bottomScale)
{
    m_gridTopScale = std::max(1, topScale);
    m_gridBottomScale = std::max(1, bottomScale);
    update();
}

void GLCanvasWidget::setMaskOutline(const cv::Mat& mask, const QColor& color, const QRect& region)
{
    if (mask.empty()) {
        clearMaskOutline();
        return;
    }
    m_outlineMask = mask.clone();
    m_outlineColor = color;
    m_outlineEnabled = true;
    m_outlineRegion = region;
    m_outlineHasRegion = region.isValid() && !region.isEmpty();
    update();
}

void GLCanvasWidget::setSecondaryMaskOutline(const cv::Mat& mask, const QColor& color, const QRect& region)
{
    if (mask.empty()) {
        m_outlineMaskSecondary.release();
        m_outlineSecondaryEnabled = false;
        m_outlineSecondaryHasRegion = false;
        update();
        return;
    }
    m_outlineMaskSecondary = mask.clone();
    m_outlineColorSecondary = color;
    m_outlineSecondaryEnabled = true;
    m_outlineRegionSecondary = region;
    m_outlineSecondaryHasRegion = region.isValid() && !region.isEmpty();
    update();
}

void GLCanvasWidget::clearPrimaryOutline()
{
    m_outlineMask.release();
    m_outlineEnabled = false;
    m_outlineHasRegion = false;
    update();
}

void GLCanvasWidget::clearMaskOutline()
{
    m_outlineMask.release();
    m_outlineEnabled = false;
    m_outlineHasRegion = false;
    m_outlineMaskSecondary.release();
    m_outlineSecondaryEnabled = false;
    m_outlineSecondaryHasRegion = false;
    update();
}

void GLCanvasWidget::setHoverPixelEnabled(bool enabled)
{
    m_hoverPixelEnabled = enabled;
    update();
}

void GLCanvasWidget::scaleZoom(double factor)
{
    if (factor <= 0.0) {
        return;
    }
    m_zoom = std::max(0.1, m_zoom * factor);
    m_fitOnResize = false;
    update();
}

void GLCanvasWidget::initializeGL()
{
    initializeOpenGLFunctions();
    glClearColor(0.08f, 0.09f, 0.1f, 1.0f);
}

void GLCanvasWidget::resizeGL(int w, int h)
{
    glViewport(0, 0, w, h);
}

void GLCanvasWidget::paintGL()
{
    const QColor bg = palette().color(QPalette::Window);
    glClearColor(bg.redF(), bg.greenF(), bg.blueF(), 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    if (!m_image.empty()) {
        QImage image;
        if (m_image.channels() == 3) {
            image = QImage(m_image.data, m_image.cols, m_image.rows, m_image.step, QImage::Format_RGB888);
        } else if (m_image.channels() == 4) {
            image = QImage(m_image.data, m_image.cols, m_image.rows, m_image.step, QImage::Format_RGBA8888);
        }
        if (!image.isNull()) {
            painter.save();
            painter.translate(width() / 2.0 + m_pan.x(), height() / 2.0 + m_pan.y());
            painter.scale(m_zoom, m_zoom);
            QRectF target(-image.width() / 2.0, -image.height() / 2.0, image.width(), image.height());
            painter.drawImage(target, image);
            painter.restore();
        }
    } else {
        painter.setPen(QColor(220, 220, 220));
        painter.drawText(rect(), Qt::AlignCenter, m_overlayText);
    }

    if (!m_preview.empty()) {
        QImage previewImage;
        if (m_preview.channels() == 3) {
            previewImage = QImage(m_preview.data, m_preview.cols, m_preview.rows, m_preview.step, QImage::Format_RGB888);
        } else if (m_preview.channels() == 4) {
            previewImage = QImage(m_preview.data, m_preview.cols, m_preview.rows, m_preview.step, QImage::Format_RGBA8888);
        }
        if (!previewImage.isNull()) {
            painter.save();
            painter.setOpacity(0.5);
            painter.translate(width() / 2.0 + m_pan.x(), height() / 2.0 + m_pan.y());
            painter.scale(m_zoom, m_zoom);
            QRectF target(-previewImage.width() / 2.0, -previewImage.height() / 2.0,
                          previewImage.width(), previewImage.height());
            painter.drawImage(target, previewImage);
            painter.restore();
        }
    }

    if (m_gridEnabled && !m_image.empty()) {
        painter.save();
        painter.setRenderHint(QPainter::Antialiasing, false);
        painter.translate(width() / 2.0 + m_pan.x(), height() / 2.0 + m_pan.y());
        painter.scale(m_zoom, m_zoom);
        const int w = m_image.cols;
        const int h = m_image.rows;
        auto drawRegion = [&](int yStart, int yEnd, int scale) {
            if (yEnd <= yStart) {
                return;
            }
            painter.save();
            painter.setClipRect(QRectF(-w / 2.0,
                                       yStart - h / 2.0,
                                       w,
                                       yEnd - yStart));
            const double gapRatio = 0.5;
            const double cell = std::max(1, scale);
            const double screenStep = cell * m_zoom;
            const double minSpacing = 4.0;
            const int skip = (screenStep > 0.0 && screenStep < minSpacing)
                ? static_cast<int>(std::ceil(minSpacing / screenStep))
                : 1;
            const int step = std::max(1, scale * skip);
            const double lineWidth = 1.0 / 3.0;
            QPen gridPen(QColor(0, 0, 0));
            gridPen.setWidthF(lineWidth);
            gridPen.setCapStyle(Qt::SquareCap);
            painter.setPen(gridPen);
            for (int x = 0; x <= w; x += step) {
                painter.drawLine(QPointF(x - w / 2.0, yStart - h / 2.0),
                                 QPointF(x - w / 2.0, yEnd - h / 2.0));
            }
            for (int y = yStart; y <= yEnd; y += step) {
                painter.drawLine(QPointF(-w / 2.0, y - h / 2.0),
                                 QPointF(w / 2.0, y - h / 2.0));
            }
            painter.restore();
        };
        if (m_gridTopHeight > 0 && m_gridBottomHeight > 0 &&
            m_gridTopHeight + m_gridGap + m_gridBottomHeight <= h) {
            drawRegion(0, m_gridTopHeight, m_gridTopScale);
            drawRegion(m_gridTopHeight + m_gridGap,
                       m_gridTopHeight + m_gridGap + m_gridBottomHeight,
                       m_gridBottomScale);
        } else {
            drawRegion(0, h, 1);
        }
        painter.restore();
    }

    auto drawOutline = [&](const cv::Mat& srcMask,
                           const QColor& color,
                           bool enabled,
                           bool hasRegion,
                           const QRect& regionOverride) {
        if (!enabled || srcMask.empty() || m_image.empty()) {
            return;
        }
        painter.save();
        painter.setRenderHint(QPainter::Antialiasing, false);
        painter.translate(width() / 2.0 + m_pan.x(), height() / 2.0 + m_pan.y());
        painter.scale(m_zoom, m_zoom);
        QRect region(0, 0, m_image.cols, m_image.rows);
        if (hasRegion) {
            region = regionOverride.intersected(QRect(0, 0, m_image.cols, m_image.rows));
        }
        if (region.width() <= 0 || region.height() <= 0) {
            painter.restore();
            return;
        }
        cv::Mat mask = srcMask;
        if (mask.cols != region.width() || mask.rows != region.height()) {
            cv::resize(mask, mask, cv::Size(region.width(), region.height()), 0.0, 0.0, cv::INTER_NEAREST);
        }
        QPen outlinePen(color);
        outlinePen.setWidthF(1.0 / 3.0);
        outlinePen.setCapStyle(Qt::SquareCap);
        painter.setPen(outlinePen);
        const int wMask = mask.cols;
        const int hMask = mask.rows;
        for (int y = 0; y < hMask; ++y) {
            const uint8_t* row = mask.ptr<uint8_t>(y);
            for (int x = 0; x < wMask; ++x) {
                if (!row[x]) {
                    continue;
                }
                const bool leftEmpty = (x == 0) || mask.at<uint8_t>(y, x - 1) == 0;
                const bool rightEmpty = (x == wMask - 1) || mask.at<uint8_t>(y, x + 1) == 0;
                const bool upEmpty = (y == 0) || mask.at<uint8_t>(y - 1, x) == 0;
                const bool downEmpty = (y == hMask - 1) || mask.at<uint8_t>(y + 1, x) == 0;
                const double left = (x + region.x()) - m_image.cols / 2.0;
                const double top = (y + region.y()) - m_image.rows / 2.0;
                if (leftEmpty) {
                    painter.drawLine(QPointF(left, top), QPointF(left, top + 1.0));
                }
                if (rightEmpty) {
                    painter.drawLine(QPointF(left + 1.0, top), QPointF(left + 1.0, top + 1.0));
                }
                if (upEmpty) {
                    painter.drawLine(QPointF(left, top), QPointF(left + 1.0, top));
                }
                if (downEmpty) {
                    painter.drawLine(QPointF(left, top + 1.0), QPointF(left + 1.0, top + 1.0));
                }
            }
        }
        painter.restore();
    };

    drawOutline(m_outlineMask,
                m_outlineColor,
                m_outlineEnabled,
                m_outlineHasRegion,
                m_outlineRegion);
    drawOutline(m_outlineMaskSecondary,
                m_outlineColorSecondary,
                m_outlineSecondaryEnabled,
                m_outlineSecondaryHasRegion,
                m_outlineRegionSecondary);

    if (m_hoverValid && !m_image.empty()) {
        painter.save();
        painter.translate(width() / 2.0 + m_pan.x(), height() / 2.0 + m_pan.y());
        painter.scale(m_zoom, m_zoom);
        const int w = m_image.cols;
        const int h = m_image.rows;
        QRectF highlight(-w / 2.0, -h / 2.0, w, h);
        bool drawHighlight = true;
        if (m_gridTopHeight > 0 && m_gridBottomHeight > 0 &&
            m_gridTopHeight + m_gridGap + m_gridBottomHeight <= h) {
            if (m_hoverY < m_gridTopHeight) {
                highlight = QRectF(-w / 2.0, -h / 2.0, w, m_gridTopHeight);
            } else if (m_hoverY >= m_gridTopHeight + m_gridGap &&
                       m_hoverY < m_gridTopHeight + m_gridGap + m_gridBottomHeight) {
                highlight = QRectF(-w / 2.0,
                                   -h / 2.0 + m_gridTopHeight + m_gridGap,
                                   w,
                                   m_gridBottomHeight);
            } else {
                drawHighlight = false;
            }
        }
        if (drawHighlight) {
            QPen pen(QColor(255, 220, 80, 220));
            const double width = 1.0 / std::max(1.0, m_zoom);
            pen.setWidthF(width);
            painter.setPen(pen);
            painter.setBrush(Qt::NoBrush);
            painter.drawRect(highlight);
        }
        painter.restore();
    }

    if (m_hoverPixelEnabled && m_hoverValid && !m_image.empty()) {
        painter.save();
        painter.setRenderHint(QPainter::Antialiasing, false);
        painter.translate(width() / 2.0 + m_pan.x(), height() / 2.0 + m_pan.y());
        painter.scale(m_zoom, m_zoom);
        const int w = m_image.cols;
        const int h = m_image.rows;
        int scale = 1;
        int regionStart = 0;
        if (m_gridTopHeight > 0 && m_gridBottomHeight > 0 &&
            m_gridTopHeight + m_gridGap + m_gridBottomHeight <= h) {
            if (m_hoverY < m_gridTopHeight) {
                scale = m_gridTopScale;
                regionStart = 0;
            } else if (m_hoverY >= m_gridTopHeight + m_gridGap &&
                       m_hoverY < m_gridTopHeight + m_gridGap + m_gridBottomHeight) {
                scale = m_gridBottomScale;
                regionStart = m_gridTopHeight + m_gridGap;
            }
        }
        const int cellX = (m_hoverX / scale) * scale;
        const int cellY = regionStart + ((m_hoverY - regionStart) / scale) * scale;
        const double left = cellX - w / 2.0;
        const double top = cellY - h / 2.0;
        QPen pen(QColor(255, 220, 80));
        pen.setWidthF(1.0 / 3.0);
        pen.setCapStyle(Qt::SquareCap);
        painter.setPen(pen);
        painter.setBrush(Qt::NoBrush);
        painter.drawRect(QRectF(left, top, scale, scale));
        painter.restore();
    }

    painter.setPen(QColor(150, 150, 150));
    painter.drawText(QRect(10, 10, width() - 20, 20),
                     Qt::AlignLeft,
                     QString("Zoom %1x  Pan %2,%3")
                         .arg(QString::number(m_zoom, 'f', 2))
                         .arg(QString::number(m_pan.x(), 'f', 1))
                         .arg(QString::number(m_pan.y(), 'f', 1)));
}

namespace {
bool MapToImage(const QPoint& pos,
                const cv::Mat& image,
                double zoom,
                const QPointF& pan,
                const QSize& widgetSize,
                int& outX,
                int& outY)
{
    if (image.empty() || zoom <= 0.0) {
        return false;
    }
    const QPointF center(widgetSize.width() / 2.0 + pan.x(), widgetSize.height() / 2.0 + pan.y());
    const QPointF local = (pos - center) / zoom + QPointF(image.cols / 2.0, image.rows / 2.0);
    const int x = static_cast<int>(std::floor(local.x()));
    const int y = static_cast<int>(std::floor(local.y()));
    if (x < 0 || y < 0 || x >= image.cols || y >= image.rows) {
        return false;
    }
    outX = x;
    outY = y;
    return true;
}

bool ExtractMaskDrop(const QMimeData* mimeData, QString& kind, int& index)
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
}

void GLCanvasWidget::wheelEvent(QWheelEvent* event)
{
    const double delta = event->angleDelta().y() / 120.0;
    m_zoom = std::max(0.1, m_zoom + delta * 0.1);
    m_fitOnResize = false;
    update();
}

void GLCanvasWidget::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton && m_panningEnabled) {
        m_panning = true;
        m_lastPos = event->pos();
    }
    int x = 0;
    int y = 0;
    const bool onImage = MapToImage(event->pos(), m_image, m_zoom, m_pan, size(), x, y);
    m_hoverValid = onImage;
    m_hoverX = x;
    m_hoverY = y;
    emit imageHovered(x, y, onImage);
    update();
    if (onImage) {
        emit imageClicked(x, y, event->button(), event->modifiers());
    }
    QOpenGLWidget::mousePressEvent(event);
}

void GLCanvasWidget::mouseMoveEvent(QMouseEvent* event)
{
    int hoverX = -1;
    int hoverY = -1;
    const bool onImage = MapToImage(event->pos(), m_image, m_zoom, m_pan, size(), hoverX, hoverY);
    m_hoverValid = onImage;
    m_hoverX = hoverX;
    m_hoverY = hoverY;
    emit imageHovered(hoverX, hoverY, onImage);
    update();
    if (m_panning) {
        const QPoint delta = event->pos() - m_lastPos;
        m_pan += QPointF(delta.x(), delta.y());
        m_lastPos = event->pos();
        m_fitOnResize = false;
        update();
    }
    if (!m_panning && (event->buttons() & (Qt::LeftButton | Qt::RightButton))) {
        int x = 0;
        int y = 0;
        if (MapToImage(event->pos(), m_image, m_zoom, m_pan, size(), x, y)) {
            emit imageDragged(x, y, event->buttons(), event->modifiers());
        }
    }
    QOpenGLWidget::mouseMoveEvent(event);
}

void GLCanvasWidget::mouseReleaseEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton) {
        m_panning = false;
    }
    int x = 0;
    int y = 0;
    const bool onImage = MapToImage(event->pos(), m_image, m_zoom, m_pan, size(), x, y);
    m_hoverValid = onImage;
    m_hoverX = x;
    m_hoverY = y;
    emit imageHovered(x, y, onImage);
    update();
    if (onImage) {
        emit imageReleased(x, y, event->button(), event->modifiers());
    }
    QOpenGLWidget::mouseReleaseEvent(event);
}

void GLCanvasWidget::resizeEvent(QResizeEvent* event)
{
    QOpenGLWidget::resizeEvent(event);
    if (m_fitOnResize) {
        fitToImage();
    }
}

void GLCanvasWidget::leaveEvent(QEvent* event)
{
    m_hoverValid = false;
    update();
    QOpenGLWidget::leaveEvent(event);
}

void GLCanvasWidget::dragEnterEvent(QDragEnterEvent* event)
{
    QString kind;
    int index = -1;
    if (ExtractMaskDrop(event->mimeData(), kind, index)) {
        event->setDropAction(Qt::CopyAction);
        event->accept();
        return;
    }
    QOpenGLWidget::dragEnterEvent(event);
}

void GLCanvasWidget::dropEvent(QDropEvent* event)
{
    QString kind;
    int index = -1;
    if (ExtractMaskDrop(event->mimeData(), kind, index)) {
        emit maskDropped(kind, index);
        event->setDropAction(Qt::CopyAction);
        event->accept();
        return;
    }
    QOpenGLWidget::dropEvent(event);
}
