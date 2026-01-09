#include "GLCanvasWidget.h"

#include <algorithm>

#include <QMouseEvent>
#include <QPainter>
#include <QWheelEvent>

GLCanvasWidget::GLCanvasWidget(QWidget* parent)
    : QOpenGLWidget(parent)
    , m_overlayText("Canvas")
    , m_zoom(1.0)
    , m_pan(0.0, 0.0)
    , m_panning(false)
{
    setFocusPolicy(Qt::StrongFocus);
}

void GLCanvasWidget::setOverlayText(const QString& text)
{
    m_overlayText = text;
    update();
}

void GLCanvasWidget::setImage(const cv::Mat& image)
{
    if (image.empty()) {
        m_image.release();
        update();
        return;
    }
    if (image.channels() == 3) {
        cv::cvtColor(image, m_image, cv::COLOR_BGR2RGB);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, m_image, cv::COLOR_BGRA2RGBA);
    } else {
        m_image = image.clone();
    }
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

    painter.setPen(QColor(150, 150, 150));
    painter.drawText(QRect(10, 10, width() - 20, 20),
                     Qt::AlignLeft,
                     QString("Zoom %1x  Pan %2,%3")
                         .arg(QString::number(m_zoom, 'f', 2))
                         .arg(QString::number(m_pan.x(), 'f', 1))
                         .arg(QString::number(m_pan.y(), 'f', 1)));
}

void GLCanvasWidget::wheelEvent(QWheelEvent* event)
{
    const double delta = event->angleDelta().y() / 120.0;
    m_zoom = std::max(0.1, std::min(8.0, m_zoom + delta * 0.1));
    update();
}

void GLCanvasWidget::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton) {
        m_panning = true;
        m_lastPos = event->pos();
    }
    QOpenGLWidget::mousePressEvent(event);
}

void GLCanvasWidget::mouseMoveEvent(QMouseEvent* event)
{
    if (m_panning) {
        const QPoint delta = event->pos() - m_lastPos;
        m_pan += QPointF(delta.x(), delta.y());
        m_lastPos = event->pos();
        update();
    }
    QOpenGLWidget::mouseMoveEvent(event);
}

void GLCanvasWidget::mouseReleaseEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton) {
        m_panning = false;
    }
    QOpenGLWidget::mouseReleaseEvent(event);
}
