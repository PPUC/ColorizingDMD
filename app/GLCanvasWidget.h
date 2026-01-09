#pragma once

#include <QOpenGLFunctions>
#include <QOpenGLWidget>

#include <opencv2/opencv.hpp>

class GLCanvasWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT
public:
    explicit GLCanvasWidget(QWidget* parent = nullptr);

    void setOverlayText(const QString& text);
    void setImage(const cv::Mat& image);

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    QString m_overlayText;
    double m_zoom;
    QPointF m_pan;
    QPoint m_lastPos;
    bool m_panning;
    cv::Mat m_image;
};
