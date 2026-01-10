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
    void setPreviewImage(const cv::Mat& image);
    void clearPreviewImage();
    void setPanningEnabled(bool enabled);
    void fitToImage();
    void requestFitOnResize(bool enabled = true);
    void setGridEnabled(bool enabled);
    void setGridSegments(int topHeight, int gapHeight, int bottomHeight);

signals:
    void imageClicked(int x, int y, Qt::MouseButton button);
    void imageDragged(int x, int y, Qt::MouseButtons buttons);
    void imageReleased(int x, int y, Qt::MouseButton button);
    void maskDropped(const QString& kind, int index);

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dropEvent(QDropEvent* event) override;

private:
    QString m_overlayText;
    double m_zoom;
    QPointF m_pan;
    QPoint m_lastPos;
    bool m_panning;
    bool m_panningEnabled;
    bool m_fitOnResize;
    bool m_gridEnabled;
    int m_gridTopHeight = 0;
    int m_gridBottomHeight = 0;
    int m_gridGap = 0;
    cv::Mat m_image;
    cv::Mat m_preview;
};
