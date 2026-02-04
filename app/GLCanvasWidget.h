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
    void setPixelDiameterPercent(int percent);
    void setGridSegments(int topHeight, int gapHeight, int bottomHeight);
    void setGridScales(int topScale, int bottomScale);
    void setGridRegions(const QRect& topRegion, const QRect& bottomRegion);
    void setMaskOutline(const cv::Mat& mask, const QColor& color, const QRect& region = QRect());
    void setSecondaryMaskOutline(const cv::Mat& mask, const QColor& color, const QRect& region = QRect());
    void setTertiaryMaskOutline(const cv::Mat& mask, const QColor& color, const QRect& region = QRect());
    void clearPrimaryOutline();
    void clearTertiaryOutline();
    void clearMaskOutline();
    void setHoverPixelEnabled(bool enabled);
    void scaleZoom(double factor);

signals:
    void imageClicked(int x, int y, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void imageDragged(int x, int y, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers);
    void imageReleased(int x, int y, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void maskDropped(const QString& kind, int index);
    void imageHovered(int x, int y, bool onImage);
    void zoomPanChanged(double zoom, const QPointF& pan);

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
    void leaveEvent(QEvent* event) override;

private:
    QString m_overlayText;
    double m_zoom;
    QPointF m_pan;
    QPoint m_lastPos;
    bool m_panning;
    bool m_panningEnabled;
    bool m_fitOnResize;
    bool m_fitOncePending = true;
    bool m_gridEnabled;
    double m_pixelDiameter = 0.8;
    int m_gridTopHeight = 0;
    int m_gridBottomHeight = 0;
    int m_gridGap = 0;
    int m_gridTopScale = 1;
    int m_gridBottomScale = 1;
    QRect m_gridTopRegion;
    QRect m_gridBottomRegion;
    bool m_gridHasTopRegion = false;
    bool m_gridHasBottomRegion = false;
    cv::Mat m_outlineMask;
    QColor m_outlineColor = QColor(200, 0, 200);
    bool m_outlineEnabled = false;
    QRect m_outlineRegion;
    bool m_outlineHasRegion = false;
    cv::Mat m_outlineMaskSecondary;
    QColor m_outlineColorSecondary = QColor(120, 200, 60);
    bool m_outlineSecondaryEnabled = false;
    QRect m_outlineRegionSecondary;
    bool m_outlineSecondaryHasRegion = false;
    cv::Mat m_outlineMaskTertiary;
    QColor m_outlineColorTertiary = QColor(255, 220, 0);
    bool m_outlineTertiaryEnabled = false;
    QRect m_outlineRegionTertiary;
    bool m_outlineTertiaryHasRegion = false;
    bool m_hoverValid = false;
    int m_hoverX = -1;
    int m_hoverY = -1;
    bool m_hoverPixelEnabled = false;
    cv::Mat m_image;
    cv::Mat m_preview;
};
