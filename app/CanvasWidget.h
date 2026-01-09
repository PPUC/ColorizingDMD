#pragma once

#include <QWidget>

class GLCanvasWidget;
class QLabel;
namespace cv {
class Mat;
}

class CanvasWidget : public QWidget
{
    Q_OBJECT
public:
    explicit CanvasWidget(const QString& title, QWidget* parent = nullptr);
    void setTitle(const QString& title);
    void setStatusText(const QString& text);
    void setImage(const cv::Mat& image);

private:
    GLCanvasWidget* m_canvas;
    QLabel* m_label;
};
