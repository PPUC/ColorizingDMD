#pragma once

#include <QWidget>

class GLCanvasWidget;
class QLabel;
class QToolButton;
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
    GLCanvasWidget* canvas() const;
    void setOriginalVisible(bool enabled);

signals:
    void fitRequested();
    void gridToggled(bool enabled);
    void originalToggled(bool enabled);

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    GLCanvasWidget* m_canvas;
    QLabel* m_label;
    QToolButton* m_fitButton;
    QToolButton* m_gridButton;
    QToolButton* m_originalButton;
};
