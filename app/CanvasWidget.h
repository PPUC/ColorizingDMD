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
    void setMaskButtonsChecked(bool maskEnabled, bool dynamicEnabled);
    void setMaskButtonsVisible(bool visible);
    void setMaskButtonsEnabled(bool enabled);
    void setHdButtonChecked(bool enabled);
    void setHdButtonEnabled(bool enabled);
    void setBackgroundChecked(bool enabled);
    void setBackgroundEnabled(bool enabled);
    void setBackgroundVisible(bool visible);
    void setBackgroundMaskChecked(bool enabled);
    void setBackgroundMaskEnabled(bool enabled);
    void setBackgroundMaskVisible(bool visible);

signals:
    void fitRequested();
    void gridToggled(bool enabled);
    void originalToggled(bool enabled);
    void backgroundToggled(bool enabled);
    void maskToggled(bool enabled);
    void dynamicToggled(bool enabled);
    void backgroundMaskToggled(bool enabled);
    void hdToggled(bool enabled);

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    GLCanvasWidget* m_canvas;
    QLabel* m_label;
    QToolButton* m_fitButton;
    QToolButton* m_gridButton;
    QToolButton* m_originalButton;
    QToolButton* m_backgroundButton;
    QToolButton* m_maskButton;
    QToolButton* m_dynamicButton;
    QToolButton* m_backgroundMaskButton;
    QToolButton* m_hdButton;
};
