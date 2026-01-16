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
    void setMaskButtonVisible(bool visible);
    void setDynamicButtonVisible(bool visible);
    void setMaskButtonsEnabled(bool enabled);
    void setMaskButtonText(const QString& text);
    void setMaskButtonToolTip(const QString& text);
    void setHdButtonChecked(bool enabled);
    void setHdButtonEnabled(bool enabled);
    void setBackgroundChecked(bool enabled);
    void setBackgroundEnabled(bool enabled);
    void setBackgroundVisible(bool visible);
    void setBackgroundMaskChecked(bool enabled);
    void setBackgroundMaskEnabled(bool enabled);
    void setBackgroundMaskVisible(bool visible);
    void setZoneButtonChecked(bool enabled);
    void setZoneButtonEnabled(bool enabled);
    void setZoneButtonVisible(bool visible);
    void setRotateChecked(bool enabled);
    void setRotateEnabled(bool enabled);
    void setBackEnabled(bool enabled);
    void setForwardEnabled(bool enabled);

signals:
    void fitRequested();
    void backRequested();
    void forwardRequested();
    void gridToggled(bool enabled);
    void originalToggled(bool enabled);
    void backgroundToggled(bool enabled);
    void maskToggled(bool enabled);
    void dynamicToggled(bool enabled);
    void backgroundMaskToggled(bool enabled);
    void zoneToggled(bool enabled);
    void hdToggled(bool enabled);
    void rotateToggled(bool enabled);

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    GLCanvasWidget* m_canvas;
    QLabel* m_label;
    QToolButton* m_fitButton;
    QToolButton* m_backButton;
    QToolButton* m_forwardButton;
    QToolButton* m_gridButton;
    QToolButton* m_originalButton;
    QToolButton* m_backgroundButton;
    QToolButton* m_maskButton;
    QToolButton* m_dynamicButton;
    QToolButton* m_backgroundMaskButton;
    QToolButton* m_zoneButton;
    QToolButton* m_hdButton;
    QToolButton* m_rotateButton;
};
