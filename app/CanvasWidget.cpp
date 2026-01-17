#include "CanvasWidget.h"

#include <QLabel>
#include <QSignalBlocker>
#include <QToolButton>
#include <QVBoxLayout>
#include <QHBoxLayout>

#include "GLCanvasWidget.h"

CanvasWidget::CanvasWidget(const QString& title, QWidget* parent)
    : QWidget(parent)
    , m_canvas(new GLCanvasWidget(this))
    , m_label(new QLabel(title, this))
    , m_viewLabel(new QLabel(this))
    , m_fitButton(new QToolButton(this))
    , m_backButton(new QToolButton(this))
    , m_forwardButton(new QToolButton(this))
    , m_gridButton(new QToolButton(this))
    , m_originalButton(new QToolButton(this))
    , m_backgroundButton(new QToolButton(this))
    , m_maskButton(new QToolButton(this))
    , m_dynamicButton(new QToolButton(this))
    , m_backgroundMaskButton(new QToolButton(this))
    , m_zoneButton(new QToolButton(this))
    , m_hdButton(new QToolButton(this))
    , m_rotateButton(new QToolButton(this))
{
    auto* layout = new QVBoxLayout(this);
    auto* toolbar = new QHBoxLayout();
    toolbar->setContentsMargins(0, 0, 0, 0);
    toolbar->setSpacing(6);
    m_canvas->setOverlayText(title);
    m_label->setAlignment(Qt::AlignCenter);
    m_viewLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    m_viewLabel->setText("Zoom 1.00x  Pan 0,0");
    toolbar->addWidget(m_backButton);
    toolbar->addWidget(m_forwardButton);
    toolbar->addWidget(m_fitButton);
    toolbar->addWidget(m_gridButton);
    toolbar->addWidget(m_originalButton);
    toolbar->addWidget(m_backgroundButton);
    toolbar->addWidget(m_maskButton);
    toolbar->addWidget(m_dynamicButton);
    toolbar->addWidget(m_backgroundMaskButton);
    toolbar->addWidget(m_zoneButton);
    toolbar->addWidget(m_hdButton);
    toolbar->addWidget(m_rotateButton);
    toolbar->addStretch(1);
    toolbar->addWidget(m_viewLabel);
    layout->addLayout(toolbar);
    layout->addWidget(m_canvas, 1);
    layout->addWidget(m_label, 0);
    setLayout(layout);

    m_fitButton->setText("Fit");
    m_fitButton->setAutoRaise(true);
    m_fitButton->setToolTip("Fit to view");
    m_fitButton->setCursor(Qt::PointingHandCursor);
    connect(m_fitButton, &QToolButton::clicked, this, &CanvasWidget::fitRequested);

    m_backButton->setText("<");
    m_backButton->setAutoRaise(true);
    m_backButton->setToolTip("Back");
    m_backButton->setCursor(Qt::PointingHandCursor);
    m_backButton->setEnabled(false);
    connect(m_backButton, &QToolButton::clicked, this, &CanvasWidget::backRequested);

    m_forwardButton->setText(">");
    m_forwardButton->setAutoRaise(true);
    m_forwardButton->setToolTip("Forward");
    m_forwardButton->setCursor(Qt::PointingHandCursor);
    m_forwardButton->setEnabled(false);
    connect(m_forwardButton, &QToolButton::clicked, this, &CanvasWidget::forwardRequested);

    m_gridButton->setText("Grid");
    m_gridButton->setCheckable(true);
    m_gridButton->setChecked(true);
    m_gridButton->setAutoRaise(true);
    m_gridButton->setToolTip("Toggle grid");
    m_gridButton->setCursor(Qt::PointingHandCursor);
    connect(m_gridButton, &QToolButton::toggled, this, &CanvasWidget::gridToggled);

    m_originalButton->setText("Original");
    m_originalButton->setCheckable(true);
    m_originalButton->setChecked(true);
    m_originalButton->setAutoRaise(true);
    m_originalButton->setToolTip("Toggle original frame");
    m_originalButton->setCursor(Qt::PointingHandCursor);
    connect(m_originalButton, &QToolButton::toggled, this, &CanvasWidget::originalToggled);

    m_backgroundButton->setText("BG");
    m_backgroundButton->setCheckable(true);
    m_backgroundButton->setChecked(true);
    m_backgroundButton->setAutoRaise(true);
    m_backgroundButton->setToolTip("Toggle background");
    m_backgroundButton->setCursor(Qt::PointingHandCursor);
    connect(m_backgroundButton, &QToolButton::toggled, this, &CanvasWidget::backgroundToggled);

    m_maskButton->setText("Mask");
    m_maskButton->setCheckable(true);
    m_maskButton->setAutoRaise(true);
    m_maskButton->setToolTip("Edit comparison mask");
    m_maskButton->setCursor(Qt::PointingHandCursor);
    connect(m_maskButton, &QToolButton::toggled, this, &CanvasWidget::maskToggled);

    m_dynamicButton->setText("Dynamic");
    m_dynamicButton->setCheckable(true);
    m_dynamicButton->setAutoRaise(true);
    m_dynamicButton->setToolTip("Edit dynamic mask");
    m_dynamicButton->setCursor(Qt::PointingHandCursor);
    connect(m_dynamicButton, &QToolButton::toggled, this, &CanvasWidget::dynamicToggled);

    m_backgroundMaskButton->setText("BG Mask");
    m_backgroundMaskButton->setCheckable(true);
    m_backgroundMaskButton->setAutoRaise(true);
    m_backgroundMaskButton->setToolTip("Edit background mask");
    m_backgroundMaskButton->setCursor(Qt::PointingHandCursor);
    connect(m_backgroundMaskButton, &QToolButton::toggled, this, &CanvasWidget::backgroundMaskToggled);

    m_zoneButton->setText("Zones");
    m_zoneButton->setCheckable(true);
    m_zoneButton->setAutoRaise(true);
    m_zoneButton->setToolTip("Edit sprite detection zones");
    m_zoneButton->setCursor(Qt::PointingHandCursor);
    connect(m_zoneButton, &QToolButton::toggled, this, &CanvasWidget::zoneToggled);

    m_hdButton->setText("HD");
    m_hdButton->setCheckable(true);
    m_hdButton->setAutoRaise(true);
    m_hdButton->setToolTip("Toggle HD frame");
    m_hdButton->setCursor(Qt::PointingHandCursor);
    connect(m_hdButton, &QToolButton::toggled, this, &CanvasWidget::hdToggled);

    m_rotateButton->setText("Rotate");
    m_rotateButton->setCheckable(true);
    m_rotateButton->setAutoRaise(true);
    m_rotateButton->setToolTip("Preview color rotations");
    m_rotateButton->setCursor(Qt::PointingHandCursor);
    connect(m_rotateButton, &QToolButton::toggled, this, &CanvasWidget::rotateToggled);

    connect(m_canvas, &GLCanvasWidget::zoomPanChanged, this, [this](double zoom, const QPointF& pan) {
        m_viewLabel->setText(QString("Zoom %1x  Pan %2,%3")
                                 .arg(QString::number(zoom, 'f', 2))
                                 .arg(QString::number(pan.x(), 'f', 1))
                                 .arg(QString::number(pan.y(), 'f', 1)));
    });
}

void CanvasWidget::setTitle(const QString& title)
{
    m_canvas->setOverlayText(title);
    m_label->setText(title);
}

void CanvasWidget::setStatusText(const QString& text)
{
    m_label->setText(text);
}

void CanvasWidget::setImage(const cv::Mat& image)
{
    m_canvas->setImage(image);
}

GLCanvasWidget* CanvasWidget::canvas() const
{
    return m_canvas;
}

void CanvasWidget::setOriginalVisible(bool enabled)
{
    if (!m_originalButton) {
        return;
    }
    QSignalBlocker blocker(m_originalButton);
    m_originalButton->setChecked(enabled);
}

void CanvasWidget::setMaskButtonsChecked(bool maskEnabled, bool dynamicEnabled)
{
    if (!m_maskButton || !m_dynamicButton) {
        return;
    }
    QSignalBlocker maskBlocker(m_maskButton);
    QSignalBlocker dynamicBlocker(m_dynamicButton);
    m_maskButton->setChecked(maskEnabled);
    m_dynamicButton->setChecked(dynamicEnabled);
}

void CanvasWidget::setMaskButtonsVisible(bool visible)
{
    if (!m_maskButton || !m_dynamicButton) {
        return;
    }
    m_maskButton->setVisible(visible);
    m_dynamicButton->setVisible(visible);
}

void CanvasWidget::setMaskButtonVisible(bool visible)
{
    if (!m_maskButton) {
        return;
    }
    m_maskButton->setVisible(visible);
}

void CanvasWidget::setDynamicButtonVisible(bool visible)
{
    if (!m_dynamicButton) {
        return;
    }
    m_dynamicButton->setVisible(visible);
}

void CanvasWidget::setMaskButtonsEnabled(bool enabled)
{
    if (!m_maskButton || !m_dynamicButton) {
        return;
    }
    m_maskButton->setEnabled(enabled);
    m_dynamicButton->setEnabled(enabled);
}

void CanvasWidget::setMaskButtonText(const QString& text)
{
    if (!m_maskButton) {
        return;
    }
    m_maskButton->setText(text);
}

void CanvasWidget::setMaskButtonToolTip(const QString& text)
{
    if (!m_maskButton) {
        return;
    }
    m_maskButton->setToolTip(text);
}

void CanvasWidget::setBackgroundMaskChecked(bool enabled)
{
    if (!m_backgroundMaskButton) {
        return;
    }
    QSignalBlocker blocker(m_backgroundMaskButton);
    m_backgroundMaskButton->setChecked(enabled);
}

void CanvasWidget::setBackgroundMaskEnabled(bool enabled)
{
    if (!m_backgroundMaskButton) {
        return;
    }
    m_backgroundMaskButton->setEnabled(enabled);
}

void CanvasWidget::setBackgroundMaskVisible(bool visible)
{
    if (!m_backgroundMaskButton) {
        return;
    }
    m_backgroundMaskButton->setVisible(visible);
}

void CanvasWidget::setZoneButtonChecked(bool enabled)
{
    if (!m_zoneButton) {
        return;
    }
    QSignalBlocker blocker(m_zoneButton);
    m_zoneButton->setChecked(enabled);
}

void CanvasWidget::setZoneButtonEnabled(bool enabled)
{
    if (!m_zoneButton) {
        return;
    }
    m_zoneButton->setEnabled(enabled);
}

void CanvasWidget::setZoneButtonVisible(bool visible)
{
    if (!m_zoneButton) {
        return;
    }
    m_zoneButton->setVisible(visible);
}

void CanvasWidget::setRotateChecked(bool enabled)
{
    if (!m_rotateButton) {
        return;
    }
    QSignalBlocker blocker(m_rotateButton);
    m_rotateButton->setChecked(enabled);
}

void CanvasWidget::setRotateEnabled(bool enabled)
{
    if (!m_rotateButton) {
        return;
    }
    m_rotateButton->setEnabled(enabled);
}

void CanvasWidget::setBackEnabled(bool enabled)
{
    if (!m_backButton) {
        return;
    }
    m_backButton->setEnabled(enabled);
}

void CanvasWidget::setForwardEnabled(bool enabled)
{
    if (!m_forwardButton) {
        return;
    }
    m_forwardButton->setEnabled(enabled);
}

void CanvasWidget::setBackgroundChecked(bool enabled)
{
    if (!m_backgroundButton) {
        return;
    }
    QSignalBlocker blocker(m_backgroundButton);
    m_backgroundButton->setChecked(enabled);
}

void CanvasWidget::setBackgroundEnabled(bool enabled)
{
    if (!m_backgroundButton) {
        return;
    }
    m_backgroundButton->setEnabled(enabled);
}

void CanvasWidget::setBackgroundVisible(bool visible)
{
    if (!m_backgroundButton) {
        return;
    }
    m_backgroundButton->setVisible(visible);
}

void CanvasWidget::setHdButtonChecked(bool enabled)
{
    if (!m_hdButton) {
        return;
    }
    QSignalBlocker blocker(m_hdButton);
    m_hdButton->setChecked(enabled);
}

void CanvasWidget::setHdButtonEnabled(bool enabled)
{
    if (!m_hdButton) {
        return;
    }
    m_hdButton->setEnabled(enabled);
}
