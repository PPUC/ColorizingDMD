#include "CanvasWidget.h"

#include <QLabel>
#include <QResizeEvent>
#include <QToolButton>
#include <QVBoxLayout>

#include "GLCanvasWidget.h"

CanvasWidget::CanvasWidget(const QString& title, QWidget* parent)
    : QWidget(parent)
    , m_canvas(new GLCanvasWidget(this))
    , m_label(new QLabel(title, this))
    , m_fitButton(new QToolButton(m_canvas))
    , m_gridButton(new QToolButton(m_canvas))
    , m_originalButton(new QToolButton(m_canvas))
{
    auto* layout = new QVBoxLayout(this);
    m_canvas->setOverlayText(title);
    m_label->setAlignment(Qt::AlignCenter);
    layout->addWidget(m_canvas, 1);
    layout->addWidget(m_label, 0);
    setLayout(layout);

    m_fitButton->setText("Fit");
    m_fitButton->setAutoRaise(true);
    m_fitButton->setToolTip("Fit to view");
    m_fitButton->setCursor(Qt::PointingHandCursor);
    connect(m_fitButton, &QToolButton::clicked, this, &CanvasWidget::fitRequested);

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

void CanvasWidget::resizeEvent(QResizeEvent* event)
{
    QWidget::resizeEvent(event);
    if (!m_fitButton || !m_gridButton || !m_originalButton) {
        return;
    }
    const int margin = 8;
    const QSize buttonSize = m_fitButton->sizeHint();
    const int x = m_canvas->width() - buttonSize.width() - margin;
    const int y = margin;
    m_fitButton->move(x, y);
    m_fitButton->raise();

    const QSize gridSize = m_gridButton->sizeHint();
    const int gridX = x - gridSize.width() - 6;
    m_gridButton->move(gridX, y);
    m_gridButton->raise();

    const QSize originalSize = m_originalButton->sizeHint();
    const int originalX = gridX - originalSize.width() - 6;
    m_originalButton->move(originalX, y);
    m_originalButton->raise();
}
