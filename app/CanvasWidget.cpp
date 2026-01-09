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

void CanvasWidget::resizeEvent(QResizeEvent* event)
{
    QWidget::resizeEvent(event);
    if (!m_fitButton || !m_gridButton) {
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
}
