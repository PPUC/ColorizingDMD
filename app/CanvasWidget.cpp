#include "CanvasWidget.h"

#include <QLabel>
#include <QVBoxLayout>

#include "GLCanvasWidget.h"

CanvasWidget::CanvasWidget(const QString& title, QWidget* parent)
    : QWidget(parent)
    , m_canvas(new GLCanvasWidget(this))
    , m_label(new QLabel(title, this))
{
    auto* layout = new QVBoxLayout(this);
    m_canvas->setOverlayText(title);
    m_label->setAlignment(Qt::AlignCenter);
    layout->addWidget(m_canvas, 1);
    layout->addWidget(m_label, 0);
    setLayout(layout);
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
