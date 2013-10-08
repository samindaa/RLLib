#include "RenderArea.h"
#include <iostream>

RenderArea::RenderArea(QWidget *parent)
  : QWidget(parent)
{
  setBackgroundRole(QPalette::Base);
  setAutoFillBackground(true);

  offsetLeft = offsetTop = 10;
  offsetRight = offsetBottom = 2 * 10;
}

RenderArea::~RenderArea()
{ 
}

QSize RenderArea::minimumSizeHint()  const
{
  return QSize(100, 100);
}

QSize RenderArea::sizeHint() const
{
  return QSize(400, 400);
}

void RenderArea::paintEvent(QPaintEvent *)
{
  //std::cout << "RenderArea::paintEvent " << poses.size() <<  std::endl;
  // To Scale
  QRect rect(offsetLeft, offsetTop, width() - offsetRight, height() - offsetBottom);

  QPainter painter(this);
  painter.setPen(QPen(Qt::blue));
  painter.setRenderHint(QPainter::Antialiasing, true);

  // Draw the bounding box
  painter.drawRect(rect);

  QPainterPath path;
  if (poses.size() > 1)
  {
    for (unsigned int i = 0; i < poses.size() - 1; i++)
    {
      const std::pair<float,float>& poseA = poses[i];
      const std::pair<float,float>& poseB = poses[(i + 1) % poses.size()];
      path.moveTo(rect.left() + poseA.first * (rect.right() - rect.left()) / 10.0,
                  rect.top() + poseA.second * (rect.bottom() - rect.top()) / 10.0);
      path.lineTo(rect.left() + poseB.first * (rect.right() - rect.left()) / 10.0,
                  rect.top() + poseB.second * (rect.bottom() - rect.top()) / 10.0);
    }
  }
  // Draw the path
  painter.drawPath(path);
}
