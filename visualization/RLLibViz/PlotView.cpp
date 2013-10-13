/*
 * PlotView.cpp
 *
 *  Created on: Oct 13, 2013
 *      Author: sam
 */

#include "PlotView.h"

using namespace RLLibViz;

PlotView::PlotView(QWidget* parent) :
    ViewBase(parent)
{
}

PlotView::~PlotView()
{
}

void PlotView::initialize()
{
  points.resize(sizeHint().width());
}

void PlotView::draw()
{
  update();
}

void PlotView::add(const Vec& p)
{
  // O(N)
  for (size_t i = 1; i < points.size(); i++)
  {
    points[i - 1].x = i - 1;
    points[i - 1].y = points[i].y;
  }
  points[points.size() - 1] = Vec(points.size() - 1, p.x);
}

void PlotView::paintEvent(QPaintEvent* event)
{
  double maxY = points[1].y;
  for (size_t i = 1; i < points.size(); i++)
  {
    double tmp = points[i].y;
    if (tmp > maxY)
      maxY = tmp;
  }
  Mat T = scale(1.0, double(sizeHint().height()) / (maxY + 0.00001), 1.0);

  QPainter painter(this);
  painter.setPen(QPen(Qt::darkBlue, 1, Qt::SolidLine));
  painter.setRenderHint(QPainter::Antialiasing, true);

  QPainterPath path;
  for (int i = 1; i < points.size(); i++)
  {
    Vec p1 = T * points.at(i - 1);
    Vec p2 = T * points.at(i);
    path.moveTo(QPoint(p1.x, p1.y));
    path.lineTo(QPoint(p2.x, p2.y));
  }
  painter.drawPath(path);
}

