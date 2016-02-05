/*
 * MountainCarView.cpp
 *
 *  Created on: Oct 17, 2013
 *      Author: sam
 */

#include "MountainCarView.h"

using namespace RLLibViz;

MountainCarView::MountainCarView(QWidget *parent) :
    ViewBase(parent), scene(new QGraphicsScene(this))
{
  // Set the background
  setBackgroundRole(QPalette::Base);
  setAutoFillBackground(true);
}

MountainCarView::~MountainCarView()
{
  delete scene;
}

void MountainCarView::initialize()
{
  std::cout << "MountainCarView" << std::endl;
  points.clear();
  x = Vec(-1.2, 0.6); // << from specification

  //Vec screen(width(), height());
  y = Vec(::sin(3.0 * x.x), ::sin(3.0 * x.x));
  for (double i = x.x; i < x.y; i += 0.1)
  {
    double tmp = ::sin(3.0 * i);
    if (tmp < y.x)
      y.x = tmp;
    if (tmp > y.y)
      y.y = tmp;
  }

  for (double i = x.x; i < x.y; i += 0.1)
  {
    Vec sp((i - x.x) / (x.y - x.x) * width(), (::sin(3.0 * i) - y.x) / (y.y - y.x) * height());
    points.push_back(QPointF(sp.x, -sp.y + height()));
  }

}

void MountainCarView::add(QWidget* that, const Vec& p1, const Vec& p2)
{
  if (this != that)
    return;
  p = p1;
}

void MountainCarView::draw(QWidget* that)
{
  if (this != that)
    return;
  update();
}

void MountainCarView::add(QWidget* that, const MatrixXd& mat)
{
  // Not used
}

void MountainCarView::resizeEvent(QResizeEvent* event)
{
  initialize();
}

void MountainCarView::paintEvent(QPaintEvent* event)
{
  QPainter painter(this);
  painter.setPen(QPen(Qt::darkBlue, 1, Qt::SolidLine));
  painter.setRenderHint(QPainter::Antialiasing, true);
  // update() call this

  // TODO:
  QPainterPath path;
  for (int i = 1; i < points.size(); i++)
  {
    path.moveTo(points.at(i - 1));
    path.lineTo(points.at(i));
  }
  painter.drawPath(path);

  Vec sp((p.x - x.x) / (x.y - x.x) * width(), (::sin(3.0 * p.x) - y.x) / (y.y - y.x) * height());
  QPointF ep(sp.x, -sp.y + height());
  QPainter painter2(this);
  painter2.setPen(QPen(Qt::darkRed, 2, Qt::SolidLine));
  painter2.setBrush(QBrush(Qt::darkRed));
  painter2.setRenderHint(QPainter::Antialiasing, true);
  painter2.drawEllipse(ep, 8, 8);
}

