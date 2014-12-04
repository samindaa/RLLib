/*
 * SwingPendulumView.cpp
 *
 *  Created on: Oct 14, 2013
 *      Author: sam
 */

#include "SwingPendulumView.h"

using namespace RLLibViz;

SwingPendulumView::SwingPendulumView(QWidget *parent) :
    ViewBase(parent)
{
  // Set the background
  setBackgroundRole(QPalette::Base);
  setAutoFillBackground(true);
}

SwingPendulumView::~SwingPendulumView()
{
}

void SwingPendulumView::initialize()
{
}

void SwingPendulumView::draw(QWidget* that)
{
  if (this != that)
    return;
  update();
}

void SwingPendulumView::add(QWidget* that, const Vec& p, const Vec& p2)
{
  if (this != that)
    return;
  this->p = p;
  trans = Mat::translate(double(width()) / 2.0, double(height()) / 2.0, 0);
  rotation = Mat::rotateZ(p.x + M_PI);
}

void SwingPendulumView::add(QWidget* that, const MatrixXd& mat)
{
}

void SwingPendulumView::paintEvent(QPaintEvent* event)
{
  QPainter painter(this);
  painter.setPen(QPen(Qt::darkRed, 2, Qt::SolidLine));
  painter.setRenderHint(QPainter::Antialiasing, true);

  Vec p1 = trans * Vec(0.0, 0.0, 0.0, 1.0);
  Vec p2 = trans * rotation * Vec(0.0, double(height()) / 2.0 - 5.0, 0.0, 1.0);

  painter.drawEllipse(QPointF(p1.x, p1.y), 6, 6);
  painter.drawLine(QPointF(p1.x, p1.y), QPointF(p2.x, p2.y));
  painter.setBrush(QBrush(Qt::darkRed));
  painter.drawEllipse(QPointF(p2.x, p2.y), 6, 6);
}
