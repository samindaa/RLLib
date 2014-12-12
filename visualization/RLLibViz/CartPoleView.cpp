/*
 * CartPoleView.cpp
 *
 *  Created on: Dec 8, 2014
 *      Author: sam
 */

#include "CartPoleView.h"

using namespace RLLibViz;

CartPoleView::CartPoleView(QWidget *parent) :
    ViewBase(parent)
{
  // Set the background
  setBackgroundRole(QPalette::Base);
  setAutoFillBackground(true);
}

CartPoleView::~CartPoleView()
{
}

void CartPoleView::initialize()
{
}

void CartPoleView::draw(QWidget* that)
{
  if (this != that)
    return;
  update();
}

void CartPoleView::add(QWidget* that, const Vec& p, const Vec& p2)
{
  if (this != that)
    return;
  this->p = p; // x -> x and y -> theta
}

void CartPoleView::add(QWidget* that, const MatrixXd& mat)
{
}

void CartPoleView::paintEvent(QPaintEvent* event)
{
  RLLib::Range<double> worldXRange(-3.0, 3.0); // spec with some padding
  RLLib::Range<double> windowXRange(0, width());
  QPainter painter1(this);
  painter1.setPen(QPen(Qt::darkRed, 2, Qt::SolidLine));
  painter1.setRenderHint(QPainter::Antialiasing, true);

  Mat worldToScreen = Mat::scale(windowXRange.length() / worldXRange.length(), 1.0, 1.0)
      * Mat::translate(-worldXRange.min(), 0, 0);
  Mat transformationY = Mat::translate(0, double(height() * 2.0 / 3.0), 0);
  Vec cartPositionX = worldToScreen * Vec(p.x, 0, 0, 1.0);
  Vec cartPositionY = transformationY * Vec(0, 0, 0, 1.0);
  painter1.drawLine(QPointF(0, cartPositionY.y), QPointF(width(), cartPositionY.y));

  QPointF topLeft(cartPositionX.x - 50, cartPositionY.y - 50);
  QPointF bottomRight(cartPositionX.x + 50, cartPositionY.y);
  QRectF rect(topLeft, bottomRight);
  QPainter painter2(this);
  painter2.setPen(QPen(Qt::darkRed, 2, Qt::SolidLine));
  painter2.setRenderHint(QPainter::Antialiasing, true);
  painter2.setBrush(QBrush(Qt::lightGray));
  painter2.drawRect(rect);

  QPainter painter3(this);
  painter3.setPen(QPen(Qt::darkGreen, 4, Qt::SolidLine));
  painter3.setRenderHint(QPainter::Antialiasing, true);

  Mat poleTranslation = Mat::translate(cartPositionX) * transformationY * Mat::translate(0, -50, 0);
  Vec pointBase = poleTranslation * Vec(0, 0, 0, 1);
  Vec pointEnd = poleTranslation * Mat::rotateZ(p.y + M_PI) * Vec(0, 100, 0, 1.0);

  painter3.drawEllipse(QPointF(pointBase.x, pointBase.y), 6, 6);
  painter3.drawLine(QPointF(pointBase.x, pointBase.y), QPointF(pointEnd.x, pointEnd.y));
}
