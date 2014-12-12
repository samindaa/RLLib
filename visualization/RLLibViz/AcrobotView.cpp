/*
 * AcrobotView.cpp
 *
 *  Created on: Dec 6, 2014
 *      Author: sam
 */

#include "AcrobotView.h"

using namespace RLLibViz;

AcrobotView::AcrobotView(QWidget *parent) :
    ViewBase(parent)
{
  // Set the background
  setBackgroundRole(QPalette::Base);
  setAutoFillBackground(true);
}

AcrobotView::~AcrobotView()
{
}

void AcrobotView::initialize()
{
}

void AcrobotView::draw(QWidget* that)
{
  if (this != that)
    return;
  update();
}

void AcrobotView::add(QWidget* that, const Vec& p, const Vec& p2)
{
  if (this != that)
    return;
  this->p = p; // x -> theta1 and y -> theta2
}

void AcrobotView::add(QWidget* that, const MatrixXd& mat)
{
}

void AcrobotView::paintEvent(QPaintEvent* event)
{
  QPainter painter(this);
  painter.setPen(QPen(Qt::darkRed, 2, Qt::SolidLine));
  painter.setRenderHint(QPainter::Antialiasing, true);

  Mat baseToMid = Mat::translate(double(width()) / 2.0, double(height()) / 2.0, 0);
  Vec baseL1 = baseToMid * Vec(0.0, 0.0, 0.0, 1.0);
  Mat midToL1End = baseToMid * Mat::rotateZ(p.x)
      * Mat::translate(Vec(0.0, double(height()) / 4.0, 0.0, 1.0));
  Vec endL1 = midToL1End * Vec(0.0, 0.0, 0.0, 1.0);
  Vec endL2 = midToL1End * Mat::rotateZ(p.y)
      * Mat::translate(Vec(0.0, double(height()) / 4.0, 0.0, 1.0)) * Mat::rotateZ(p.y)
      * Vec(0.0, 0.0, 0.0, 1.0);

  painter.drawEllipse(QPointF(baseL1.x, baseL1.y), 6, 6);
  painter.drawLine(QPointF(baseL1.x, baseL1.y), QPointF(endL1.x, endL1.y));
  painter.setBrush(QBrush(Qt::darkRed));
  painter.drawEllipse(QPointF(endL1.x, endL1.y), 6, 6);
  painter.drawLine(QPointF(endL1.x, endL1.y), QPointF(endL2.x, endL2.y));
  painter.drawEllipse(QPointF(endL2.x, endL2.y), 6, 6);
}

