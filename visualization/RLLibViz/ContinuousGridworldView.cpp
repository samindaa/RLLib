/*
 * ContinuousGridworldView.cpp
 *
 *  Created on: Oct 11, 2013
 *      Author: sam
 */

#include "ContinuousGridworldView.h"

using namespace RLLibViz;

ContinuousGridworldView::ContinuousGridworldView(QWidget *parent) :
    ViewBase(parent)
{
  // Set the background
  setBackgroundRole(QPalette::Base);
  setAutoFillBackground(true);

  for (int i = 0; i < 2; i++)
    buffers[i] = new Framebuffer;
  // Double buffering
  current = buffers[0];
  next = buffers[1];
}

ContinuousGridworldView::~ContinuousGridworldView()
{
  for (int i = 0; i < 2; i++)
    delete buffers[i];
}

void ContinuousGridworldView::initialize()
{
  std::cout << "ContinuousGridworldView" << std::endl;
  vecE = Vec(1.0, 1.0);
  vecX = Vec(0.0f, 0.0f);
  vecY = Vec(width(), height());

  Vec diag = vecY - vecX;
  Mat scaleX = Mat::scale(diag.x / vecE.x, 1.0, 1.0);
  Mat scaleY = Mat::scale(1.0, diag.y / vecE.y, 1.0);
  Mat trans = Mat::translate(vecX);
  T = trans * (scaleX * scaleY);

}

void ContinuousGridworldView::add(QWidget* that, const Vec& p1, const Vec& p2)
{
  if (this != that)
    return;
  current->add(T * p1);
}

void ContinuousGridworldView::draw(QWidget* that)
{
  if (this != that)
    return;
  swap();
  current->clear();
  update();
}

void ContinuousGridworldView::add(QWidget* that, const MatrixXd& mat)
{
}

void ContinuousGridworldView::resizeEvent(QResizeEvent* event)
{
  initialize();
}

void ContinuousGridworldView::paintEvent(QPaintEvent* event)
{
  // Draw ellipse
  Vec pVec[3] = { T * Vec(0.3, 0.6, 0.0, 1.0), T * Vec(0.4, 0.5, 0.0, 1.0), T
      * Vec(0.8, 0.9, 0.0, 1.0) };
  Vec vVec[3] =
      { T * (Vec(0.1, 0.03) * 2.0), T * (Vec(0.03, 0.1) * 2.0), T * (Vec(0.03, 0.1) * 2.0) };

  for (int i = 0; i < 3; i++)
  {
    QRadialGradient gradient(QPointF(pVec[i].x, pVec[i].y), std::max(vVec[i].x, vVec[i].y));
    gradient.setColorAt(0.0, Qt::blue);
    gradient.setColorAt(0.5, Qt::cyan);
    gradient.setColorAt(1.0, Qt::green);
    QPen circlePen = QPen(Qt::white);
    circlePen.setWidth(1);
    QPainter painter(this);
    painter.setBrush(gradient);
    painter.setPen(circlePen);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.drawEllipse(QPointF(pVec[i].x, pVec[i].y), vVec[i].x, vVec[i].y);
  }

  QPainter painter2(this);
  painter2.setPen(QPen(Qt::blue, 1, Qt::SolidLine));
  painter2.setRenderHint(QPainter::Antialiasing, true);
  // update() call this
  next->draw(painter2);
}

void ContinuousGridworldView::swap()
{
  Framebuffer* temp = current;
  current = next;
  next = temp;
}

