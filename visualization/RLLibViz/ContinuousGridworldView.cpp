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

void ContinuousGridworldView::add(QWidget* that, const Matrix* mat, double const& minV,
    double const& maxV)
{
}

void ContinuousGridworldView::resizeEvent(QResizeEvent* event)
{
  initialize();
}

void ContinuousGridworldView::paintEvent(QPaintEvent* event)
{
  QPainter painter(this);
  painter.setPen(QPen(Qt::blue, 1, Qt::SolidLine));
  painter.setRenderHint(QPainter::Antialiasing, true);
  // update() call this
  next->draw(painter);
}

void ContinuousGridworldView::swap()
{
  Framebuffer* temp = current;
  current = next;
  next = temp;
}

