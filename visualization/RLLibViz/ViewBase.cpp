/*
 * ViewBase.cpp
 *
 *  Created on: Oct 11, 2013
 *      Author: sam
 */

#include "ViewBase.h"

using namespace RLLibViz;

ViewBase::ViewBase(QWidget *parent) :
    QWidget(parent)
{
  for (int i = 0; i < 2; i++)
    buffers[i] = new Framebuffer;
  // Double buffering
  current = buffers[0];
  next = buffers[1];
  // Set the background
  setBackgroundRole(QPalette::Base);
  setAutoFillBackground(true);
}

ViewBase::~ViewBase()
{
  for (int i = 0; i < 2; i++)
    delete buffers[i];
}

QSize ViewBase::minimumSizeHint() const
{
  return QSize(100, 100);
}

QSize ViewBase::sizeHint() const
{
  return QSize(250, 250);
}

void ViewBase::paintEvent(QPaintEvent* event)
{
  QPainter painter(this);
  painter.setPen(QPen(Qt::blue, 1, Qt::SolidLine));
  painter.setRenderHint(QPainter::Antialiasing, true);
  // update() call this
  next->draw(painter);
}

void ViewBase::draw()
{
  swap();
  current->clear();
  update();
}

void ViewBase::swap()
{
  Framebuffer* temp = current;
  current = next;
  next = temp;
}
