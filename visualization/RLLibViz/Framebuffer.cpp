/*
 * Framebuffer.cpp
 *
 *  Created on: Oct 12, 2013
 *      Author: sam
 */

#include "Framebuffer.h"
#include <cassert>

using namespace RLLibViz;

Framebuffer::Framebuffer()
{

}

Framebuffer::~Framebuffer()
{
}

void Framebuffer::draw(QPainter& painter)
{
  if (points.empty() || points.size() < 1)
    return;
  QPainterPath path;
  for (int i = 1; i < points.size(); i++)
  {
    path.moveTo(points.at(i-1));
    path.lineTo(points.at(i));
  }
  painter.drawPath(path);
}

void Framebuffer::clear()
{
  points.clear();
}

void Framebuffer::add(const Vec& p)
{
  points.push_back(QPoint(p.x, p.y));
}

