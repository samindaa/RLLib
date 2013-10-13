/*
 * Framebuffer.h
 *
 *  Created on: Oct 12, 2013
 *      Author: sam
 */

#ifndef FRAMEBUFFER_H_
#define FRAMEBUFFER_H_

#include <QPainter>

#include <vector>

#include "Vec.h"

namespace RLLibViz
{

class Framebuffer
{
  protected:
    QVector<QPoint> points;
  public:
    Framebuffer();
    ~Framebuffer();
    void draw(QPainter& painter);
    void clear();
    void add(const Vec& p);
};

}  // namespace RLLibViz

#endif /* FRAMEBUFFER_H_ */
