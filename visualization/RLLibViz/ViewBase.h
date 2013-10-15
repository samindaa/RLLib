/*
 * ViewBase.h
 *
 *  Created on: Oct 11, 2013
 *      Author: sam
 */

#ifndef VIEWBASE_H_
#define VIEWBASE_H_

// Qt4
#include <QWidget>
#include <QBrush>
#include <QPen>
#include <QPainter>
#include <QKeyEvent>

// RLLibViz
#include "Mat.h"
#include "Framebuffer.h"

#include <vector>

namespace RLLibViz
{

class ViewBase: public QWidget
{
  protected:
    Framebuffer* buffers[2];
    Framebuffer* current;
    Framebuffer* next;

  public:
    ViewBase(QWidget *parent = 0);
    ~ViewBase();

    QSize minimumSizeHint() const;
    QSize sizeHint() const;
    virtual void initialize() =0;

  public slots:
    virtual void add(QWidget* that, const Vec& p1, const Vec& p2) =0;
    virtual void draw(QWidget* that) =0;

  protected:
    virtual void swap();
    void paintEvent(QPaintEvent *);
};

}  // namespace RLLibViz

#endif /* VIEWBASE_H_ */
