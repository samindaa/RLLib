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
#include "Matrix.h"

#include <vector>

using namespace RLLib;

namespace RLLibViz
{

class ViewBase: public QWidget
{
  public:
    ViewBase(QWidget *parent = 0);
    ~ViewBase();

    // RLLibViz
    virtual void initialize() =0;

  public slots:
    virtual void add(QWidget* that, const Vec& p1, const Vec& p2) =0;
    virtual void add(QWidget* that, const Matrix* mat, double const& minV, double const& maxV) =0;
    virtual void draw(QWidget* that) =0;
};

}  // namespace RLLibViz

#endif /* VIEWBASE_H_ */
