/*
 * ViewBase.h
 *
 *  Created on: Oct 11, 2013
 *      Author: sam
 */

#ifndef VIEWBASE_H_
#define VIEWBASE_H_

#include <vector>
#include <QWidget>
#include <QBrush>
#include <QPen>
#include <QPainter>
#include <QKeyEvent>

// Affine transformations
#include "Mat.h"
#include "Mathema.h"
// For dense matrices
#include "Eigen/Dense"

using namespace RLLib;
using namespace Eigen;

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
    virtual void add(QWidget* that, const MatrixXd& mat) =0;
    virtual void draw(QWidget* that) =0;
};

}  // namespace RLLibViz

#endif /* VIEWBASE_H_ */
