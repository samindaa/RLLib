/*
 * AcrobotView.h
 *
 *  Created on: Dec 6, 2014
 *      Author: sam
 */

#ifndef ACROBOTVIEW_H_
#define ACROBOTVIEW_H_

#include "ViewBase.h"

namespace RLLibViz
{

class AcrobotView: public ViewBase
{
  Q_OBJECT

  private:
    Vec p;

  public:
    AcrobotView(QWidget *parent = 0);
    virtual ~AcrobotView();
    void initialize();

  public slots:
    void draw(QWidget* that);
    void add(QWidget* that, const Vec& p1, const Vec& p2);
    void add(QWidget* that, const MatrixXd& mat);

  protected:
    void paintEvent(QPaintEvent* event);
};

}

#endif /* ACROBOTVIEW_H_ */
