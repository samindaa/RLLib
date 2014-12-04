/*
 * SwingPendulumView.h
 *
 *  Created on: Oct 14, 2013
 *      Author: sam
 */

#ifndef SWINGPENDULUMVIEW_H_
#define SWINGPENDULUMVIEW_H_

#include "ViewBase.h"

namespace RLLibViz
{

class SwingPendulumView: public ViewBase
{
  Q_OBJECT

  private:
    Vec p;
    Mat trans;
    Mat rotation;

  public:
    SwingPendulumView(QWidget *parent = 0);
    virtual ~SwingPendulumView();
    void initialize();

  public slots:
    void draw(QWidget* that);
    void add(QWidget* that, const Vec& p1, const Vec& p2);
    void add(QWidget* that, const MatrixXd& mat);

  protected:
    void paintEvent(QPaintEvent* event);
};

}

#endif /* SWINGPENDULUMVIEW_H_ */
