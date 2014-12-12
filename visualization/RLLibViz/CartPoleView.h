/*
 * CartPoleView.h
 *
 *  Created on: Dec 8, 2014
 *      Author: sam
 */

#ifndef CARTPOLEVIEW_H_
#define CARTPOLEVIEW_H_

#include "ViewBase.h"

namespace RLLibViz
{

class CartPoleView: public ViewBase
{
  Q_OBJECT

  private:
    Vec p;

  public:
    CartPoleView(QWidget *parent = 0);
    virtual ~CartPoleView();
    void initialize();

  public slots:
    void draw(QWidget* that);
    void add(QWidget* that, const Vec& p1, const Vec& p2);
    void add(QWidget* that, const MatrixXd& mat);

  protected:
    void paintEvent(QPaintEvent* event);
};

}

#endif /* CARTPOLEVIEW_H_ */
