/*
 * MountainCarView.h
 *
 *  Created on: Oct 17, 2013
 *      Author: sam
 */

#ifndef MOUNTAINCARVIEW_H_
#define MOUNTAINCARVIEW_H_

#include "ViewBase.h"

#include <QResizeEvent>
#include <QVector>
#include <QPoint>
#include <QPaintEvent>
#include <QGraphicsScene>

#include <cmath>

namespace RLLibViz
{

class MountainCarView: public ViewBase
{
  Q_OBJECT

  protected:
    QVector<QPointF> points;
    Vec p;
    Vec x;
    Vec y;
    QGraphicsScene* scene;

  public:
    MountainCarView(QWidget *parent = 0);
    virtual ~MountainCarView();
    void initialize();

  public slots:
    void add(QWidget* that, const Vec& p1, const Vec& p2);
    void draw(QWidget* that);
    void add(QWidget* that, const MatrixXd& mat);

  protected:
    void resizeEvent(QResizeEvent *);
    void paintEvent(QPaintEvent *);
};

}  // namespace RLLibViz

#endif /* MOUNTAINCARVIEW_H_ */
