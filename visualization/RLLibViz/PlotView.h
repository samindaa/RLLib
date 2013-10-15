/*
 * PlotView.h
 *
 *  Created on: Oct 13, 2013
 *      Author: sam
 */

#ifndef PLOTVIEW_H_
#define PLOTVIEW_H_

#include <vector>

#include <QHBoxLayout>
#include <QVector>
#include "ViewBase.h"
#include "Mat.h"
#include "plot/qcustomplot.h"

namespace RLLibViz
{

class PlotView: public ViewBase
{
  Q_OBJECT
  private:
    QHBoxLayout* grid;
    QCustomPlot* plot;
    QVector<double> x, yOne, yTwo;
    Vec graphOne;
    Vec graphTwo;
  public:
    PlotView(QWidget *parent = 0);
    virtual ~PlotView();
    void initialize();

  public slots:
    void add(QWidget* that, const Vec& p1, const Vec& p2);
    void draw(QWidget* that);
};

}  // namespace RLLibViz

#endif /* PLOTVIEW_H_ */
