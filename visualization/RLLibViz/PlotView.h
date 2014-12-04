/*
 * PlotView.h
 *
 *  Created on: Oct 13, 2013
 *      Author: sam
 */

#ifndef PLOTVIEW_H_
#define PLOTVIEW_H_

#include <vector>
#include <cmath>

#include <QHBoxLayout>
#include <QVector>
#include <QString>
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
    QVector<double> xR1;
    std::vector<QVector<double> > yR2;
    Vec graphOne;
    Vec graphTwo;
    std::vector<Qt::GlobalColor> globalColors;
    double y2MinAvg;
    double y2MaxAvg;

  public:
    PlotView(const QString& title, QWidget *parent = 0);
    virtual ~PlotView();
    void initialize();

  public slots:
    void add(QWidget* that, const Vec& p1, const Vec& p2);
    void add(QWidget* that, const MatrixXd& mat);
    void draw(QWidget* that);
};

}  // namespace RLLibViz

#endif /* PLOTVIEW_H_ */
