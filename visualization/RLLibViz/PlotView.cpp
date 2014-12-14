/*
 * PlotView.cpp
 *
 *  Created on: Oct 13, 2013
 *      Author: sam
 */

#include "PlotView.h"

using namespace RLLibViz;

PlotView::PlotView(const QString& title, QWidget* parent) :
    ViewBase(parent), y2MinAvg(0), y2MaxAvg(0)
{
  grid = new QHBoxLayout(this);
  plot = new QCustomPlot(this);
  plot->setGeometry(0, 0, sizeHint().width(), sizeHint().height());
  grid->addWidget(plot);

  xR1.resize(sizeHint().width());
  for (int i = 0; i < xR1.size(); i++)
    xR1[i] = i - (xR1.size() - 1);

  globalColors.push_back(Qt::darkRed);
  globalColors.push_back(Qt::darkGreen);

  for (int i = 0; i < 2; i++)
  {
    yR2.push_back(QVector<double>());
    yR2[i].resize(sizeHint().width());
    for (int j = 0; j < sizeHint().width(); j++)
      yR2[i][j] = 0.0f;
  }

  // create graph and assign data to it:
  for (int i = 0; i < 2; i++)
    plot->addGraph();

  for (int i = 0; i < 2; i++)
    plot->graph(i)->setPen(QPen(globalColors[i]));

  plot->xAxis->setLabel("Episodes");
  plot->yAxis->setLabel("Steps");
  plot->yAxis2->setLabel("Rewards");
  plot->xAxis2->setLabel(title);
  plot->xAxis2->setVisible(true);
  plot->yAxis2->setVisible(true);
  setLayout(grid);
}

PlotView::~PlotView()
{
  delete grid;
  delete plot;
}

void PlotView::initialize()
{
}

void PlotView::draw(QWidget* that)
{
  if (this != that)
    return;

  for (int i = 0; i < 2; i++)
  {
    plot->graph(i)->setData(xR1, yR2[i]);
    plot->graph(i)->rescaleAxes();
    plot->xAxis->setRange(xR1[0], xR1[xR1.size() - 1]);
    plot->xAxis2->setRange(xR1[0], xR1[xR1.size() - 1]);
    plot->yAxis->setRange(yR2[0][0], yR2[0][yR2[0].size() - 1]);

    y2MinAvg += 0.01 * (yR2[1][0] - y2MinAvg);
    y2MaxAvg += 0.01 * (yR2[1][yR2[1].size() - 1] - y2MaxAvg);
    plot->yAxis2->setRange(std::ceil(y2MinAvg), std::ceil(y2MaxAvg));
  }

  plot->rescaleAxes();
  plot->replot();
}

void PlotView::add(QWidget* that, const Vec& graphOneP, const Vec& graphTwoP)
{
  if (this != that)
    return;

  // O(N)
  for (size_t i = 0; i < yR2.size(); i++)
  {
    for (int j = 1; j < yR2[i].size(); j++)
      yR2[i][j - 1] = yR2[i][j];
  }

  yR2[0][yR2[0].size() - 1] = graphOneP.x;
  yR2[1][yR2[1].size() - 1] = graphTwoP.x;
  for (int i = 1; i < xR1.size(); i++)
    xR1[i - 1] = xR1[i];
  ++xR1[xR1.size() - 1];
}

void PlotView::add(QWidget* that, const MatrixXd& mat)
{
  (void) that;
  (void) mat;
}
