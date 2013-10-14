/*
 * PlotView.cpp
 *
 *  Created on: Oct 13, 2013
 *      Author: sam
 */

#include "PlotView.h"

using namespace RLLibViz;

PlotView::PlotView(QWidget* parent) :
    ViewBase(parent)
{
  grid = new QHBoxLayout(this);
  plot = new QCustomPlot(this);
  plot->setGeometry(0, 0, sizeHint().width(), sizeHint().height());
  grid->addWidget(plot);

  x.resize(sizeHint().width());
  y.resize(sizeHint().width());
  for (int i = 0; i < x.size(); i++)
  {
    x[i] = i;
    y[i] = 0.0f;
  }
  gMinY = gMaxY = 0.0f;
  // create graph and assign data to it:
  plot->addGraph();
  plot->xAxis->setLabel("Time");
  plot->yAxis->setLabel("Steps");
  setLayout(grid);
}

PlotView::~PlotView()
{
  //fixMe
}

void PlotView::initialize()
{
}

void PlotView::draw()
{
  // find min and max
  for (QVector<double>::iterator i = y.begin(); i != y.end(); ++i)
  {
    double tmp = (*i);
    if (tmp > gMaxY)
      gMaxY = tmp;
    if (tmp < gMinY)
      gMinY = tmp;
  }
  plot->xAxis->setRange(0, x.size());
  plot->yAxis->setRange(gMinY, gMaxY);
  plot->graph(0)->setData(x, y);
  plot->replot();
}

void PlotView::add(const Vec& p)
{
  // O(N)
  for (size_t i = 1; i < y.size(); i++)
    y[i - 1] = y[i];
  y[y.size() - 1] = p.x;
}
