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
  yOne.resize(sizeHint().width());
  yTwo.resize(sizeHint().width());
  for (int i = 0; i < x.size(); i++)
  {
    x[i] = i;
    yOne[i] = yTwo[i] = 0.0f;
  }
  // create graph and assign data to it:
  plot->addGraph();
  plot->addGraph();
  plot->graph(1)->setPen(QPen(Qt::red));
  plot->xAxis->setLabel("Time");
  //plot->yAxis->setLabel("Steps");
  setLayout(grid);
}

PlotView::~PlotView()
{
  //fixMe
}

void PlotView::initialize()
{
}

void PlotView::draw(QWidget* that)
{
  if (this != that)
    return;
  // find min and max
  for (QVector<double>::iterator i = yOne.begin(); i != yOne.end(); ++i)
  {
    double tmp = (*i);
    if (tmp > graphOne.y)
      graphOne.y = tmp;
    if (tmp < graphOne.x)
      graphOne.x = tmp;
  }

  for (QVector<double>::iterator i = yTwo.begin(); i != yTwo.end(); ++i)
  {
    double tmp = (*i);
    if (tmp > graphTwo.y)
      graphTwo.y = tmp;
    if (tmp < graphTwo.x)
      graphTwo.x = tmp;
  }

  double minY = std::min(graphOne.x, graphTwo.x);
  double maxY = std::max(graphOne.y, graphTwo.y);
  plot->xAxis->setRange(0, x.size());
  plot->yAxis->setRange(minY, maxY);
  plot->graph(0)->setData(x, yOne);
  plot->graph(1)->setData(x, yTwo);
  plot->replot();
}

void PlotView::add(QWidget* that, const Vec& graphOneP, const Vec& graphTwoP)
{
  if (this != that)
    return;
  // O(N)
  for (size_t i = 1; i < yOne.size(); i++)
  {
    yOne[i - 1] = yOne[i];
    yTwo[i - 1] = yTwo[i];
  }
  yOne[yOne.size() - 1] = graphOneP.x;
  yTwo[yTwo.size() - 1] = graphTwoP.x;
}
