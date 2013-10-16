/*
 * ValueFunctionView.cpp
 *
 *  Created on: Oct 15, 2013
 *      Author: sam
 */

#include "ValueFunctionView.h"

using namespace RLLibViz;

ValueFunctionView::ValueFunctionView(QWidget *parent) :
    ViewBase(parent), image(0)
{
  // Set the background
  imageLabel = new QLabel;
  imageLabel->setBackgroundRole(QPalette::Base);
  imageLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
  imageLabel->setScaledContents(true);

  mainLayour = new QHBoxLayout;
  mainLayour->addWidget(imageLabel);

  setLayout(mainLayour);
}

ValueFunctionView::~ValueFunctionView()
{
}

void ValueFunctionView::initialize()
{
}

void ValueFunctionView::add(QWidget*, const Vec&, const Vec&)
{ /*Not used*/
}

void ValueFunctionView::add(QWidget* that, const Matrix* mat, double const& minV,
    double const& maxV)
{
  if (this != that)
    return;
  if (!image)
    image = new QImage(mat->cols(), mat->rows(), QImage::Format_RGB32);

  // Value function
  for (unsigned int y = 0; y < mat->rows(); y++)
  {
    for (unsigned int x = 0; x < mat->cols(); x++)
    {
      float value = (mat->at(y, x) - minV) / (maxV - minV);
      image->setPixel(y, x, heatMapGradient.getColorAtValue(value));
    }
  }

  QPixmap p = QPixmap::fromImage(*image);
  imageLabel->setPixmap(p);
  imageLabel->adjustSize();
  imageLabel->setPixmap(p.scaled(width(), height(), Qt::KeepAspectRatio));
  //imageLabel->resize(imageLabel->pixmap()->size());
}

void ValueFunctionView::draw(QWidget* that)
{
  if (this != that)
    return;
  update();
}

