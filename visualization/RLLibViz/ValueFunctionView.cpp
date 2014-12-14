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

void ValueFunctionView::add(QWidget* that, const Vec& p, const Vec&)
{
  if (this != that)
    return;
  // Nearest
  if (image)
  {
    image->setPixel(p.y, p.x, qRgb(255, 255, 255));
    update();
  }
}

void ValueFunctionView::add(QWidget* that, const MatrixXd& valueFunction2D)
{
  if (this != that)
    return;
  if (!image)
    image = new QImage(valueFunction2D.cols(), valueFunction2D.rows(), QImage::Format_RGB32);

  RLLib::Range<double> valueFunction2DRange(valueFunction2D.minCoeff(), valueFunction2D.maxCoeff());
  // Value function
  for (int y = 0; y < valueFunction2D.rows(); y++)
  {
    for (int x = 0; x < valueFunction2D.cols(); x++)
    {
      image->setPixel(y, x,
          heatMapGradient.getColorAtValue(
              valueFunction2DRange.toUnit((double) valueFunction2D(y, x))));
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

