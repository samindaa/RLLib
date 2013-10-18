/*
 * MainMountainCar.cpp
 *
 *  Created on: Oct 17, 2013
 *      Author: sam
 */
#include <QApplication>

#include "MountainCarModel.h"
#include "MountainCarView.h"
#include "ValueFunctionView.h"
#include "NULLView.h"
#include "PlotView.h"
#include "ModelThread.h"
#include "Window.h"

using namespace RLLibViz;
using namespace RLLib;

Q_DECLARE_METATYPE(Vec)
Q_DECLARE_METATYPE(Matrix)

int main(int argc, char *argv[])
{
  QApplication a(argc, argv);
  qRegisterMetaType<Vec>();
  qRegisterMetaType<Matrix>();

  RLLibViz::ViewBase* behaviorView = new RLLibViz::MountainCarView;
  RLLibViz::ViewBase* targetView = new RLLibViz::MountainCarView;
  RLLibViz::ViewBase* behaviorPlot = new RLLibViz::PlotView;
  RLLibViz::ViewBase* targetPlot = new RLLibViz::PlotView;
  RLLibViz::ViewBase* valueFunctionNULLView = new RLLibViz::NULLView;
  RLLibViz::ViewBase* valueFunctionView = new RLLibViz::ValueFunctionView;

  RLLibViz::Window* window = new RLLibViz::Window;
  window->addView(behaviorView);
  window->addView(targetView);
  window->addPlot(behaviorPlot);
  window->addPlot(targetPlot);
  window->addValueFunctionView(valueFunctionNULLView);
  window->addValueFunctionView(valueFunctionView);
  window->setWindowTitle("RLLibViz");
  window->show();

  RLLibViz::ModelBase* model = new RLLibViz::MountainCarModel;
  model->setWindow(window);
  model->initialize();

  RLLibViz::ModelThread* thread = new RLLibViz::ModelThread;
  thread->setModel(model);
  thread->start();

  return a.exec();
}

