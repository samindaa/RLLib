/*
 * MainSwingPendulum2.cpp
 *
 *  Created on: Nov 20, 2013
 *      Author: sam
 */

#include <QApplication>
#include "SwingPendulumModel2.h"
#include "SwingPendulumView.h"
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

  RLLibViz::ViewBase* behaviorView = new RLLibViz::SwingPendulumView;
  RLLibViz::ViewBase* targetView = new RLLibViz::SwingPendulumView;
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
  window->setWindowTitle("RLLibViz (SwingPendulumModel Off-Policy)");
  window->show();

  RLLibViz::ModelBase* model = new RLLibViz::SwingPendulumModel2;
  model->setWindow(window);
  model->initialize();

  RLLibViz::ModelThread* thread = new RLLibViz::ModelThread;
  thread->setModel(model);
  thread->start();

  return a.exec();
}


