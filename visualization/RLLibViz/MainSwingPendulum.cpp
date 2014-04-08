/*
 * MainSwingPendulum.cpp
 *
 *  Created on: Oct 14, 2013
 *      Author: sam
 */

#include <QApplication>
#include "SwingPendulumModel.h"
#include "SwingPendulumView.h"
#include "ValueFunctionView.h"
#include "PlotView.h"
#include "NULLView.h"
#include "ModelThread.h"
#include "Window.h"

using namespace RLLib;
using namespace RLLibViz;

Q_DECLARE_METATYPE(Vec)
Q_DECLARE_METATYPE(Matrix)

int main(int argc, char *argv[])
{
  QApplication a(argc, argv);
  qRegisterMetaType<Vec>();
  qRegisterMetaType<Matrix>();

  RLLibViz::ViewBase* actingView = new RLLibViz::SwingPendulumView;
  RLLibViz::ViewBase* actingPlot = new RLLibViz::PlotView("Target Policy");
  RLLibViz::ViewBase* valueFunctionView = new RLLibViz::ValueFunctionView;

  RLLibViz::Window* window = new RLLibViz::Window;
  window->addView(actingView);
  window->addPlot(actingPlot);
  window->addValueFunctionView(valueFunctionView);
  window->setWindowTitle("RLLibViz (SwingPendulum)");
  window->show();

  RLLibViz::ModelBase* model = new RLLibViz::SwingPendulumModel;
  model->setWindow(window);
  model->initialize();

  RLLibViz::ModelThread* thread = new RLLibViz::ModelThread;
  thread->setModel(model);
  thread->start();

  return a.exec();
}

