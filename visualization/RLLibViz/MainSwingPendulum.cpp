/*
 * MainSwingPendulum.cpp
 *
 *  Created on: Oct 14, 2013
 *      Author: sam
 */

#include <QApplication>
#include "SwingPendulumModel.h"
#include "SwingPendulumView.h"
#include "PlotView.h"
#include "ModelThread.h"
#include "Window.h"

using namespace RLLibViz;

Q_DECLARE_METATYPE(Vec)

int main(int argc, char *argv[])
{
  QApplication a(argc, argv);
  qRegisterMetaType<Vec>();

  RLLibViz::ViewBase* actingView = new RLLibViz::SwingPendulumView;
  RLLibViz::ViewBase* actingPlot = new RLLibViz::PlotView;

  RLLibViz::Window* window = new RLLibViz::Window;
  window->addView(actingView);
  window->addPlot(actingPlot);
  window->setWindowTitle("RLLibViz");
  window->show();

  RLLibViz::ModelBase* model = new RLLibViz::SwingPendulumModel;
  model->setWindow(window);
  model->initialize();

  RLLibViz::ModelThread* thread = new RLLibViz::ModelThread;
  thread->setModel(model);
  thread->start();

  return a.exec();
}

