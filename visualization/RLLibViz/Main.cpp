#include <QApplication>
#include "ContinuousGridworldModel.h"
#include "ContinuousGridworldView.h"
#include "PlotView.h"
#include "ModelThread.h"
#include "Window.h"

int main(int argc, char *argv[])
{
  QApplication a(argc, argv);

  RLLibViz::ViewBase* behaviorView = new RLLibViz::ContinuousGridworldView;
  RLLibViz::ViewBase* targetView = new RLLibViz::ContinuousGridworldView;
  RLLibViz::ViewBase* behaviorPlot = new RLLibViz::PlotView;
  RLLibViz::ViewBase* targetPlot = new RLLibViz::PlotView;


  RLLibViz::Window* window = new RLLibViz::Window;
  window->addView(behaviorView);
  window->addView(targetView);
  window->addPlot(behaviorPlot);
  window->addPlot(targetPlot);
  window->setWindowTitle("RLLibViz");
  window->show();

  RLLibViz::ModelBase* model = new RLLibViz::ContinuousGridworldModel;
  model->setWindow(window);
  model->initialize();

  RLLibViz::ModelThread* thread = new RLLibViz::ModelThread;
  thread->setModel(model);
  thread->start();

  return a.exec();
}
