#include <QtGui/QApplication>
#include "Model.h"
#include "Window.h"

int main(int argc, char *argv[])
{
  QApplication a(argc, argv);
  Window window;
  window.setWindowTitle("RLLibViz");
  Model model;
  model.setWindow(&window);
  window.show();
  return a.exec();
}
