#include "Window.h"
#include <QApplication>
#include <QTime>
#include <iostream>

using namespace RLLibViz;

Window::Window(QWidget *parent) :
    QWidget(parent), valueFunctionView(0), grid(0), colsA(0), colsB(0)
{
  grid = new QGridLayout(this);
  setLayout(grid);
}

Window::~Window()
{
}

void Window::addView(ViewBase* view)
{
  views.push_back(view);
  ((QGridLayout*) grid)->addWidget(view, 0, colsA++);
}

void Window::addPlot(ViewBase* view)
{
  plots.push_back(view);
  ((QGridLayout*) grid)->addWidget(view, 1, colsB++);
}

void Window::setValueFunctionView(ViewBase* valueFunctionView)
{
  this->valueFunctionView = valueFunctionView;
  ((QGridLayout*) grid)->addWidget(this->valueFunctionView, 0, colsA++);
}

void Window::keyPressEvent(QKeyEvent* event)
{
  if (event->key() == Qt::Key_Escape)
  {
    std::cout << "Window::Quit()" << std::endl;
    qApp->quit();
  }
}
