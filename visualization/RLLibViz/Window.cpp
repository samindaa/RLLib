#include "Window.h"
#include <QApplication>
#include <QTime>
#include <iostream>

using namespace RLLibViz;

Window::Window(QWidget *parent) :
    QWidget(parent), mainLayout(0)
{
  mainLayout = new QHBoxLayout;
  setLayout(mainLayout);
}

Window::~Window()
{
}

void Window::addView(ViewBase* view)
{
  views.push_back(view);
  mainLayout->addWidget(view);
}

void Window::keyPressEvent(QKeyEvent* event)
{
  if (event->key() == Qt::Key_Escape)
  {
    std::cout << "Window::Quit()" << std::endl;
    qApp->quit();
  }
}
