#include "Window.h"
#include <QApplication>
#include <QTime>
#include <iostream>

Window::Window(QWidget *parent) :
  QWidget(parent)
{
  QHBoxLayout* mainLayout = new QHBoxLayout;
  for (int i = 0; i < NB_AREA; i++)
    renders.push_back(new RenderArea);
  for (Renders::iterator iter = renders.begin(); iter != renders.end(); ++iter)
    mainLayout->addWidget(*iter);
  setLayout(mainLayout);
}

 void Window::keyPressEvent(QKeyEvent* event)
 {
   if (event->key() == Qt::Key_Escape)
   {
     std::cout << "Window::Quit()" << std::endl;
     qApp->quit();
   }
 }
