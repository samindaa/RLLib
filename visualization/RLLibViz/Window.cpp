#include "Window.h"

#include <QApplication>
#include <QTime>
#include <iostream>

using namespace RLLibViz;

Window::Window(QWidget* parent) :
    QWidget(parent), windowLayout(new WindowLayout)
{
  setLayout(windowLayout);
  setVisible(false);
}

Window::~Window()
{
}

void Window::initialize(ModelBase* modelBase)
{
  for (WindowVector::iterator iter = problemVector.begin(); iter != problemVector.end(); ++iter)
  {
    ViewBase* view = *iter;
    view->initialize();
    connect(modelBase, SIGNAL(signal_draw(QWidget*)), view, SLOT(draw(QWidget*)));
    connect(modelBase, SIGNAL(signal_add(QWidget*, const Vec&, const Vec&)), view,
        SLOT(add(QWidget*,const Vec&, const Vec&)));
  }

  for (WindowVector::iterator iter = plotVector.begin(); iter != plotVector.end(); ++iter)
  {
    ViewBase* view = *iter;
    view->initialize();
    connect(modelBase, SIGNAL(signal_draw(QWidget*)), view, SLOT(draw(QWidget*)));
    connect(modelBase, SIGNAL(signal_add(QWidget*, const Vec&, const Vec&)), view,
        SLOT(add(QWidget*,const Vec&, const Vec&)));
  }

  for (WindowVector::iterator iter = valueFunctionVector.begin(); iter != valueFunctionVector.end();
      ++iter)
  {
    ViewBase* view = *iter;
    view->initialize();
    connect(modelBase, SIGNAL(signal_add(QWidget*, const MatrixXd&)), view,
        SLOT(add(QWidget*, const MatrixXd&)));
    connect(modelBase, SIGNAL(signal_add(QWidget*, const Vec&, const Vec&)), view,
        SLOT(add(QWidget*,const Vec&, const Vec&)));
  }
}

bool Window::empty() const
{
  return problemVector.empty();
}

void Window::newLayout()
{
  if (this->layout())
  {
    for (WindowVector::iterator iter = problemVector.begin(); iter != problemVector.end(); ++iter)
      this->layout()->removeWidget(*iter);
    for (WindowVector::iterator iter = plotVector.begin(); iter != plotVector.end(); ++iter)
      this->layout()->removeWidget(*iter);
    for (WindowVector::iterator iter = valueFunctionVector.begin();
        iter != valueFunctionVector.end(); ++iter)
      this->layout()->removeWidget(*iter);
    delete this->layout();
  }

  //if (windowLayout)
  //  delete windowLayout;
  windowLayout = new WindowLayout;
  setLayout(windowLayout);

  for (WindowVector::iterator iter = problemVector.begin(); iter != problemVector.end(); ++iter)
    windowLayout->addTopWidget(*iter);

  for (WindowVector::iterator iter = plotVector.begin(); iter != plotVector.end(); ++iter)
    windowLayout->addCenterWidget(*iter);

  for (WindowVector::iterator iter = valueFunctionVector.begin(); iter != valueFunctionVector.end();
      ++iter)
    windowLayout->addBottomWidget(*iter);
}

void Window::addProblemView(ViewBase* view)
{
  problemVector.push_back(view);
}

void Window::addPlotView(ViewBase* view)
{
  plotVector.push_back(view);
}

void Window::addValueFunctionView(ViewBase* view)
{
  valueFunctionVector.push_back(view);
}

void Window::keyPressEvent(QKeyEvent* event)
{
  if (event->key() == Qt::Key_Escape)
  {
    std::cout << "Window::Quit()" << std::endl;
    qApp->quit();
  }
}
