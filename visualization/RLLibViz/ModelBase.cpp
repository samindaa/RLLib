/*
 * ModelBase.cpp
 *
 *  Created on: Oct 11, 2013
 *      Author: sam
 */
#include "ModelBase.h"

using namespace RLLibViz;

ModelBase::ModelBase(QObject *parent) :
    QObject(parent), window(0)
{
}

ModelBase::~ModelBase()
{
}

void ModelBase::setWindow(Window* window)
{
  this->window = window;
}

void ModelBase::initialize()
{
  for (Window::Views::iterator iter = window->views.begin(); iter != window->views.end(); ++iter)
  {
    ViewBase* view = *iter;
    view->initialize();
    connect(this, SIGNAL(signal_draw(QWidget*)), view, SLOT(draw(QWidget*)));
    connect(this, SIGNAL(signal_add(QWidget*, const Vec&, const Vec&)), view,
    SLOT(add(QWidget*,const Vec&, const Vec&)));
  }

  for (Window::Plots::iterator iter = window->plots.begin(); iter != window->plots.end(); ++iter)
  {
    ViewBase* view = *iter;
    view->initialize();
    connect(this, SIGNAL(signal_draw(QWidget*)), view, SLOT(draw(QWidget*)));
    connect(this, SIGNAL(signal_add(QWidget*, const Vec&, const Vec&)), view,
    SLOT(add(QWidget*,const Vec&, const Vec&)));
  }

  for (Window::VFuns::iterator iter = window->vfuns.begin(); iter != window->vfuns.end(); ++iter)
  {
    ViewBase* view = *iter;
    view->initialize();
    connect(this,
    SIGNAL(signal_add(QWidget*, const Matrix*, double const&, double const&)), view,
    SLOT(add(QWidget*, const Matrix*, double const&, double const&)));
  }
}

void ModelBase::run()
{
  if (window != 0 && !window->views.empty())
    doWork();
  else
    std::cerr << " Model is invalid" << std::endl;
}

