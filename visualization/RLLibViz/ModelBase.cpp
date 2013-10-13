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
    (*iter)->initialize();
}

void ModelBase::run()
{
  if (window != 0 && !window->views.empty())
    doWork();
  else
    std::cerr << " Model is invalid" << std::endl;
}

