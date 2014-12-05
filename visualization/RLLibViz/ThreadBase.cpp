/*
 * ThreadBase.cpp
 *
 *  Created on: Dec 3, 2014
 *      Author: sam
 */

#include "ThreadBase.h"
#include <unistd.h>

using namespace RLLibViz;

ThreadBase::ThreadBase(QObject* parent) :
    QThread(parent), model(0), window(0), isActive(false), simulationSpeed(100/*maximum*/)
{
}

ThreadBase::~ThreadBase()
{
}

void ThreadBase::setActive(const bool& isActive)
{
  this->isActive = isActive;
}

void ThreadBase::setModel(ModelBase* model)
{
  this->model = model;
}

void ThreadBase::setWindow(Window* window)
{
  this->window = window;
}

void ThreadBase::setSimulationSpeed(const int& simulationSpeed)
{
  this->simulationSpeed = simulationSpeed;
}

void ThreadBase::run()
{
  while (true)
  {
    usleep((100 - simulationSpeed) * 500);
    doWork(model, window);
  }
}
