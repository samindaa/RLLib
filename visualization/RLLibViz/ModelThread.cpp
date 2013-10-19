/*
 * ModelThread.cpp
 *
 *  Created on: Oct 11, 2013
 *      Author: sam
 */

#include "ModelThread.h"
#include <unistd.h>

using namespace RLLibViz;

ModelThread::ModelThread(QObject* parent) :
    QThread(parent), model(0), isActive(true)
{
}

ModelThread::~ModelThread()
{
}

void ModelThread::setModel(ModelBase* model)
{
  this->model = model;
}

void ModelThread::setActive(const bool& isActive)
{
  this->isActive = isActive;
}

void ModelThread::run()
{
  if (model != 0)
  {
    //sleep(30); // only for testing
    while (isActive)
      model->run();
  }
  else
    std::cerr << "Model is invalid" << std::endl;
}

