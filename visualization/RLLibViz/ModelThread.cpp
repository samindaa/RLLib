/*
 * ModelThread.cpp
 *
 *  Created on: Oct 11, 2013
 *      Author: sam
 */

#include "ModelThread.h"

using namespace RLLibViz;

ModelThread::ModelThread(QObject* parent) :
    QThread(parent), model(0)
{
}

ModelThread::~ModelThread()
{
}

void ModelThread::setModel(ModelBase* model)
{
  this->model = model;
}

void ModelThread::run()
{
  if (model != 0)
  {
    while (true) /*fixMe*/
      model->run();
  }
  else
    std::cerr << "Model is invalid" << std::endl;
}

