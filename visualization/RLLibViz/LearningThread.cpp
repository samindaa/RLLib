/*
 * LearningThread.cpp
 *
 *  Created on: Dec 3, 2014
 *      Author: sam
 */

#include "LearningThread.h"

using namespace RLLibViz;

LearningThread::LearningThread(QObject* parent) :
    ThreadBase(parent)
{
}

LearningThread::~LearningThread()
{
}

void LearningThread::doWork(ModelBase* modelBase, Window* window)
{
  if (isActive && modelBase && window)
    modelBase->doLearning(window);
}

