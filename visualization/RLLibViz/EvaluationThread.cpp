/*
 * EvaluationThread.cpp
 *
 *  Created on: Dec 3, 2014
 *      Author: sam
 */

#include "EvaluationThread.h"

using namespace RLLibViz;

EvaluationThread::EvaluationThread(QObject* parent) :
    ThreadBase(parent)
{
}

EvaluationThread::~EvaluationThread()
{
}

void EvaluationThread::doWork(ModelBase* modelBase, Window* window)
{
  if (isActive && modelBase && window)
    modelBase->doEvaluation(window);
}

