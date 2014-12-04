/*
 * EvaluationThread.h
 *
 *  Created on: Dec 3, 2014
 *      Author: sam
 */

#ifndef EVALUATIONTHREAD_H_
#define EVALUATIONTHREAD_H_

#include "ThreadBase.h"

namespace RLLibViz
{

class EvaluationThread: public ThreadBase
{
  Q_OBJECT

  public:
    explicit EvaluationThread(QObject* parent = 0);
    virtual ~EvaluationThread();
    void doWork(ModelBase* modelBase, Window* window);
};

}  // namespace RLLibViz

#endif /* EVALUATIONTHREAD_H_ */
