/*
 * LearningThread.h
 *
 *  Created on: Dec 3, 2014
 *      Author: sam
 */

#ifndef LEARNINGTHREAD_H_
#define LEARNINGTHREAD_H_

#include "ThreadBase.h"

namespace RLLibViz
{

class LearningThread: public ThreadBase
{
  Q_OBJECT

  public:
    explicit LearningThread(QObject* parent = 0);
    virtual ~LearningThread();
    void doWork(ModelBase* modelBase, Window* window);
};

}  // namespace RLLibViz



#endif /* LEARNINGTHREAD_H_ */
