/*
 * ModelThread.h
 *
 *  Created on: Oct 11, 2013
 *      Author: sam
 */

#ifndef MODELTHREAD_H_
#define MODELTHREAD_H_

#include <QThread>
#include "ModelBase.h"

namespace RLLibViz
{

class ModelThread: public QThread
{
Q_OBJECT
private:
  ModelBase* model;
  bool isActive;

public:
  explicit ModelThread(QObject * parent = 0);
  virtual ~ModelThread();
  void setModel(ModelBase* model);
  void setActive(const bool& isActive);

protected:
  void run();
};

}  // namespace RLLibViz

#endif /* MODELTHREAD_H_ */
