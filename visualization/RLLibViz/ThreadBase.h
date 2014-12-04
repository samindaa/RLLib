/*
 * ThreadBase.h
 *
 *  Created on: Dec 3, 2014
 *      Author: sam
 */

#ifndef THREADBASE_H_
#define THREADBASE_H_

#include <QThread>
//
#include "ModelBase.h"
#include "Window.h"

namespace RLLibViz
{

class ThreadBase: public QThread
{
  Q_OBJECT
  protected:
    ModelBase* model;
    Window* window;
    bool isActive;
    int simulationSpeed;

  public:
    explicit ThreadBase(QObject* parent = 0);
    virtual ~ThreadBase();
    void setActive(const bool& isActive);
    void setModel(ModelBase* model);
    void setWindow(Window* window);
    void setSimulationSpeed(const int& simulationSpeed);
    virtual void doWork(ModelBase* modelBase, Window* window) =0;
    void run();
};

}  // namespace RLLibViz

#endif /* THREADBASE_H_ */
