/*
 * ModelBase.h
 *
 *  Created on: Oct 11, 2013
 *      Author: sam
 */

#ifndef MODELBASE_H_
#define MODELBASE_H_

#include <QObject>
#include "Window.h"

namespace RLLibViz
{

class ModelBase: public QObject
{
  protected:
    Window* window;
  public:
    explicit ModelBase(QObject *parent = 0);
    virtual ~ModelBase();
    void setWindow(Window* window);
    void run();
    virtual void initialize();

  protected:
    virtual void doWork() =0;

};

}  // namespace RLLibViz

#endif /* MODELBASE_H_ */
