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
#include "Matrix.h"

using namespace RLLib;

namespace RLLibViz
{

class ModelBase: public QObject
{
  Q_OBJECT

  protected:
    Window* window;
  public:
    explicit ModelBase(QObject *parent = 0);
    virtual ~ModelBase();
    void setWindow(Window* window);
    void run();
    virtual void initialize();

  public:
  signals:
    void signal_draw(QWidget* that);
    void signal_add(QWidget* that, const Vec& p, const Vec& q);
    void signal_add(QWidget* that, const Matrix* mat, double const& minV, double const& maxV);

  protected:
    virtual void doWork() =0;

};

}  // namespace RLLibViz

#endif /* MODELBASE_H_ */
