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
#include "Vec.h"
#include "Matrix.h"
#include "Control.h"
#include "RL.h"

using namespace RLLib;

namespace RLLibViz
{

class Window;

class ModelBase: public QObject
{
  Q_OBJECT

  private:
    RLLib::Matrix* valueFunction;

  public:
    explicit ModelBase();
    virtual ~ModelBase();

  public:
  signals:
    void signal_draw(QWidget* that);
    void signal_add(QWidget* that, const Vec& p, const Vec& q);
    void signal_add(QWidget* that, const Matrix* mat, double const& minV, double const& maxV);

  public:
    virtual void doLearning(Window* window) =0;
    virtual void doEvaluation(Window* window) =0;

  protected:
    virtual void updateValueFunction(Window* window, const RLLib::Control<double>* control,
        const RLLib::Ranges<double>* ranges, const bool& isEndingOfEpisode, const int& index);

};

}  // namespace RLLibViz

#endif /* MODELBASE_H_ */
