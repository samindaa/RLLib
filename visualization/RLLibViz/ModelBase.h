/*
 * ModelBase.h
 *
 *  Created on: Oct 11, 2013
 *      Author: sam
 */

#ifndef MODELBASE_H_
#define MODELBASE_H_

#include <QObject>
#include <map>
//
#include "Window.h"
#include "Vec.h"
//
// From the RLLib
#include "Vector.h"
#include "Trace.h"
#include "Projector.h"
#include "ControlAlgorithm.h"
#include "StateToStateAction.h"
#include "RL.h"
#include "FourierBasis.h"
//
#include "Eigen/Dense"

using namespace RLLib;
using namespace Eigen;

namespace RLLibViz
{

class Window;

class ModelBase: public QObject
{
  Q_OBJECT

  protected:
    MatrixXd valueFunction2D;
    typedef std::map<int, RLRunner<double>*> Simulators;
    Simulators simulators;

  public:
    explicit ModelBase();
    virtual ~ModelBase();

  public:
  signals:
    void signal_draw(QWidget* that);
    void signal_add(QWidget* that, const Vec& p, const Vec& q);
    void signal_add(QWidget* that, const MatrixXd& mat);

  public:
    virtual void doLearning(Window* window) =0;
    virtual void doEvaluation(Window* window) =0;

  protected:
    virtual void updateValueFunction(Window* window, const RLLib::Control<double>* control,
        const TRStep<double>* output, const RLLib::Ranges<double>* ranges, const bool& isEndingOfEpisode,
        const int& index);

};

}  // namespace RLLibViz

#endif /* MODELBASE_H_ */
