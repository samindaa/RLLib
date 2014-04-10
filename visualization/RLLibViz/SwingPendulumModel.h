/*
 * SwingPendulumModel.h
 *
 *  Created on: Oct 14, 2013
 *      Author: sam
 */

#ifndef SWINGPENDULUMMODEL_H_
#define SWINGPENDULUMMODEL_H_

#include "ModelBase.h"

// From the RLLib
#include "Vector.h"
#include "Trace.h"
#include "Projector.h"
#include "ControlAlgorithm.h"
#include "StateToStateAction.h"

// From the simulation
#include "SwingPendulum.h"
#include "RL.h"

#include <unistd.h>

using namespace RLLib;

namespace RLLibViz
{

class SwingPendulumModel: public ModelBase
{
    Q_OBJECT

  protected:
    Random<double>* random;
    RLProblem<double>* problem;
    Hashing* hashing;
    Projector<double>* projector;
    StateToStateAction<double>* toStateAction;

    double alpha_v;
    double alpha_u;
    double alpha_r;
    double gamma;
    double lambda;

    Trace<double>* critice;
    TDLambda<double>* critic;

    PolicyDistribution<double>* policyDistribution;
    Range<double>* policyRange;
    Range<double>* problemRange;
    PolicyDistribution<double>* acting;

    Trace<double>* actore1;
    Trace<double>* actore2;
    Traces<double>* actoreTraces;
    ActorOnPolicy<double>* actor;

    OnPolicyControlLearner<double>* control;
    RLAgent<double>* agent;

    Simulator<double>* simulator;

    RLLib::Matrix* valueFunction;

  public:
    SwingPendulumModel(QObject *parent = 0);
    virtual ~SwingPendulumModel();
    void initialize();
  protected:
    void doWork();
};

}  // namespace RLLibViz

#endif /* SWINGPENDULUMMODEL_H_ */
