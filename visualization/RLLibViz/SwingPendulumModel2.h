/*
 * SwingPendulumModel2.h
 *
 *  Created on: Nov 20, 2013
 *      Author: sam
 */

#ifndef SWINGPENDULUMMODEL2_H_
#define SWINGPENDULUMMODEL2_H_

#include "ModelBase.h"

// From the RLLib
#include "Vector.h"
#include "Trace.h"
#include "Projector.h"
#include "ControlAlgorithm.h"
#include "StateToStateAction.h"
#include "Matrix.h"

// From the simulation
#include "SwingPendulum.h"
#include "RL.h"

// View
#include "SwingPendulumView.h"

//
#include <vector>
#include <tr1/unordered_map>

using namespace RLLib;
namespace RLLibViz
{

class SwingPendulumModel2: public ModelBase
{
  Q_OBJECT

  protected:
// RLLib
    Random<double>* random;
    RLProblem<double>* behaviourEnvironment;
    RLProblem<double>* evaluationEnvironment;
    Hashing<double>* hashing;
    Projector<double>* projector;
    StateToStateAction<double>* toStateAction;

    double alpha_v;
    double alpha_w;
    double gamma;
    double lambda;

    Trace<double>* critice;
    GTDLambda<double>* critic;

    double alpha_u;

    PolicyDistribution<double>* target;

    Trace<double>* actore;
    Traces<double>* actoreTraces;
    ActorOffPolicy<double>* actor;

    Policy<double>* behavior;
    OffPolicyControlLearner<double>* control;

    RLAgent<double>* learningAgent;
    RLAgent<double>* evaluationAgent;

    Simulator<double>* learningRunner;
    Simulator<double>* evaluationRunner;

    std::tr1::unordered_map<int, Simulator<double>*> simulators;

    RLLib::Matrix* valueFunction;

  public:
    SwingPendulumModel2(QObject *parent = 0);
    virtual ~SwingPendulumModel2();
    void initialize();
  protected:
    void doWork();
};

}  // namespace RLLibViz

#endif /* SWINGPENDULUMMODEL2_H_ */
