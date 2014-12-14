/*
 * SwingPendulumModel2.h
 *
 *  Created on: Nov 20, 2013
 *      Author: sam
 */

#ifndef SWINGPENDULUMMODEL2_H_
#define SWINGPENDULUMMODEL2_H_

#include "ModelBase.h"

// From the simulation
#include "SwingPendulum.h"

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

    RLRunner<double>* learningRunner;
    RLRunner<double>* evaluationRunner;

    std::map<int, RLRunner<double>*> simulators;

  public:
    SwingPendulumModel2();
    virtual ~SwingPendulumModel2();

  protected:
    void doLearning(Window* window);
    void doEvaluation(Window* window);
};

}  // namespace RLLibViz

#endif /* SWINGPENDULUMMODEL2_H_ */
