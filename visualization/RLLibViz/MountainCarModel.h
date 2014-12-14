/*
 * MountainCarModel.h
 *
 *  Created on: Oct 17, 2013
 *      Author: sam
 */

#ifndef MOUNTAINCARMODEL_H_
#define MOUNTAINCARMODEL_H_

#include "ModelBase.h"

// From the simulation
#include "MountainCar.h"

namespace RLLibViz
{

class MountainCarModel: public ModelBase
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

  public:
    MountainCarModel();
    virtual ~MountainCarModel();

  protected:
    void doLearning(Window* window);
    void doEvaluation(Window* window);
};

}  // namespace RLLibViz

#endif /* MOUNTAINCARMODEL_H_ */
