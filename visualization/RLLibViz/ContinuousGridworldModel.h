/*
 * ContinuousGridworldModel.h
 *
 *  Created on: Oct 11, 2013
 *      Author: sam
 */

#ifndef CONTINUOUSGRIDWORLDMODEL_H_
#define CONTINUOUSGRIDWORLDMODEL_H_

#include "ModelBase.h"

// From the simulation
#include "ContinuousGridworld.h"

namespace RLLibViz
{

class ContinuousGridworldModel: public ModelBase
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
    ContinuousGridworldModel();
    virtual ~ContinuousGridworldModel();

  protected:
    void doLearning(Window* window);
    void doEvaluation(Window* window);
    void step(Window* window, RLLib::RLRunner<double>* runner);
};

}  // namespace RLLibViz

#endif /* CONTINUOUSGRIDWORLDMODEL_H_ */
