/*
 * MountainCarModel.h
 *
 *  Created on: Oct 17, 2013
 *      Author: sam
 */

#ifndef MOUNTAINCARMODEL_H_
#define MOUNTAINCARMODEL_H_

#include "ModelBase.h"

// From the RLLib
#include "Vector.h"
#include "Trace.h"
#include "Projector.h"
#include "ControlAlgorithm.h"
#include "StateToStateAction.h"
#include "Matrix.h"

// From the simulation
#include "MountainCarModel.h"
#include "RL.h"

// View
#include "MountainCar.h"
#include "MountainCarView.h"

//
#include <vector>
#include <tr1/unordered_map>

using namespace RLLib;

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

    Simulator<double>* learningRunner;
    Simulator<double>* evaluationRunner;

    std::tr1::unordered_map<int, Simulator<double>*> simulators;

  public:
    MountainCarModel();
    virtual ~MountainCarModel();

  protected:
    void doLearning(Window* window);
    void doEvaluation(Window* window);
};

}  // namespace RLLibViz

#endif /* MOUNTAINCARMODEL_H_ */
