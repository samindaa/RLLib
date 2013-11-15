/*
 * ContinuousGridworldModel.h
 *
 *  Created on: Oct 11, 2013
 *      Author: sam
 */

#ifndef CONTINUOUSGRIDWORLDMODEL_H_
#define CONTINUOUSGRIDWORLDMODEL_H_

#include "ModelBase.h"

// From the RLLib
#include "Vector.h"
#include "Trace.h"
#include "Projector.h"
#include "ControlAlgorithm.h"
#include "StateToStateAction.h"
#include "Matrix.h"

// From the simulation
#include "ContinuousGridworld.h"
#include "RL.h"

// View
#include "ContinuousGridworldView.h"

//
#include <vector>
#include <tr1/unordered_map>

using namespace RLLib;
namespace RLLibViz
{

class ContinuousGridworldModel: public ModelBase
{
    Q_OBJECT

  protected:
// RLLib
    RLProblem<double>* behaviourEnvironment;
    RLProblem<double>* evaluationEnvironment;
    Hashing* hashing;
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
    ContinuousGridworldModel(QObject *parent = 0);
    virtual ~ContinuousGridworldModel();
    void initialize();
  protected:
    void doWork();
};

}  // namespace RLLibViz

#endif /* CONTINUOUSGRIDWORLDMODEL_H_ */
