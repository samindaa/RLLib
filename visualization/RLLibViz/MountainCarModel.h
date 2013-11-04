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
#include "MCar2D.h"
#include "Simulator.h"

// View
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
    Environment<float>* behaviourEnvironment;
    Environment<float>* evaluationEnvironment;
    Hashing* hashing;
    Projector<double, float>* projector;
    StateToStateAction<double, float>* toStateAction;

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
    ActorOffPolicy<double, float>* actor;

    Policy<double>* behavior;
    OffPolicyControlLearner<double, float>* control;

    Simulator<double, float>* learningRunner;
    Simulator<double, float>* evaluationRunner;

    std::tr1::unordered_map<int, Simulator<double, float>*> simulators;

    RLLib::Matrix* valueFunction;

  public:
    MountainCarModel(QObject *parent = 0);
    virtual ~MountainCarModel();
    void initialize();
  protected:
    void doWork();
};

}  // namespace RLLibViz

#endif /* MOUNTAINCARMODEL_H_ */
