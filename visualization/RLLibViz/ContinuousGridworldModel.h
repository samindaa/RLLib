/*
 * ContinuousGridworldModel.h
 *
 *  Created on: Oct 11, 2013
 *      Author: sam
 */

#ifndef CONTINUOUSGRIDWORLDMODEL_H_
#define CONTINUOUSGRIDWORLDMODEL_H_

#include "ModelBase.h"

#include <QObject>
#include <vector>
#include <QTimerEvent>

#include "Window.h"

// From the RLLib
#include "Vector.h"
#include "Trace.h"
#include "Projector.h"
#include "ControlAlgorithm.h"
#include "Representation.h"

// From the simulation
#include "ContinuousGridworld.h"
#include "Simulator.h"

// View
#include "ContinuousGridworldView.h"

//
#include <tr1/unordered_map>

using namespace RLLib;
namespace RLLibViz
{

class ContinuousGridworldModel: public ModelBase
{
  Q_OBJECT

  protected:
// RLLib
    Environment<float>* behaviourEnvironment;
    Environment<float>* evaluationEnvironment;
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

  public:
    ContinuousGridworldModel(QObject *parent = 0);
    virtual ~ContinuousGridworldModel();
    void initialize();
  protected:
    void doWork();
};

}  // namespace RLLibViz

#endif /* CONTINUOUSGRIDWORLDMODEL_H_ */
