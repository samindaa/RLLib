/*
 * ContinuousGridworldModel.cpp
 *
 *  Created on: Oct 11, 2013
 *      Author: sam
 */

#include "ContinuousGridworldModel.h"

using namespace RLLibViz;

ContinuousGridworldModel::ContinuousGridworldModel(QObject *parent) :
    ModelBase(parent)
{
  // RLLib:
  behaviourEnvironment = new ContinuousGridworld;
  evaluationEnvironment = new ContinuousGridworld;
  hashing = new MurmurHashing;
  projector = new TileCoderHashing<double, float>(1000000, 10, true, hashing);
  toStateAction = new StateActionTilings<double, float>(projector,
      &behaviourEnvironment->getDiscreteActionList());

  alpha_v = 0.1 / projector->vectorNorm();
  alpha_w = 0.0; //0.0001 / projector->vectorNorm();
  gamma = 0.99;
  lambda = 0.4;
  critice = new ATrace<double>(projector->dimension());
  critic = new GTDLambda<double>(alpha_v, alpha_w, gamma, lambda, critice);

  alpha_u = 0.001 / projector->vectorNorm();

  target = new BoltzmannDistribution<double>(projector->dimension(),
      &behaviourEnvironment->getDiscreteActionList());

  actore = new ATrace<double>(projector->dimension());
  actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore);
  actor = new ActorLambdaOffPolicy<double, float>(alpha_u, gamma, lambda, target, actoreTraces);

  behavior = new RandomPolicy<double>(&behaviourEnvironment->getDiscreteActionList());
  control = new OffPAC<double, float>(behavior, critic, actor, toStateAction, projector, gamma);

  learningRunner = new Simulator<double, float>(control, behaviourEnvironment, 5000);
  evaluationRunner = new Simulator<double, float>(control, evaluationEnvironment, 5000);
  learningRunner->setVerbose(false);
  evaluationRunner->setEvaluate(true);

  simulators.insert(std::make_pair(simulators.size(), learningRunner));
  simulators.insert(std::make_pair(simulators.size(), evaluationRunner));

}

ContinuousGridworldModel::~ContinuousGridworldModel()
{
  delete behaviourEnvironment;
  delete evaluationEnvironment;
  delete hashing;
  delete projector;
  delete toStateAction;
  delete critice;
  delete critic;
  delete actore;
  delete actoreTraces;
  delete actor;
  delete behavior;
  delete target;
  delete control;
  delete learningRunner;
  delete evaluationRunner;
}

void ContinuousGridworldModel::initialize()
{
  ModelBase::initialize();
}

void ContinuousGridworldModel::doWork()
{
  for (std::tr1::unordered_map<int, Simulator<double, float>*>::iterator i = simulators.begin();
      i != simulators.end(); ++i)
    i->second->step();

  for (std::tr1::unordered_map<int, Simulator<double, float>*>::iterator i = simulators.begin();
      i != simulators.end(); ++i)
  {
    if (i->second->isBeginingOfEpisode())
      window->views[i->first]->draw();
    else
      window->views[i->first]->add(
          Vec(i->second->getEnvironment()->getTRStep().o_tp1->at(0),
              i->second->getEnvironment()->getTRStep().o_tp1->at(1)));
  }

}
