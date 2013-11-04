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
      behaviourEnvironment->getDiscreteActionList());

  alpha_v = 0.1 / projector->vectorNorm();
  alpha_w = 0.0001 / projector->vectorNorm();
  gamma = 0.99;
  lambda = 0.4;
  critice = new ATrace<double>(projector->dimension());
  critic = new GTDLambda<double>(alpha_v, alpha_w, gamma, lambda, critice);

  alpha_u = 0.001 / projector->vectorNorm();

  target = new BoltzmannDistribution<double>(projector->dimension(),
      behaviourEnvironment->getDiscreteActionList());

  actore = new ATrace<double>(projector->dimension());
  actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore);
  actor = new ActorLambdaOffPolicy<double, float>(alpha_u, gamma, lambda, target, actoreTraces);

  behavior = new RandomPolicy<double>(behaviourEnvironment->getDiscreteActionList());
  control = new OffPAC<double, float>(behavior, critic, actor, toStateAction, projector, gamma);

  learningRunner = new Simulator<double, float>(control, behaviourEnvironment, 5000);
  evaluationRunner = new Simulator<double, float>(control, evaluationEnvironment, 5000);
  learningRunner->setVerbose(false);
  evaluationRunner->setEvaluate(true);
  evaluationRunner->setVerbose(false);

  simulators.insert(std::make_pair(simulators.size(), learningRunner));
  simulators.insert(std::make_pair(simulators.size(), evaluationRunner));

  valueFunction = new Matrix(101, 101); // << Fixed for 0:0.1:10

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
  delete valueFunction;
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
    if (i->second->isEndingOfEpisode())
    {
      emit signal_draw(window->views[i->first]);
      emit signal_add(window->plots[i->first], Vec(i->second->timeStep, 0),
          Vec(i->second->episodeR, 0));
      emit signal_draw(window->plots[i->first]);
    }
    else
      emit signal_add(window->views[i->first],
          Vec(i->second->getEnvironment()->getObservations()->at(0),
              i->second->getEnvironment()->getObservations()->at(1)), Vec(0.0, 0.0, 0.0, 1.0));
  }

  // Value function
  if (evaluationRunner->isEndingOfEpisode() && window->vfuns.size() > 1)
  {
    RLLib::PVector<float> x_t(2);
    double maxValue = 0, minValue = 0;
    float y = 0;
    for (int i = 0; i < valueFunction->rows(); i++)
    {
      float x = 0;
      for (int j = 0; j < valueFunction->cols(); j++)
      {
        x_t[0] = y;
        x_t[1] = x;
        double v = control->computeValueFunction(&x_t);
        valueFunction->at(i, j) = v;
        if (v > maxValue)
          maxValue = v;
        if (v < minValue)
          minValue = v;
        x += 0.1;
      }
      y += 0.1;
    }
    //out.close();
    emit signal_add(window->vfuns[1], valueFunction, minValue, maxValue);
    emit signal_draw(window->vfuns[1]);
  }
}
