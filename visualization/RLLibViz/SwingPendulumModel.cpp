/*
 * SwingPendulumModel.cpp
 *
 *  Created on: Oct 14, 2013
 *      Author: sam
 */

#include "SwingPendulumModel.h"

using namespace RLLibViz;

SwingPendulumModel::SwingPendulumModel(QObject *parent) :
    ModelBase(parent)
{
  problem = new SwingPendulum;
  hashing = new MurmurHashing;
  projector = new TileCoderHashing<double, float>(1000, 10, false, hashing);
  toStateAction = new StateActionTilings<double, float>(projector,
      &problem->getContinuousActionList());

  alpha_v = 0.1 / projector->vectorNorm();
  alpha_u = 0.001 / projector->vectorNorm();
  alpha_r = .0001;
  gamma = 1.0;
  lambda = 0.5;

  critice = new ATrace<double>(projector->dimension());
  critic = new TDLambda<double>(alpha_v, gamma, lambda, critice);

  policyDistribution = new NormalDistributionScaled<double>(0, 1.0, projector->dimension(),
      &problem->getContinuousActionList());
  policyRange = new Range<double>(-2.0, 2.0);
  problemRange = new Range<double>(-2.0, 2.0);
  acting = new ScaledPolicyDistribution<double>(&problem->getContinuousActionList(),
      policyDistribution, policyRange, problemRange);

  actore1 = new ATrace<double>(projector->dimension());
  actore2 = new ATrace<double>(projector->dimension());
  actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore1);
  actoreTraces->push_back(actore2);
  actor = new ActorLambda<double, float>(alpha_u, gamma, lambda, acting, actoreTraces);

  control = new AverageRewardActorCritic<double, float>(critic, actor, projector, toStateAction,
      alpha_r);

  simulator = new Simulator<double, float>(control, problem, 5000);
}

SwingPendulumModel::~SwingPendulumModel()
{
  delete problem;
  delete hashing;
  delete projector;
  delete toStateAction;
  delete critice;
  delete critic;
  delete actore1;
  delete actore2;
  delete actoreTraces;
  delete actor;
  delete policyDistribution;
  delete policyRange;
  delete problemRange;
  delete acting;
  delete control;
  delete simulator;
}

void SwingPendulumModel::initialize()
{
  ModelBase::initialize();
}

void SwingPendulumModel::doWork()
{
  simulator->step();
  if (simulator->isEndingOfEpisode())
  {
    emit signal_add(window->plots[0], Vec(simulator->episodeR, 0), Vec(0.0, 0.0, 0.0, 1.0));
    emit signal_draw(window->plots[0]);
  }

  emit signal_add(window->views[0],
      Vec(simulator->getEnvironment()->getObservations().at(0),
          simulator->getEnvironment()->getObservations().at(1)), Vec(0.0, 0.0, 0.0, 1.0));
  emit signal_draw(window->views[0]);
}
