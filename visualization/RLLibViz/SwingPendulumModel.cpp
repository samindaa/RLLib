/*
 * SwingPendulumModel.cpp
 *
 *  Created on: Oct 14, 2013
 *      Author: sam
 */

#include "SwingPendulumModel.h"

using namespace RLLibViz;

SwingPendulumModel::SwingPendulumModel()
{
  random = new Random<double>;
  problem = new SwingPendulum<double>;
  hashing = new MurmurHashing<double>(random, 1000);
  projector = new TileCoderHashing<double>(hashing, problem->dimension(), 10, 10, false);
  toStateAction = new StateActionTilings<double>(projector, problem->getContinuousActions());

  alpha_v = 0.1 / projector->vectorNorm();
  alpha_u = 0.001 / projector->vectorNorm();
  alpha_r = .0001;
  gamma = 1.0;
  lambda = 0.5;

  critice = new ATrace<double>(projector->dimension());
  critic = new TDLambda<double>(alpha_v, gamma, lambda, critice);

  policyDistribution = new NormalDistributionScaled<double>(random, problem->getContinuousActions(),
      0, 1.0, projector->dimension());
  policyRange = new Range<double>(-2.0, 2.0);
  problemRange = new Range<double>(-2.0, 2.0);
  acting = new ScaledPolicyDistribution<double>(problem->getContinuousActions(), policyDistribution,
      policyRange, problemRange);

  actore1 = new ATrace<double>(projector->dimension());
  actore2 = new ATrace<double>(projector->dimension());
  actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore1);
  actoreTraces->push_back(actore2);
  actor = new ActorLambda<double>(alpha_u, gamma, lambda, acting, actoreTraces);

  control = new AverageRewardActorCritic<double>(critic, actor, projector, toStateAction, alpha_r);
  agent = new LearnerAgent<double>(control);
  simulator = new RLRunner<double>(agent, problem, 5000);
  simulator->setVerbose(false);

}

SwingPendulumModel::~SwingPendulumModel()
{
  delete random;
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
  delete agent;
  delete simulator;
}

void SwingPendulumModel::doLearning(Window* window)
{
  simulator->step();
  if (simulator->isEndingOfEpisode())
  {
    emit signal_add(window->plotVector[0], Vec(simulator->timeStep, 0),
        Vec(simulator->episodeR, 0));
    emit signal_draw(window->plotVector[0]);
  }

  emit signal_add(window->problemVector[0],
      Vec(simulator->getRLProblem()->getTRStep()->observation_tp1->getEntry(0),
          simulator->getRLProblem()->getTRStep()->observation_tp1->getEntry(1)),
      Vec(0.0, 0.0, 0.0, 1.0));
  emit signal_draw(window->problemVector[0]);

  updateValueFunction(window, control, problem->getTRStep(), problem->getObservationRanges(),
      simulator->isEndingOfEpisode(), 0);

}

void SwingPendulumModel::doEvaluation(Window* window)
{
  // Nothing
}
