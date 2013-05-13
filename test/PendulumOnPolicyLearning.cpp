/*
 * PendulumOnPolicyLearning.cpp
 *
 *  Created on: Mar 11, 2013
 *      Author: sam
 */

#include <iostream>
#include "PendulumOnPolicyLearning.h"

using namespace std;
using namespace RLLib;

ActorCriticOnPolicyControlLearnerPendulumTest::ActorCriticOnPolicyControlLearnerPendulumTest()
{
  problem = new SwingPendulum;
  projector = new TileCoderHashing<double, float>(1000, 10, true);
  toStateAction = new StateActionTilings<double, float>(projector,
      &problem->getContinuousActionList());

  alpha_v = alpha_u = alpha_r = gamma = lambda = 0;

  criticE = new ATrace<double>(projector->dimension());
  critic = 0;

  policyDistribution = new NormalDistributionScaled<double>(0, 1.0,
      projector->dimension(), &problem->getContinuousActionList());

  actorMuE = new ATrace<double>(projector->dimension());
  actorSigmaE = new ATrace<double>(projector->dimension());
  actorTraces = new Traces<double>();
  actorTraces->push_back(actorMuE);
  actorTraces->push_back(actorSigmaE);
  actor = 0;

  control = 0;
  sim = 0;
}

ActorCriticOnPolicyControlLearnerPendulumTest::~ActorCriticOnPolicyControlLearnerPendulumTest()
{
  delete problem;
  delete projector;
  delete toStateAction;
  delete criticE;
  delete policyDistribution;
  delete actorMuE;
  delete actorSigmaE;
  delete actorTraces;
  deleteObjects();
}

void ActorCriticOnPolicyControlLearnerPendulumTest::evaluate()
{
  if (sim)
  {
    cout << sim->episodeR << " " << sim->time << " "
        << (sim->episodeR / sim->time) << endl;
  }
}

void ActorCriticOnPolicyControlLearnerPendulumTest::deleteObjects()
{
  if (critic)
    delete critic;
  critic = 0;
  if (actor)
    delete actor;
  actor = 0;
  if (control)
    delete control;
  control = 0;
  if (sim)
    delete sim;
  sim = 0;
}

void ActorCriticOnPolicyControlLearnerPendulumTest::testRandom()
{
  alpha_v = alpha_u = alpha_r = gamma = lambda = 0;
  critic = new TD<double>(alpha_v, gamma, projector->dimension());
  actor = new Actor<double, float>(alpha_u, policyDistribution);
  control = new ActorCritic<double, float>(critic, actor, projector,
      toStateAction);

  sim = new Simulator<double, float>(control, problem);
  sim->run(1, 5000, 50, false, false);

  evaluate();
  deleteObjects();
}

void ActorCriticOnPolicyControlLearnerPendulumTest::testActorCritic()
{

  gamma = 1.0;
  alpha_v = 0.5 / projector->vectorNorm();
  critic = new TD<double>(alpha_v, gamma, projector->dimension());
  alpha_u = 0.05 / projector->vectorNorm();
  actor = new Actor<double, float>(alpha_u, policyDistribution);
  control = new AverageRewardActorCritic<double, float>(critic, actor,
      projector, toStateAction, 0.01);

  sim = new Simulator<double, float>(control, problem);
  sim->run(1, 5000, 50, false, false);
  sim->computeValueFunction();
  evaluate();
  deleteObjects();
}

void ActorCriticOnPolicyControlLearnerPendulumTest::testActorCriticWithEligiblity()
{
  gamma = 1.0;
  lambda = 0.5;
  alpha_v = 0.1 / projector->vectorNorm();
  critic = new TDLambda<double>(alpha_v, gamma, lambda, criticE);
  alpha_u = 0.05 / projector->vectorNorm();
  actor = new ActorLambda<double, float>(alpha_u, gamma, lambda,
      policyDistribution, actorTraces);
  control = new AverageRewardActorCritic<double, float>(critic, actor,
      projector, toStateAction, 0.01);

  sim = new Simulator<double, float>(control, problem);
  sim->run(1, 5000, 50, false, false);
  sim->computeValueFunction();
  evaluate();
  deleteObjects();
}

void ActorCriticOnPolicyControlLearnerPendulumTest::run()
{
  testRandom();
  testActorCritic();
  testActorCriticWithEligiblity();
}
