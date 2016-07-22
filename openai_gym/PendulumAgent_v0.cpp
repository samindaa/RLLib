/*
 * PendulumAgent.cpp
 *
 *  Created on: Jun 27, 2016
 *      Author: sabeyruw
 */

#include "PendulumAgent_v0.h"

OPENAI_AGENT_MAKE(PendulumAgent_v0)
PendulumAgent_v0::PendulumAgent_v0()
{
  problem = new Pendulum_v0;
  random = new RLLib::Random<double>;
  hashing = new RLLib::MurmurHashing<double>(random, 1000);
  projector = new RLLib::TileCoderHashing<double>(hashing, problem->dimension(), 10, 10, false);
  toStateAction = new RLLib::StateActionTilings<double>(projector, problem->getContinuousActions());

  alpha_v = 0.05 / projector->vectorNorm();
  alpha_u = 0.001 / projector->vectorNorm();
  alpha_r = .0001;
  gamma = 0.9;
  lambda = 0.5;

  critice = new RLLib::ATrace<double>(projector->dimension());
  critic = new RLLib::TDLambda<double>(alpha_v, gamma, lambda, critice);

  policyDistribution = new RLLib::NormalDistributionScaled<double>(random,
      problem->getContinuousActions(), 0, 1.0, projector->dimension());
  policyRange = new RLLib::Range<double>(-2.0, 2.0);
  problemRange = new RLLib::Range<double>(-2.0, 2.0);
  acting = new RLLib::ScaledPolicyDistribution<double>(problem->getContinuousActions(),
      policyDistribution, policyRange, problemRange);

  actore1 = new RLLib::ATrace<double>(projector->dimension());
  actore2 = new RLLib::ATrace<double>(projector->dimension());
  actoreTraces = new RLLib::Traces<double>();
  actoreTraces->push_back(actore1);
  actoreTraces->push_back(actore2);
  actor = new RLLib::ActorLambda<double>(alpha_u, gamma, lambda, acting, actoreTraces);

  control = new RLLib::AverageRewardActorCritic<double>(critic, actor, projector, toStateAction,
      alpha_r);
  agent = new RLLib::LearnerAgent<double>(control);
  simulator = new RLLib::RLRunner<double>(agent, problem, 5000);
}

PendulumAgent_v0::~PendulumAgent_v0()
{
  delete problem;
  delete random;
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

const RLLib::Action<double>* PendulumAgent_v0::step()
{
  simulator->step();
  return simulator->getAgentAction();
}
