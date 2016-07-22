/*
 * LunarLanderAgent_v2.cpp
 *
 *  Created on: Jul 18, 2016
 *      Author: sabeyruw
 */

#include "LunarLanderAgent_v2.h"

OPENAI_AGENT_MAKE(LunarLanderAgent_v2)
LunarLanderAgent_v2::LunarLanderAgent_v2()
{

  random = new RLLib::Random<double>;
  problem = new LunarLander_v2;
  projector = new LunarLanderProjector_v2(random);
  toStateAction = new RLLib::StateActionTilings<double>(projector, problem->getDiscreteActions());

  alpha_v = 0.1 / projector->vectorNorm();
  alpha_u = 0.001 / projector->vectorNorm();
  alpha_r = .0001;
  gamma = 1.0;
  lambda = 0.5;

  critice = new RLLib::ATrace<double>(projector->dimension());
  critic = new RLLib::TDLambda<double>(alpha_v, gamma, lambda, critice);

  acting = new RLLib::BoltzmannDistribution<double>(random, problem->getDiscreteActions(),
      projector->dimension());

  actore1 = new RLLib::ATrace<double>(projector->dimension());
  actoreTraces = new RLLib::Traces<double>();
  actoreTraces->push_back(actore1);
  actor = new RLLib::ActorLambda<double>(alpha_u, gamma, lambda, acting, actoreTraces);

  control = new RLLib::AverageRewardActorCritic<double>(critic, actor, projector, toStateAction,
      alpha_r);

  agent = new RLLib::LearnerAgent<double>(control);
  simulator = new RLLib::RLRunner<double>(agent, problem, 1600);
  simulator->setVerbose(true);

}

LunarLanderAgent_v2::~LunarLanderAgent_v2()
{
  delete random;
  delete problem;
  delete projector;
  delete toStateAction;
  delete critice;
  delete critic;
  delete actore1;
  delete actoreTraces;
  delete actor;
  delete acting;
  delete control;
  delete agent;
  delete simulator;
}

const RLLib::Action<double>* LunarLanderAgent_v2::step()
{
  simulator->step();
  return simulator->getAgentAction();
}

