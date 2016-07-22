/*
 * MountainCarAgent.cpp
 *
 *  Created on: Jun 25, 2016
 *      Author: sabeyruw
 */

#include "MountainCarAgent_v0.h"

OPENAI_AGENT_MAKE(MountainCarAgent_v0)
MountainCarAgent_v0::MountainCarAgent_v0()
{
  problem = new MountainCar_v0();
  random = new RLLib::Random<double>();
  hashing = new RLLib::MurmurHashing<double>(random, 10000);
  projector = new RLLib::TileCoderHashing<double>(hashing, problem->dimension(), 10, 10, true);
  toStateAction = new RLLib::StateActionTilings<double>(projector, problem->getDiscreteActions());
  e = new RLLib::ATrace<double>(projector->dimension());
  alpha_v = 1.0f;
  gamma = 0.99;
  lambda = 0.3;
  sarsa = new RLLib::SarsaAlphaBound<double>(alpha_v, gamma, lambda, e);
  epsilon = 0.01;
  acting = new RLLib::EpsilonGreedy<double>(random, problem->getDiscreteActions(), sarsa, epsilon);
  control = new RLLib::SarsaControl<double>(acting, toStateAction, sarsa);

  agent = new RLLib::LearnerAgent<double>(control);
  simulator = new RLLib::RLRunner<double>(agent, problem, 5000);
  simulator->setVerbose(false);
}

MountainCarAgent_v0::~MountainCarAgent_v0()
{
  delete random;
  delete problem;
  delete hashing;
  delete projector;
  delete toStateAction;
  delete e;
  delete sarsa;
  delete acting;
  delete control;
  delete agent;
  delete simulator;
}

const RLLib::Action<double>* MountainCarAgent_v0::step()
{
  simulator->step();
  return simulator->getAgentAction();
}

