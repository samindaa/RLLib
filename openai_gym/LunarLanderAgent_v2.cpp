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

  order = 5;
  projector = new LunarLanderProjector_v2(order, problem->getDiscreteActions());
  toStateAction = new RLLib::StateActionTilings<double>(projector, problem->getDiscreteActions());
  e = new RLLib::ATrace<double>(projector->dimension());

  alpha = 0.01;
  gamma = 0.95;
  lambda = 0.9;
  sarsa = new RLLib::SarsaAlphaBound<double>(alpha, gamma, lambda, e);

  epsilon = 0.01;
  //acting = new RLLib::EpsilonGreedy<double>(random, problem->getDiscreteActions(), sarsa, epsilon);
  acting = new RLLib::SoftMax<double>(random, problem->getDiscreteActions(), sarsa);
  control = new RLLib::SarsaControl<double>(acting, toStateAction, sarsa);

  agent = new RLLib::LearnerAgent<double>(control);
  simulator = new RLLib::RLRunner<double>(agent, problem, 2000);
  simulator->setVerbose(false);
}

LunarLanderAgent_v2::~LunarLanderAgent_v2()
{
  delete random;
  delete problem;
  delete projector;
  delete toStateAction;
  delete e;
  delete sarsa;
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

