/*
 * LunarLanderAgent_v2.cpp
 *
 *  Created on: Jul 18, 2016
 *      Author: sabeyruw
 */

#include "LunarLanderAgent_v2.h"

LunarLanderAgent_v2::LunarLanderAgent_v2()
{

  random = new RLLib::Random<double>;
  problem = new LunarLander_v2;
  projector = new LunarLanderProjector_v2(random);
  toStateAction = new RLLib::StateActionTilings<double>(projector, problem->getDiscreteActions());

  e = new RLLib::ATrace<double>(toStateAction->dimension());
  alpha = 0.2f / projector->vectorNorm();
  gamma = 0.9f;
  lambda = 0.4f;
  sarsa = new RLLib::SarsaTrue<double>(alpha, gamma, lambda, e);
  epsilon = 0.01;
  //acting = new RLLib::EpsilonGreedy<double>(random, problem->getDiscreteActions(), sarsa, epsilon);
  acting = new RLLib::SoftMax<double>(random, problem->getDiscreteActions(), sarsa);
  control = new RLLib::SarsaControl<double>(acting, toStateAction, sarsa);

  agent = new RLLib::LearnerAgent<double>(control);
  simulator = new RLLib::RLRunner<double>(agent, problem, 300);
  simulator->setVerbose(true);

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

