/*
 * CartPoleAgent.cpp
 *
 *  Created on: Jun 27, 2016
 *      Author: sabeyruw
 */

#include "CartPoleAgent_v0.h"

CartPoleAgent_v0::CartPoleAgent_v0()
{

  random = new RLLib::Random<double>;
  problem = new CartPole_v0;
  projector = new CartPoleProjector_v0(random);
  toStateAction = new RLLib::StateActionTilings<double>(projector, problem->getDiscreteActions());

  e = new RLLib::ATrace<double>(toStateAction->dimension());
  alpha = 0.1 / projector->vectorNorm();
  gamma = 0.99;
  lambda = 0.4;
  sarsa = new RLLib::SarsaTrue<double>(alpha, gamma, lambda, e);
  epsilon = 0.01;
  //acting = new RLLib::EpsilonGreedy<double>(random, problem->getDiscreteActions(), sarsa, epsilon);
  acting = new RLLib::SoftMax<double>(random, problem->getDiscreteActions(), sarsa);
  control = new RLLib::SarsaControl<double>(acting, toStateAction, sarsa);

  agent = new RLLib::LearnerAgent<double>(control);
  simulator = new RLLib::RLRunner<double>(agent, problem, 1000);
  simulator->setVerbose(false);

}

CartPoleAgent_v0::~CartPoleAgent_v0()
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

const RLLib::Action<double>* CartPoleAgent_v0::toRLLibStep()
{
  simulator->step();
  return simulator->getAgentAction();
}
