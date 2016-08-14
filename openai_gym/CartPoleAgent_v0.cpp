/*
 * CartPoleAgent.cpp
 *
 *  Created on: Jun 27, 2016
 *      Author: sabeyruw
 */

#include "CartPoleAgent_v0.h"
#include "FourierBasis.h"

OPENAI_AGENT_MAKE(CartPoleAgent_v0)
CartPoleAgent_v0::CartPoleAgent_v0()
{

  random = new RLLib::Random<double>;
  problem = new CartPole_v0;

  order = 5;
  projector = new RLLib::FourierBasis<double>(problem->dimension(), order,
      problem->getDiscreteActions());
  toStateAction = new RLLib::StateActionTilings<double>(projector, problem->getDiscreteActions());
  e = new RLLib::ATrace<double>(projector->dimension());

  alpha = 0.001;
  gamma = 1.0;
  lambda = 0.9;
  sarsa = new RLLib::SarsaAlphaBound<double>(alpha, gamma, lambda, e);

  epsilon = 0.01;
  //acting = new RLLib::EpsilonGreedy<double>(random, problem->getDiscreteActions(), sarsa, epsilon);
  acting = new RLLib::SoftMax<double>(random, problem->getDiscreteActions(), sarsa);
  control = new RLLib::SarsaControl<double>(acting, toStateAction, sarsa);

  agent = new RLLib::LearnerAgent<double>(control);
  simulator = new RLLib::RLRunner<double>(agent, problem, 250);
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

const RLLib::Action<double>* CartPoleAgent_v0::step()
{
  simulator->step();
  return simulator->getAgentAction();
}
