/*
 * AcrobotAgent.cpp
 *
 *  Created on: Jun 27, 2016
 *      Author: sabeyruw
 */

#include "AcrobotAgent.h"

AcrobotAgent::AcrobotAgent()
{
  random = new RLLib::Random<double>;
  problem = new Acrobot();
  order = 5;
  projector = new RLLib::FourierBasis<double>(problem->dimension(), order,
      problem->getDiscreteActions());
  toStateAction = new RLLib::StateActionTilings<double>(projector, problem->getDiscreteActions());
  e = new RLLib::ATrace<double>(projector->dimension());
  alpha = 1.0;
  gamma = 1.0;
  lambda = 0.9;
  sarsa = new RLLib::SarsaAlphaBound<double>(alpha, gamma, lambda, e);
  epsilon = 0.01;
  acting = new RLLib::EpsilonGreedy<double>(random, problem->getDiscreteActions(), sarsa, epsilon);
  control = new RLLib::SarsaControl<double>(acting, toStateAction, sarsa);

  agent = new RLLib::LearnerAgent<double>(control);
  simulator = new RLLib::RLRunner<double>(agent, problem, 5000);
  simulator->setVerbose(false);
}

AcrobotAgent::~AcrobotAgent()
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

const RLLib::Action<double>* AcrobotAgent::toRLLibStep()
{
  simulator->step();
  return simulator->getAgentAction();
}

