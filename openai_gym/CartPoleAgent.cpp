/*
 * CartPoleAgent.cpp
 *
 *  Created on: Jun 27, 2016
 *      Author: sabeyruw
 */

#include "CartPoleAgent.h"

CartPoleAgent::CartPoleAgent()
{

  random = new RLLib::Random<double>;
  problem = new CartPole;

  order = 5;
  //projector = new RLLib::FourierBasis<double>(problem->dimension(), order,
  //    problem->getDiscreteActions());
  //toStateAction = new RLLib::StateActionTilings<double>(projector, problem->getDiscreteActions());
  //hashing = NULL;
  //gridResolutions = NULL;

  hashing = new RLLib::MurmurHashing<double>(random, 1000);
  gridResolutions = new RLLib::PVector<double>(problem->dimension());
  gridResolutions->setEntry(0, 8);
  gridResolutions->setEntry(1, 16);
  gridResolutions->setEntry(2, 8);
  gridResolutions->setEntry(3, 16);
  projector = new RLLib::TileCoderHashing<double>(hashing, problem->dimension(), gridResolutions,
      12);

  toStateAction = new RLLib::TabularAction<double>(projector, problem->getDiscreteActions(), true);

  e = new RLLib::ATrace<double>(toStateAction->dimension());
  alpha = 1.0;
  gamma = 0.99;
  lambda = 0.3;
  sarsa = new RLLib::SarsaAlphaBound<double>(alpha, gamma, lambda, e);
  epsilon = 0.01;
  acting = new RLLib::EpsilonGreedy<double>(random, problem->getDiscreteActions(), sarsa, epsilon);
  control = new RLLib::SarsaControl<double>(acting, toStateAction, sarsa);

  agent = new RLLib::LearnerAgent<double>(control);
  simulator = new RLLib::RLRunner<double>(agent, problem, 1000);
  simulator->setVerbose(false);

}

CartPoleAgent::~CartPoleAgent()
{

  delete random;
  delete problem;
  if (gridResolutions)
  {
    delete gridResolutions;
  }
  if (hashing)
  {
    delete hashing;
  }
  delete projector;
  delete toStateAction;
  delete e;
  delete sarsa;
  delete acting;
  delete control;
  delete agent;
  delete simulator;
}

const RLLib::Action<double>* CartPoleAgent::toRLLibStep()
{
  simulator->step();
  return simulator->getAgentAction();
}
