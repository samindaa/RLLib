/*
 * MountainCarAgent.cpp
 *
 *  Created on: Jun 25, 2016
 *      Author: sabeyruw
 */

#include "MountainCarAgent.h"

MountainCarAgent::MountainCarAgent()
{
  problem = new MountainCar();
  random = new RLLib::Random<double>();
  hashing = new RLLib::MurmurHashing<double>(random, 10000);
  projector = new RLLib::TileCoderHashing<double>(hashing, problem->dimension(), 9, 10, false);
  toStateAction = new RLLib::StateActionTilings<double>(projector, problem->getDiscreteActions());
  e = new RLLib::ATrace<double>(projector->dimension());
  gamma = 0.99;
  lambda = 0.3;
  sarsaAdaptive = new RLLib::SarsaAlphaBound<double>(1.0f, gamma, lambda, e);
  epsilon = 0.01;
  acting = new RLLib::EpsilonGreedy<double>(random, problem->getDiscreteActions(), sarsaAdaptive,
      epsilon);
  control = new RLLib::SarsaControl<double>(acting, toStateAction, sarsaAdaptive);

  agent = new RLLib::LearnerAgent<double>(control);
  simulator = new RLLib::RLRunner<double>(agent, problem, 5000);

}

MountainCarAgent::~MountainCarAgent()
{
  delete random;
  delete problem;
  delete hashing;
  delete projector;
  delete toStateAction;
  delete e;
  delete sarsaAdaptive;
  delete acting;
  delete control;
  delete agent;
  delete simulator;
}

const RLLib::Action<double>* MountainCarAgent::toRLLibStep()
{
  simulator->step();
  return simulator->getAgentAction();
}

