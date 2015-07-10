/*
 * CartPoleModel.cpp
 *
 *  Created on: Dec 8, 2014
 *      Author: sam
 */

#include "CartPoleModel.h"

using namespace RLLibViz;

CartPoleModel::CartPoleModel()
{
  random = new Random<double>;
  problem = new CartPole(0);
  order = 3;
  //projector = new FourierBasis<double>(problem->dimension(), order, problem->getDiscreteActions());
  //toStateAction = new StateActionTilings<double>(projector, problem->getDiscreteActions());
  hashing = new MurmurHashing<double>(random, 10000);
  gridResolutions = new PVector<double>(problem->dimension());
  gridResolutions->setEntry(0, 7);
  gridResolutions->setEntry(1, 15);
  gridResolutions->setEntry(2, 7);
  gridResolutions->setEntry(3, 15);

  projector = new TileCoderHashing<double>(hashing, problem->dimension(), gridResolutions, 8, 0);
  toStateAction = new TabularAction<double>(projector, problem->getDiscreteActions(), false);

  e = new ATrace<double>(toStateAction->dimension());
  alpha = 1.0;
  gamma = 0.99;
  lambda = 0.3;
  sarsaAdaptive = new SarsaAlphaBound<double>(alpha, gamma, lambda, e);
  epsilon = 0.01;
  acting = new EpsilonGreedy<double>(random, problem->getDiscreteActions(), sarsaAdaptive, epsilon);
  control = new SarsaControl<double>(acting, toStateAction, sarsaAdaptive);

  agent = new LearnerAgent<double>(control);
  simulator = new RLRunner<double>(agent, problem, 1000);
  simulator->setVerbose(false);
}

CartPoleModel::~CartPoleModel()
{
  delete random;
  delete problem;
  delete gridResolutions;
  if (hashing)
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

void CartPoleModel::doLearning(Window* window)
{
  simulator->step();
  if (simulator->isEndingOfEpisode())
  {
    emit signal_add(window->plotVector[0], Vec(simulator->timeStep, 0),
        Vec(simulator->episodeR, 0));
    emit signal_draw(window->plotVector[0]);
  }

  emit signal_add(window->problemVector[0],
      Vec(simulator->getRLProblem()->getTRStep()->observation_tp1->getEntry(0),
          simulator->getRLProblem()->getTRStep()->observation_tp1->getEntry(2)),
      Vec(0.0, 0.0, 0.0, 1.0));
  emit signal_draw(window->problemVector[0]);

  //updateValueFunction(window, control, problem->getTRStep(), problem->getObservationRanges(),
  //    simulator->isEndingOfEpisode(), 0);

}

void CartPoleModel::doEvaluation(Window* window)
{
}

