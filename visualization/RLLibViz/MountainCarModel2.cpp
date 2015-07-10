/*
 * MountainCarModel2.cpp
 *
 *  Created on: Dec 4, 2014
 *      Author: sam
 */

#include "MountainCarModel2.h"

using namespace RLLibViz;

MountainCarModel2::MountainCarModel2()
{
  random = new Random<double>;
  problem = new MountainCar<double>(random);
  hashing = new MurmurHashing<double>(random, 10000);
  projector = new TileCoderHashing<double>(hashing, problem->dimension(), 9, 10, false);
  toStateAction = new StateActionTilings<double>(projector, problem->getDiscreteActions());
  e = new ATrace<double>(projector->dimension());
  gamma = 0.99;
  lambda = 0.3;
  sarsaAdaptive = new SarsaAlphaBound<double>(1.0f, gamma, lambda, e);
  epsilon = 0.01;
  acting = new EpsilonGreedy<double>(random, problem->getDiscreteActions(), sarsaAdaptive, epsilon);
  control = new SarsaControl<double>(acting, toStateAction, sarsaAdaptive);

  agent = new LearnerAgent<double>(control);
  simulator = new RLRunner<double>(agent, problem, 5000);
  simulator->setVerbose(false);
}

MountainCarModel2::~MountainCarModel2()
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

void MountainCarModel2::doLearning(Window* window)
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
          simulator->getRLProblem()->getTRStep()->observation_tp1->getEntry(1)),
      Vec(0.0, 0.0, 0.0, 1.0));
  emit signal_draw(window->problemVector[0]);

  updateValueFunction(window, control, problem->getTRStep(), problem->getObservationRanges(),
      simulator->isEndingOfEpisode(), 0);

}

void MountainCarModel2::doEvaluation(Window* window)
{
}

