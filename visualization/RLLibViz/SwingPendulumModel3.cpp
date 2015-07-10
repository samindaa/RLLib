/*
 * SwingPendulumModel3.cpp
 *
 *  Created on: Dec 4, 2014
 *      Author: sam
 */

#include "SwingPendulumModel3.h"

using namespace RLLibViz;

SwingPendulumModel3::SwingPendulumModel3()
{
  random = new Random<double>;
  problem = new SwingPendulum<double>(0, true);
  hashing = new MurmurHashing<double>(random, 100000);
  projector = new TileCoderHashing<double>(hashing, problem->dimension(), 16, 16, true);
  toStateAction = new StateActionTilings<double>(projector, problem->getDiscreteActions());
  e = new ATrace<double>(projector->dimension());
  alpha = 0.2 / projector->vectorNorm();
  gamma = 0.99;
  lambda = 0.4;
  sarsaTrue = new SarsaTrue<double>(alpha, gamma, lambda, e);
  epsilon = 0.01;
  acting = new EpsilonGreedy<double>(random, problem->getDiscreteActions(), sarsaTrue, epsilon);
  control = new SarsaControl<double>(acting, toStateAction, sarsaTrue);

  agent = new LearnerAgent<double>(control);
  simulator = new RLRunner<double>(agent, problem, 5000);
  simulator->setVerbose(false);
}

SwingPendulumModel3::~SwingPendulumModel3()
{
  delete random;
  delete problem;
  delete hashing;
  delete projector;
  delete toStateAction;
  delete e;
  delete sarsaTrue;
  delete acting;
  delete control;
  delete agent;
  delete simulator;
}

void SwingPendulumModel3::doLearning(Window* window)
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

void SwingPendulumModel3::doEvaluation(Window* window)
{
}

