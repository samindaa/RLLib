/*
 * SwingPendulumModel4.cpp
 *
 *  Created on: Dec 5, 2014
 *      Author: sam
 */

#include "SwingPendulumModel4.h"

using namespace RLLibViz;

SwingPendulumModel4::SwingPendulumModel4()
{
  random = new Random<double>;
  problem = new SwingPendulum<double>(0, true);
  order = 11;
  projector = new FourierBasis<double>(problem->dimension(), order, problem->getDiscreteActions());
  toStateAction = new StateActionTilings<double>(projector, problem->getDiscreteActions());
  e = new ATrace<double>(projector->dimension());
  alpha = 0.001;
  gamma = 0.99;
  lambda = 0.3;
  sarsaAdaptive = new SarsaAlphaBound<double>(alpha, gamma, lambda, e);
  epsilon = 0.01;
  acting = new EpsilonGreedy<double>(random, problem->getDiscreteActions(), sarsaAdaptive, epsilon);
  control = new SarsaControl<double>(acting, toStateAction, sarsaAdaptive);

  agent = new LearnerAgent<double>(control);
  simulator = new RLRunner<double>(agent, problem, 5000);
  simulator->setVerbose(false);
}

SwingPendulumModel4::~SwingPendulumModel4()
{
  delete random;
  delete problem;
  delete projector;
  delete toStateAction;
  delete e;
  delete sarsaAdaptive;
  delete acting;
  delete control;
  delete agent;
  delete simulator;
}

void SwingPendulumModel4::doLearning(Window* window)
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

void SwingPendulumModel4::doEvaluation(Window* window)
{
}

