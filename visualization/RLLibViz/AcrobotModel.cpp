/*
 * AcrobotModel.cpp
 *
 *  Created on: Dec 6, 2014
 *      Author: sam
 */

#include "AcrobotModel.h"

using namespace RLLibViz;

AcrobotModel::AcrobotModel()
{
  random = new Random<double>;
  problem = new Acrobot(0);
  order = 5;
  projector = new FourierBasis<double>(problem->dimension(), order, problem->getDiscreteActions());
  toStateAction = new StateActionTilings<double>(projector, problem->getDiscreteActions());
  e = new ATrace<double>(projector->dimension());
  alpha = 1.0;
  gamma = 1.0;
  lambda = 0.9;
  sarsaAdaptive = new SarsaAlphaBound<double>(alpha, gamma, lambda, e);
  epsilon = 0.01;
  acting = new EpsilonGreedy<double>(random, problem->getDiscreteActions(), sarsaAdaptive, epsilon);
  control = new SarsaControl<double>(acting, toStateAction, sarsaAdaptive);

  agent = new LearnerAgent<double>(control);
  simulator = new RLRunner<double>(agent, problem, 5000);
  simulator->setVerbose(false);
}

AcrobotModel::~AcrobotModel()
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

void AcrobotModel::doLearning(Window* window)
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

  //updateValueFunction(window, control, problem->getTRStep(), problem->getObservationRanges(),
  //    simulator->isEndingOfEpisode(), 0);

}

void AcrobotModel::doEvaluation(Window* window)
{
}

