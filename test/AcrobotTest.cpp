/*
 * AcrobotTest.cpp
 *
 *  Created on: Dec 10, 2013
 *      Author: sam
 */

#include "AcrobotTest.h"

void AcrobotTest::testAcrobotOnPolicy()
{
  Probabilistic<double>::srand(time(0));
  RLProblem<double>* problem = new Acrobot<double>(false);
  Projector<double>* projector = new AcrobotProjector<double>(problem->dimension(),
      problem->getDiscreteActions()->dimension(), 8);
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());
  Trace<double>* e = new ATrace<double>(projector->dimension());
  double alpha = 0.1 / projector->vectorNorm();
  double gamma = 1.0;
  double lambda = 0.9;
  Sarsa<double>* sarsa = new SarsaTrue<double>(alpha, gamma, lambda, e);
  double epsilon = 0.0;
  Policy<double>* acting = new EpsilonGreedy<double>(sarsa, problem->getDiscreteActions(), epsilon);
  OnPolicyControlLearner<double>* control = new SarsaControl<double>(acting, toStateAction, sarsa);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  Simulator<double>* sim = new Simulator<double>(agent, problem, 1000, 300, 10);
  sim->setTestEpisodesAfterEachRun(true);
  sim->run();
  control->persist("visualization/Acrobot.dat");
  control->reset();
  control->resurrect("visualization/Acrobot.dat");
  sim->runEvaluate(10);

  delete problem;
  delete projector;
  delete toStateAction;
  delete e;
  delete sarsa;
  delete acting;
  delete control;
  delete agent;
  delete sim;
}

void AcrobotTest::run()
{
  testAcrobotOnPolicy();
}

RLLIB_TEST_MAKE(AcrobotTest)
