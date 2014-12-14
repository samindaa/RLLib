/*
 * AcrobotTest.cpp
 *
 *  Created on: Dec 10, 2013
 *      Author: sam
 */

#include "AcrobotTest.h"

void AcrobotTest::testAcrobotOnPolicy()
{
  Random<double>* random = new Random<double>;
  RLProblem<double>* problem = new Acrobot(0);
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
  Policy<double>* acting = new EpsilonGreedy<double>(random, problem->getDiscreteActions(), sarsa,
      epsilon);
  OnPolicyControlLearner<double>* control = new SarsaControl<double>(acting, toStateAction, sarsa);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  RLRunner<double>* sim = new RLRunner<double>(agent, problem, 1000, 300, 10);
  sim->setTestEpisodesAfterEachRun(true);
  sim->run();
  control->persist("visualization/Acrobot.dat");
  control->reset();
  control->resurrect("visualization/Acrobot.dat");
  sim->runEvaluate(10);

  delete random;
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
