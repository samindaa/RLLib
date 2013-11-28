/*
 * Copyright 2013 Saminda Abeyruwan (saminda@cs.miami.edu)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * MountainCarTest.cpp
 *
 *  Created on: May 15, 2013
 *      Author: sam
 */

#include "MountainCarTest.h"

RLLIB_TEST_MAKE(MountainCarTest)

void MountainCarTest::testSarsaTabularActionMountainCar()
{
  Probabilistic::srand(0);
  cout << "time=" << time(0) << endl;
  RLProblem<double>* problem = new MountainCar<double>;
  //Projector<double>* projector = new TileCoderNoHashing<double>(1000, 10, true); << FixMe
  Projector<double>* projector = new TileCoderHashing<double>(1000, 10, true);
  StateToStateAction<double>* toStateAction = new TabularAction<double>(projector,
      problem->getDiscreteActions(), true);
  Trace<double>* e = new RTrace<double>(toStateAction->dimension());

  cout << "|phi_sa|=" << toStateAction->dimension() << endl;
  cout << "||.||=" << toStateAction->vectorNorm() << endl;

  double alpha = 0.15 / toStateAction->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.3;
  Sarsa<double>* sarsa = new Sarsa<double>(alpha, gamma, lambda, e);
  double epsilon = 0.01;
  Policy<double>* acting = new EpsilonGreedy<double>(sarsa, problem->getDiscreteActions(), epsilon);
  OnPolicyControlLearner<double>* control = new SarsaControl<double>(acting, toStateAction, sarsa);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  Simulator<double>* sim = new Simulator<double>(agent, problem, 5000, 300, 1);
  sim->run();
  sim->computeValueFunction();

  delete problem;
  delete projector;
  delete toStateAction;
  delete e;
  delete sarsa;
  delete acting;
  delete control;
  delete agent;
  delete agent;
  delete sim;
}

void MountainCarTest::testOnPolicyBoltzmannRTraceTabularActionCar()
{
  Probabilistic::srand(0);
  RLProblem<double>* problem = new MountainCar<double>;

  Projector<double>* projector = new TileCoderHashing<double>(1000, 10, false);
  StateToStateAction<double>* toStateAction = new TabularAction<double>(projector,
      problem->getDiscreteActions(), false);

  cout << "|x_t|=" << projector->dimension() << endl;
  cout << "||.||=" << projector->vectorNorm() << endl;
  cout << "|phi_sa|=" << toStateAction->dimension() << endl;
  cout << "||.||=" << toStateAction->vectorNorm() << endl;

  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_u = 0.01 / projector->vectorNorm();
  double lambda = 0.3;
  double gamma = 0.99;

  Trace<double>* critice = new RTrace<double>(projector->dimension());
  TDLambda<double>* critic = new TDLambda<double>(alpha_v, gamma, lambda, critice);

  PolicyDistribution<double>* acting = new BoltzmannDistribution<double>(toStateAction->dimension(),
      problem->getDiscreteActions());

  Trace<double>* actore = new RTrace<double>(toStateAction->dimension());
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore);
  ActorOnPolicy<double>* actor = new ActorLambda<double>(alpha_u, gamma, lambda, acting,
      actoreTraces);

  OnPolicyControlLearner<double>* control = new ActorCritic<double>(critic, actor, projector,
      toStateAction);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  Simulator<double>* sim = new Simulator<double>(agent, problem, 5000, 300, 1);
  sim->run();
  sim->computeValueFunction();

  delete problem;
  delete projector;
  delete toStateAction;
  delete critice;
  delete critic;
  delete actore;
  delete actoreTraces;
  delete actor;
  delete acting;
  delete control;
  delete agent;
  delete sim;
}

void MountainCarTest::testSarsaMountainCar()
{
  Probabilistic::srand(0);
  RLProblem<float>* problem = new MountainCar<float>;
  Hashing* hashing = new UNH;
  Projector<float>* projector = new TileCoderHashing<float>(10000, 10, true, hashing);
  StateToStateAction<float>* toStateAction = new StateActionTilings<float>(projector,
      problem->getDiscreteActions());
  Trace<float>* e = new RTrace<float>(projector->dimension());
  double alpha = 0.15 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.3;
  Sarsa<float>* sarsa = new Sarsa<float>(alpha, gamma, lambda, e);
  double epsilon = 0.01;
  Policy<float>* acting = new EpsilonGreedy<float>(sarsa, problem->getDiscreteActions(), epsilon);
  OnPolicyControlLearner<float>* control = new SarsaControl<float>(acting, toStateAction, sarsa);

  RLAgent<float>* agent = new LearnerAgent<float>(control);
  Simulator<float>* sim = new Simulator<float>(agent, problem, 5000, 300, 1);
  sim->run();
  sim->computeValueFunction();

  delete problem;
  delete hashing;
  delete projector;
  delete toStateAction;
  delete e;
  delete sarsa;
  delete acting;
  delete control;
  delete agent;
  delete sim;
}

void MountainCarTest::testSarsaAdaptiveMountainCar()
{
  Probabilistic::srand(0);
  RLProblem<double>* problem = new MountainCar<double>;
  problem->setResolution(9.0);
  Projector<double>* projector = new TileCoderHashing<double>(10000, 10, false);
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());
  Trace<double>* e = new ATrace<double>(projector->dimension());
  double gamma = 0.99;
  double lambda = 0.3;
  Sarsa<double>* sarsaAdaptive = new SarsaAlphaBound<double>(gamma, lambda, e);
  double epsilon = 0.01;
  Policy<double>* acting = new EpsilonGreedy<double>(sarsaAdaptive, problem->getDiscreteActions(),
      epsilon);
  OnPolicyControlLearner<double>* control = new SarsaControl<double>(acting, toStateAction,
      sarsaAdaptive);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  Simulator<double>* sim = new Simulator<double>(agent, problem, 5000, 300, 1);
  sim->setTestEpisodesAfterEachRun(true);
  sim->run();
  sim->computeValueFunction();

  delete problem;
  delete projector;
  delete toStateAction;
  delete e;
  delete sarsaAdaptive;
  delete acting;
  delete control;
  delete agent;
  delete sim;
}

void MountainCarTest::testExpectedSarsaMountainCar()
{
  RLProblem<double>* problem = new MountainCar<double>;
  Projector<double>* projector = new TileCoderHashing<double>(10000, 10, true);
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());
  Trace<double>* e = new RTrace<double>(projector->dimension());
  double alpha = 0.2 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.1;
  Sarsa<double>* sarsa = new Sarsa<double>(alpha, gamma, lambda, e);
  double epsilon = 0.01;
  Policy<double>* acting = new EpsilonGreedy<double>(sarsa, problem->getDiscreteActions(), epsilon);
  OnPolicyControlLearner<double>* control = new ExpectedSarsaControl<double>(acting, toStateAction,
      sarsa, problem->getDiscreteActions());

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  Simulator<double>* sim = new Simulator<double>(agent, problem, 5000, 300, 5);
  sim->run();
  sim->computeValueFunction();

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

void MountainCarTest::testGreedyGQOnPolicyMountainCar()
{
  Probabilistic::srand(0);
  RLProblem<double>* problem = new MountainCar<double>;
  Projector<double>* projector = new TileCoderHashing<double>(10000, 10, true);
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());
  Trace<double>* e = new ATrace<double>(projector->dimension());
  double alpha_v = 0.05 / projector->vectorNorm();
  double alpha_w = 0.0 / projector->vectorNorm();
  double gamma_tp1 = 0.9;
  double beta_tp1 = 1.0 - gamma_tp1;
  double lambda_t = 0.1;
  GQ<double>* gq = new GQ<double>(alpha_v, alpha_w, beta_tp1, lambda_t, e);
  //double epsilon = 0.01;
  Policy<double>* acting = new EpsilonGreedy<double>(gq, problem->getDiscreteActions(), 0.01);

  OffPolicyControlLearner<double>* control = new GQOnPolicyControl<double>(acting,
      problem->getDiscreteActions(), toStateAction, gq);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  Simulator<double>* sim = new Simulator<double>(agent, problem, 5000, 300, 1);
  sim->run();
  sim->computeValueFunction();

  delete problem;
  delete projector;
  delete toStateAction;
  delete e;
  delete gq;
  delete acting;
  delete control;
  delete agent;
  delete sim;
}

void MountainCarTest::testGreedyGQMountainCar()
{
  Probabilistic::srand(0);
  RLProblem<double>* problem = new MountainCar<double>;
  Hashing* hashing = new MurmurHashing;
  Projector<double>* projector = new TileCoderHashing<double>(1000000, 10, true, hashing);
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());
  Trace<double>* e = new ATrace<double>(projector->dimension());
  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_w = 0.0001 / projector->vectorNorm();
  double gamma_tp1 = 0.99;
  double beta_tp1 = 1.0 - gamma_tp1;
  double lambda_t = 0.4;
  GQ<double>* gq = new GQ<double>(alpha_v, alpha_w, beta_tp1, lambda_t, e);
  //double epsilon = 0.01;
  //Policy<double>* behavior = new EpsilonGreedy<double>(gq,
  //    problem->getActions(), epsilon);
  Policy<double>* behavior = new RandomPolicy<double>(problem->getDiscreteActions());
  Policy<double>* target = new Greedy<double>(gq, problem->getDiscreteActions());
  OffPolicyControlLearner<double>* control = new GreedyGQ<double>(target, behavior,
      problem->getDiscreteActions(), toStateAction, gq);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  Simulator<double>* sim = new Simulator<double>(agent, problem, 5000, 100, 10);
  sim->setTestEpisodesAfterEachRun(true);
  sim->run();
  sim->computeValueFunction();

  delete problem;
  delete hashing;
  delete projector;
  delete toStateAction;
  delete e;
  delete gq;
  delete behavior;
  delete target;
  delete control;
  delete agent;
  delete sim;
}

void MountainCarTest::testSoftmaxGQOnMountainCar()
{
  Probabilistic::srand(0);
  RLProblem<double>* problem = new MountainCar<double>;
  Projector<double>* projector = new TileCoderHashing<double>(1000000, 10, true);
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());
  Trace<double>* e = new ATrace<double>(projector->dimension());
  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_w = .0005 / projector->vectorNorm();
  double gamma_tp1 = 0.99;
  double beta_tp1 = 1.0 - gamma_tp1;
  double lambda_t = 0.4;
  GQ<double>* gq = new GQ<double>(alpha_v, alpha_w, beta_tp1, lambda_t, e);
  Policy<double>* behavior = new RandomPolicy<double>(problem->getDiscreteActions());
  Policy<double>* target = new SoftMax<double>(gq, problem->getDiscreteActions(), 0.1);
  OffPolicyControlLearner<double>* control = new GreedyGQ<double>(target, behavior,
      problem->getDiscreteActions(), toStateAction, gq);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  Simulator<double>* sim = new Simulator<double>(agent, problem, 5000, 100, 10);
  sim->setTestEpisodesAfterEachRun(true);
  sim->run();
  sim->computeValueFunction();

  delete problem;
  delete projector;
  delete toStateAction;
  delete e;
  delete gq;
  delete behavior;
  delete target;
  delete control;
  delete agent;
  delete sim;
}

void MountainCarTest::testOffPACMountainCar()
{
  Probabilistic::srand(0);
  RLProblem<float>* problem = new MountainCar<float>;
  Hashing* hashing = new MurmurHashing;
  Projector<float>* projector = new TileCoderHashing<float>(1000000, 10, true, hashing);
  StateToStateAction<float>* toStateAction = new StateActionTilings<float>(projector,
      problem->getDiscreteActions());

  double alpha_v = 0.05 / projector->vectorNorm();
  double alpha_w = 0.0001 / projector->vectorNorm();
  double lambda = 0.0;  //0.4;
  double gamma = 0.99;
  Trace<float>* critice = new ATrace<float>(projector->dimension());
  OffPolicyTD<float>* critic = new GTDLambda<float>(alpha_v, alpha_w, gamma, lambda, critice);
  double alpha_u = 1.0 / projector->vectorNorm();
  PolicyDistribution<float>* target = new BoltzmannDistribution<float>(projector->dimension(),
      problem->getDiscreteActions());

  Trace<float>* actore = new ATrace<float>(projector->dimension());
  Traces<float>* actoreTraces = new Traces<float>();
  actoreTraces->push_back(actore);
  ActorOffPolicy<float>* actor = new ActorLambdaOffPolicy<float>(alpha_u, gamma, lambda, target,
      actoreTraces);

  Policy<float>* behavior = new RandomPolicy<float>(problem->getDiscreteActions());

  OffPolicyControlLearner<float>* control = new OffPAC<float>(behavior, critic, actor,
      toStateAction, projector);

  RLAgent<float>* agent = new LearnerAgent<float>(control);
  Simulator<float>* sim = new Simulator<float>(agent, problem, 5000, 100, 10);
  sim->setTestEpisodesAfterEachRun(true);
  //sim->setVerbose(false);
  sim->run();
  sim->computeValueFunction();
  control->persist("visualization/mcar_offpac.data");

  control->reset();
  control->resurrect("visualization/mcar_offpac.data");
  sim->runEvaluate(10, 10);

  delete problem;
  delete hashing;
  delete projector;
  delete toStateAction;
  delete critice;
  delete critic;
  delete actore;
  delete actoreTraces;
  delete actor;
  delete behavior;
  delete target;
  delete control;
  delete agent;
  delete sim;
}

void MountainCarTest::testOffPACOnPolicyMountainCar()
{
  Probabilistic::srand(0);
  RLProblem<double>* problem = new MountainCar<double>;
  Projector<double>* projector = new TileCoderHashing<double>(1000, 10, true);
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());

  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_w = .0001 / projector->vectorNorm();
  double lambda = 0.4;
  double gamma = 0.99;
  Trace<double>* critice = new ATrace<double>(projector->dimension());
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma, lambda, critice);
  double alpha_u = 1.0 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(projector->dimension(),
      problem->getDiscreteActions());
  Policy<double>* behavior = (Policy<double>*) target;

  Trace<double>* actore = new ATrace<double>(projector->dimension());
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore);
  ActorOffPolicy<double>* actor = new ActorLambdaOffPolicy<double>(alpha_u, gamma, lambda, target,
      actoreTraces);

  OffPolicyControlLearner<double>* control = new OffPAC<double>(behavior, critic, actor,
      toStateAction, projector);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  Simulator<double>* sim = new Simulator<double>(agent, problem, 5000, 100, 1);
  sim->run();
  sim->computeValueFunction();
  control->persist("visualization/mcar_offpac.data");

  control->reset();
  control->resurrect("visualization/mcar_offpac.data");
  sim->runEvaluate();

  delete problem;
  delete projector;
  delete toStateAction;
  delete critice;
  delete critic;
  delete actore;
  delete actoreTraces;
  delete actor;
  delete target;
  delete control;
  delete agent;
  delete sim;
}

void MountainCarTest::testOnPolicyContinousActionCar(const int& nbMemory, const double& lambda,
    const double& gamma, double alpha_v, double alpha_u)
{
  Probabilistic::srand(0);
  RLProblem<double>* problem = new MountainCar<double>;
  Projector<double>* projector = new TileCoderHashing<double>(nbMemory, 10, false);
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getContinuousActions());

  alpha_v /= projector->vectorNorm();
  alpha_u /= projector->vectorNorm();

  Trace<double>* critice = new RTrace<double>(projector->dimension());
  TDLambda<double>* critic = new TDLambda<double>(alpha_v, gamma, lambda, critice);

  PolicyDistribution<double>* acting = new NormalDistributionScaled<double>(0, 1.0,
      projector->dimension(), problem->getContinuousActions());

  Trace<double>* actore1 = new RTrace<double>(projector->dimension());
  Trace<double>* actore2 = new RTrace<double>(projector->dimension());
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore1);
  actoreTraces->push_back(actore2);
  ActorOnPolicy<double>* actor = new ActorLambda<double>(alpha_u, gamma, lambda, acting,
      actoreTraces);

  OnPolicyControlLearner<double>* control = new AverageRewardActorCritic<double>(critic, actor,
      projector, toStateAction, 0);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  Simulator<double>* sim = new Simulator<double>(agent, problem, 5000, 200, 1);
  sim->setTestEpisodesAfterEachRun(true);
  sim->run();
  sim->computeValueFunction();

  delete problem;
  delete projector;
  delete toStateAction;
  delete critice;
  delete critic;
  delete actore1;
  delete actore2;
  delete actoreTraces;
  delete actor;
  delete acting;
  delete control;
  delete agent;
  delete sim;
}

void MountainCarTest::testOnPolicyBoltzmannATraceCar()
{
  Probabilistic::srand(0);
  RLProblem<double>* problem = new MountainCar<double>;

  Projector<double>* projector = new TileCoderHashing<double>(10000, 10, false);
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());

  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_u = 0.01 / projector->vectorNorm();
  double lambda = 0.3;
  double gamma = 0.99;

  Trace<double>* critice = new ATrace<double>(projector->dimension());
  TDLambda<double>* critic = new TDLambda<double>(alpha_v, gamma, lambda, critice);

  PolicyDistribution<double>* acting = new BoltzmannDistribution<double>(projector->dimension(),
      problem->getDiscreteActions());

  Trace<double>* actore = new ATrace<double>(projector->dimension());
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore);
  ActorOnPolicy<double>* actor = new ActorLambda<double>(alpha_u, gamma, lambda, acting,
      actoreTraces);

  OnPolicyControlLearner<double>* control = new ActorCritic<double>(critic, actor, projector,
      toStateAction);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  Simulator<double>* sim = new Simulator<double>(agent, problem, 5000, 300, 1);
  sim->setTestEpisodesAfterEachRun(true);
  sim->run();
  sim->computeValueFunction();

  delete problem;
  delete projector;
  delete toStateAction;
  delete critice;
  delete critic;
  delete actore;
  delete actoreTraces;
  delete actor;
  delete acting;
  delete control;
  delete agent;
  delete sim;
}

void MountainCarTest::testOnPolicyBoltzmannRTraceCar()
{
  Probabilistic::srand(0);
  RLProblem<double>* problem = new MountainCar<double>;

  Projector<double>* projector = new TileCoderHashing<double>(10000, 10, false);
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());

  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_u = 0.01 / projector->vectorNorm();
  double lambda = 0.3;
  double gamma = 0.99;

  Trace<double>* critice = new RTrace<double>(projector->dimension());
  TDLambda<double>* critic = new TDLambda<double>(alpha_v, gamma, lambda, critice);

  PolicyDistribution<double>* acting = new BoltzmannDistribution<double>(projector->dimension(),
      problem->getDiscreteActions());

  Trace<double>* actore = new RTrace<double>(projector->dimension());
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore);
  ActorOnPolicy<double>* actor = new ActorLambda<double>(alpha_u, gamma, lambda, acting,
      actoreTraces);

  OnPolicyControlLearner<double>* control = new ActorCritic<double>(critic, actor, projector,
      toStateAction);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  Simulator<double>* sim = new Simulator<double>(agent, problem, 5000, 300, 1);
  sim->setTestEpisodesAfterEachRun(true);
  sim->run();
  sim->computeValueFunction();

  delete problem;
  delete projector;
  delete toStateAction;
  delete critice;
  delete critic;
  delete actore;
  delete actoreTraces;
  delete actor;
  delete acting;
  delete control;
  delete agent;
  delete sim;
}

void MountainCarTest::testOnPolicyContinousActionCar()
{
  testOnPolicyContinousActionCar(10000, 0.4, 0.99, 0.1, 0.001);
}

void MountainCarTest::testOnPolicyBoltzmannATraceNaturalActorCriticCar()
{
  Probabilistic::srand(0);
  RLProblem<double>* problem = new MountainCar<double>;

  Projector<double>* projector = new TileCoderHashing<double>(10000, 10, false);
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());

  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_u = 0.001 / projector->vectorNorm();
  double lambda = 0.4;
  double gamma = 0.99;

  Trace<double>* critice = new ATrace<double>(projector->dimension());
  TDLambda<double>* critic = new TDLambda<double>(alpha_v, gamma, lambda, critice);

  PolicyDistribution<double>* acting = new BoltzmannDistribution<double>(projector->dimension(),
      problem->getDiscreteActions());

  ActorOnPolicy<double>* actor = new ActorNatural<double>(alpha_u, alpha_v, acting);
  OnPolicyControlLearner<double>* control = new ActorCritic<double>(critic, actor, projector,
      toStateAction);
  RLAgent<double>* agent = new LearnerAgent<double>(control);
  Simulator<double>* sim = new Simulator<double>(agent, problem, 5000, 100, 10);
  sim->setTestEpisodesAfterEachRun(true);
  sim->run();
  sim->computeValueFunction();

  delete problem;
  delete projector;
  delete toStateAction;
  delete critice;
  delete critic;
  delete actor;
  delete acting;
  delete control;
  delete agent;
  delete sim;
}

void MountainCarTest::run()
{
  testSarsaMountainCar();
  testSarsaAdaptiveMountainCar();
  testExpectedSarsaMountainCar();

  testGreedyGQMountainCar();
  testSoftmaxGQOnMountainCar();
  testOffPACMountainCar();
  testGreedyGQOnPolicyMountainCar();
  testOffPACOnPolicyMountainCar();

  testOnPolicyBoltzmannATraceCar();
  testOnPolicyBoltzmannRTraceCar();
  testOnPolicyContinousActionCar();
  testOnPolicyBoltzmannATraceNaturalActorCriticCar();

  testSarsaTabularActionMountainCar();
  testOnPolicyBoltzmannRTraceTabularActionCar();

}

