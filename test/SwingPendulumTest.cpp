/*
 * Copyright 2015 Saminda Abeyruwan (saminda@cs.miami.edu)
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
 * SwingPendulumTest.cpp
 *
 *  Created on: May 15, 2013
 *      Author: sam
 */

#include "SwingPendulumTest.h"

RLLIB_TEST_MAKE(SwingPendulumTest)

void SwingPendulumTest::testOffPACSwingPendulum()
{
  Random<double>* random = new Random<double>;
  RLProblem<double>* problem = new SwingPendulum<double>;
  Hashing<double>* hashing = new MurmurHashing<double>(random, 1000000);
  Projector<double>* projector = new TileCoderHashing<double>(hashing, problem->dimension(), 10, 10,
      true);
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());

  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_w = .0001 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.4;

  Trace<double>* critice = new ATrace<double>(projector->dimension());
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma, lambda, critice);
  double alpha_u = 0.5 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(random,
      problem->getDiscreteActions(), projector->dimension());

  Trace<double>* actore = new ATrace<double>(projector->dimension());
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore);
  ActorOffPolicy<double>* actor = new ActorLambdaOffPolicy<double>(alpha_u, gamma, lambda, target,
      actoreTraces);

  Policy<double>* behavior = new RandomPolicy<double>(random, problem->getDiscreteActions());
  /*Policy<double>* behavior = new RandomBiasPolicy<double>(
   &problem->getDiscreteActions());*/
  OffPolicyControlLearner<double>* control = new OffPAC<double>(behavior, critic, actor,
      toStateAction, projector);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  RLRunner<double>* sim = new RLRunner<double>(agent, problem, 5000, 200, 1);
  sim->setTestEpisodesAfterEachRun(true);
  sim->run();
  sim->computeValueFunction();

  delete random;
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

void SwingPendulumTest::testOnPolicySwingPendulum()
{
  Random<double>* random = new Random<double>;
  RLProblem<double>* problem = new SwingPendulum<double>;
  Hashing<double>* hashing = new MurmurHashing<double>(random, 1000);
  Projector<double>* projector = new TileCoderHashing<double>(hashing, problem->dimension(), 10, 10,
      false);
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getContinuousActions());

  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_u = 0.001 / projector->vectorNorm();
  double alpha_r = .0001;
  double gamma = 1.0;
  double lambda = 0.5;

  Trace<double>* critice = new ATrace<double>(projector->dimension());
  TDLambda<double>* critic = new TDLambda<double>(alpha_v, gamma, lambda, critice);

  PolicyDistribution<double>* policyDistribution = new NormalDistributionScaled<double>(random,
      problem->getContinuousActions(), 0, 1.0, projector->dimension());
  Range<double> policyRange(-2.0, 2.0);
  Range<double> problemRange(-2.0, 2.0);
  PolicyDistribution<double>* acting = new ScaledPolicyDistribution<double>(
      problem->getContinuousActions(), policyDistribution, &policyRange, &problemRange);

  Trace<double>* actore1 = new ATrace<double>(projector->dimension());
  Trace<double>* actore2 = new ATrace<double>(projector->dimension());
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore1);
  actoreTraces->push_back(actore2);
  ActorOnPolicy<double>* actor = new ActorLambda<double>(alpha_u, gamma, lambda, acting,
      actoreTraces);

  OnPolicyControlLearner<double>* control = new AverageRewardActorCritic<double>(critic, actor,
      projector, toStateAction, alpha_r);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  RLRunner<double>* sim = new RLRunner<double>(agent, problem, 5000, 100, 10);
  sim->setVerbose(true);
  sim->run();

  sim->runEvaluate(100);
  sim->computeValueFunction();

  delete random;
  delete problem;
  delete hashing;
  delete projector;
  delete toStateAction;
  delete critice;
  delete critic;
  delete actore1;
  delete actore2;
  delete actoreTraces;
  delete actor;
  delete policyDistribution;
  delete acting;
  delete control;
  delete agent;
  delete sim;
}

void SwingPendulumTest::testOffPACSwingPendulum2()
{
  Random<double>* random = new Random<double>;
  RLProblem<double>* problem = new SwingPendulum<double>;
  Hashing<double>* hashing = new MurmurHashing<double>(random, 1000000);
  Projector<double>* projector = new TileCoderHashing<double>(hashing, problem->dimension(), 10, 10,
      true);
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());

  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_w = .005 / projector->vectorNorm();
  double gamma = 0.99;
  Trace<double>* critice = new AMaxTrace<double>(projector->dimension());
  Trace<double>* criticeML = new MaxLengthTrace<double>(critice, 1000);
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma, 0.4, criticeML);
  double alpha_u = 0.5 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(random,
      problem->getDiscreteActions(), projector->dimension());

  Trace<double>* actore = new AMaxTrace<double>(projector->dimension());
  Trace<double>* actoreML = new MaxLengthTrace<double>(actore, 1000);
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actoreML);
  ActorOffPolicy<double>* actor = new ActorLambdaOffPolicy<double>(alpha_u, gamma, 0.4, target,
      actoreTraces);

  /*Policy<double>* behavior = new RandomPolicy<double>(
   &problem->getActions());*/
  Policy<double>* behavior = new BoltzmannDistribution<double>(random,
      problem->getDiscreteActions(), projector->dimension());
  OffPolicyControlLearner<double>* control = new OffPAC<double>(behavior, critic, actor,
      toStateAction, projector);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  RLRunner<double>* sim = new RLRunner<double>(agent, problem, 5000, 200, 1);
  sim->setTestEpisodesAfterEachRun(true);
  sim->run();

  delete random;
  delete problem;
  delete hashing;
  delete projector;
  delete toStateAction;
  delete critice;
  delete criticeML;
  delete critic;
  delete actore;
  delete actoreML;
  delete actoreTraces;
  delete actor;
  delete behavior;
  delete target;
  delete control;
  delete agent;
  delete sim;
}

void SwingPendulumTest::testOffPACOnPolicySwingPendulum()
{
  Random<double>* random = new Random<double>;
  RLProblem<double>* problem = new SwingPendulum<double>;
  Hashing<double>* hashing = new MurmurHashing<double>(random, 1000);
  Projector<double>* projector = new TileCoderHashing<double>(hashing, problem->dimension(), 10, 10,
      true);
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());

  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_w = .0001 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.4;

  Trace<double>* critice = new ATrace<double>(projector->dimension());
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma, lambda, critice);
  double alpha_u = 0.5 / projector->vectorNorm();
  PolicyDistribution<double>* acting = new BoltzmannDistribution<double>(random,
      problem->getDiscreteActions(), projector->dimension());

  Trace<double>* actore = new ATrace<double>(projector->dimension());
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore);
  ActorOffPolicy<double>* actor = new ActorLambdaOffPolicy<double>(alpha_u, gamma, lambda, acting,
      actoreTraces);

  OffPolicyControlLearner<double>* control = new OffPAC<double>(acting, critic, actor,
      toStateAction, projector);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  RLRunner<double>* sim = new RLRunner<double>(agent, problem, 5000, 10, 5);
  sim->setTestEpisodesAfterEachRun(true);
  sim->run();
  sim->computeValueFunction();

  delete random;
  delete problem;
  delete hashing;
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

void SwingPendulumTest::testOnPolicyBoltzmannATraceNaturalActorCriticSwingPendulum()
{
  Random<double>* random = new Random<double>;
  RLProblem<double>* problem = new SwingPendulum<double>;
  Hashing<double>* hashing = new MurmurHashing<double>(random, 1000);
  Projector<double>* projector = new TileCoderHashing<double>(hashing, problem->dimension(), 10, 10,
      true);
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());

  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_u = 0.001 / projector->vectorNorm();
  double lambda = 0.4;
  double gamma = 0.99;

  Trace<double>* critice = new ATrace<double>(projector->dimension());
  TDLambda<double>* critic = new TDLambda<double>(alpha_v, gamma, lambda, critice);

  PolicyDistribution<double>* acting = new BoltzmannDistribution<double>(random,
      problem->getDiscreteActions(), projector->dimension());

  ActorOnPolicy<double>* actor = new ActorNatural<double>(alpha_u, alpha_v, acting);
  OnPolicyControlLearner<double>* control = new ActorCritic<double>(critic, actor, projector,
      toStateAction);
  RLAgent<double>* agent = new LearnerAgent<double>(control);
  RLRunner<double>* sim = new RLRunner<double>(agent, problem, 5000, 50, 10);
  sim->setTestEpisodesAfterEachRun(true);
  sim->run();
  sim->computeValueFunction();

  delete random;
  delete problem;
  delete hashing;
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

void SwingPendulumTest::run()
{
  testOffPACSwingPendulum();
  testOnPolicySwingPendulum();
  testOffPACSwingPendulum2();
  testOffPACOnPolicySwingPendulum();
  testOnPolicyBoltzmannATraceNaturalActorCriticSwingPendulum();
}

