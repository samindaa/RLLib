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
 * ContinuousGridworldTest.cpp
 *
 *  Created on: May 15, 2013
 *      Author: sam
 */

#include "ContinuousGridworldTest.h"

RLLIB_TEST_MAKE(ContinuousGridworldTest)

void ContinuousGridworldTest::testGreedyGQContinuousGridworld()
{
  Random<double>* random = new Random<double>;
  RLProblem<double>* problem = new ContinuousGridworld<double>(random);
  Hashing<double>* hashing = new MurmurHashing<double>(random, 1000000);
  Projector<double>* projector = new TileCoderHashing<double>(hashing, problem->dimension(), 10, 10,
      false);
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());
  Trace<double>* e = new ATrace<double>(projector->dimension());
  double alpha_v = 0.01 / projector->vectorNorm();
  double alpha_w = .0001 / projector->vectorNorm();
  double gamma_tp1 = 0.99;
  double lambda_t = 0.1;
  GQ<double>* gq = new GQ<double>(alpha_v, alpha_w, gamma_tp1, lambda_t, e);
  //double epsilon = 0.01;
  /*Policy<double>* behavior = new EpsilonGreedy<double>(gq,
   problem->getDiscreteActions(), 0.01);*/
  Policy<double>* behavior = new RandomPolicy<double>(random, problem->getDiscreteActions());
  Policy<double>* target = new Greedy<double>(problem->getDiscreteActions(), gq);
  OffPolicyControlLearner<double>* control = new GreedyGQ<double>(target, behavior,
      problem->getDiscreteActions(), toStateAction, gq);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  RLRunner<double>* sim = new RLRunner<double>(agent, problem, 5000, 50001, 1);
  sim->setTestEpisodesAfterEachRun(true);
  sim->run();
  sim->computeValueFunction();

  delete random;
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

void ContinuousGridworldTest::testOffPACContinuousGridworld()
{
  Random<double>* random = new Random<double>;
  RLProblem<double>* problem = new ContinuousGridworld<double>(random);
  Hashing<double>* hashing = new MurmurHashing<double>(random, 1000000);
  Projector<double>* projector = new TileCoderHashing<double>(hashing, problem->dimension(), 10, 10,
      true);
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());

  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_w = 0.0001 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.4;
  Trace<double>* critice = new ATrace<double>(projector->dimension());
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma, lambda, critice);
  double alpha_u = 0.001 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(random,
      problem->getDiscreteActions(), projector->dimension());

  Trace<double>* actore = new ATrace<double>(projector->dimension());
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore);
  ActorOffPolicy<double>* actor = new ActorLambdaOffPolicy<double>(alpha_u, gamma, lambda, target,
      actoreTraces);

  Policy<double>* behavior = new RandomPolicy<double>(random, problem->getDiscreteActions());
  OffPolicyControlLearner<double>* control = new OffPAC<double>(behavior, critic, actor,
      toStateAction, projector);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  RLRunner<double>* sim = new RLRunner<double>(agent, problem, 5000, 2000, 5);
  sim->setTestEpisodesAfterEachRun(true);
  //sim->setVerbose(false);
  sim->run();
  sim->computeValueFunction();

  control->persist("visualization/cgw_offpac.data");

  control->reset();
  control->resurrect("visualization/cgw_offpac.data");
  sim->runEvaluate(100);

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

void ContinuousGridworldTest::testOffPACContinuousGridworld2()
{
  Random<double>* random = new Random<double>;
  RLProblem<double>* problem = new ContinuousGridworld<double>(random);
  Hashing<double>* hashing = new MurmurHashing<double>(random, 1000000);
  Projector<double>* projector = new TileCoderHashing<double>(hashing, problem->dimension(), 10, 10,
      true);
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());

  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_w = 0.0001 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.4;
  Trace<double>* critice = new ATrace<double>(projector->dimension());
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma, lambda, critice);
  double alpha_u = 0.001 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(random,
      problem->getDiscreteActions(), projector->dimension());

  Trace<double>* actore = new ATrace<double>(projector->dimension());
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore);
  ActorOffPolicy<double>* actor = new ActorLambdaOffPolicy<double>(alpha_u, gamma, lambda, target,
      actoreTraces);

  Policy<double>* behavior = new RandomPolicy<double>(random, problem->getDiscreteActions());
  OffPolicyControlLearner<double>* control = new OffPAC<double>(behavior, critic, actor,
      toStateAction, projector);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  RLRunner<double>* sim = new RLRunner<double>(agent, problem, 5000, 2000, 1);
  //sim->run(5, 5000, 3000);
  sim->setTestEpisodesAfterEachRun(true);
  sim->run();
  //sim->computeValueFunction();

  //control->persist("visualization/cgw_offpac.data");

  //control->reset();
  //control->resurrect("visualization/cgw_offpac.data");
  //sim->test(100, 2000);

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

void ContinuousGridworldTest::testOffPACOnPolicyContinuousGridworld()
{
  Random<double>* random = new Random<double>;
  RLProblem<double>* problem = new ContinuousGridworld<double>(random);
  Hashing<double>* hashing = new MurmurHashing<double>(random, 1000000);
  Projector<double>* projector = new TileCoderHashing<double>(hashing, problem->dimension(), 10, 10,
      true);
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());

  double alpha_v = 0.01 / projector->vectorNorm();
  double alpha_w = 0.0001 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda_critic = 0.3;
  double lambda_actor = 0.3;
  Trace<double>* critice = new RTrace<double>(projector->dimension());
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma, lambda_critic,
      critice);
  double alpha_u = 0.001 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(random,
      problem->getDiscreteActions(), projector->dimension());

  Trace<double>* actore = new RTrace<double>(projector->dimension());
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore);
  ActorOffPolicy<double>* actor = new ActorLambdaOffPolicy<double>(alpha_u, gamma, lambda_actor,
      target, actoreTraces);

  Policy<double>* behavior = new BoltzmannDistributionPerturbed<double>(random,
      problem->getDiscreteActions(), target->parameters()->getEntry(0), 0.01f, 1.0f);
  OffPolicyControlLearner<double>* control = new OffPAC<double>(behavior, critic, actor,
      toStateAction, projector);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  RLRunner<double>* sim = new RLRunner<double>(agent, problem, 5000, 8000, 1);
  sim->run();
  sim->computeValueFunction();

  control->persist("visualization/cgw_offpac.data");

  control->reset();
  control->resurrect("visualization/cgw_offpac.data");
  sim->runEvaluate(100);

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
  delete target;
  delete behavior;
  delete control;
  delete agent;
  delete sim;
}

void ContinuousGridworldTest::testOffPACContinuousGridworldOPtimized()
{
  Random<double>* random = new Random<double>;
  RLProblem<double>* problem = new ContinuousGridworld<double>(random);
  Hashing<double>* hashing = new MurmurHashing<double>(random, 1000000);
  Projector<double>* projector = new TileCoderHashing<double>(hashing, problem->dimension(), 10, 10,
      true);
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());

  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_w = 0.0001 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.4;
  Trace<double>* critice = new ATrace<double>(projector->dimension());
  Trace<double>* criticeML = new MaxLengthTrace<double>(critice, 1000);
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma, lambda, criticeML);
  double alpha_u = 0.001 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(random,
      problem->getDiscreteActions(), projector->dimension());

  Trace<double>* actore = new ATrace<double>(projector->dimension());
  Trace<double>* actoreML = new MaxLengthTrace<double>(actore, 1000);
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actoreML);
  ActorOffPolicy<double>* actor = new ActorLambdaOffPolicy<double>(alpha_u, gamma, lambda, target,
      actoreTraces);

  Policy<double>* behavior = new RandomPolicy<double>(random, problem->getDiscreteActions());
  OffPolicyControlLearner<double>* control = new OffPAC<double>(behavior, critic, actor,
      toStateAction, projector);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  RLRunner<double>* sim = new RLRunner<double>(agent, problem, 5000, 5000, 1);
  sim->run();
  sim->computeValueFunction();

  control->persist("visualization/cgw_offpac.data");

  control->reset();
  control->resurrect("visualization/cgw_offpac.data");
  sim->runEvaluate(100);

  delete random;
  delete problem;
  delete hashing;
  delete projector;
  delete toStateAction;
  delete critice;
  delete criticeML;
  delete critic;
  delete actore;
  delete actor;
  delete actoreML;
  delete actoreTraces;
  delete behavior;
  delete target;
  delete control;
  delete agent;
  delete sim;
}

void ContinuousGridworldTest::run()
{
  testGreedyGQContinuousGridworld();
  testOffPACContinuousGridworld();
  testOffPACContinuousGridworld2();
  testOffPACOnPolicyContinuousGridworld();
  testOffPACContinuousGridworldOPtimized();
}

