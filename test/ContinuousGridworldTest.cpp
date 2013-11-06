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
 * ContinuousGridworldTest.cpp
 *
 *  Created on: May 15, 2013
 *      Author: sam
 */

#include "ContinuousGridworldTest.h"

RLLIB_TEST_MAKE(ContinuousGridworldTest)

void ContinuousGridworldTest::testGreedyGQContinuousGridworld()
{
  srand(time(0));
  Environment<float>* problem = new ContinuousGridworld;
  Projector<double, float>* projector = new TileCoderHashing<double, float>(1000000, 10, false);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<double, float>(
      projector, problem->getDiscreteActionList());
  Trace<double>* e = new ATrace<double>(projector->dimension());
  double alpha_v = 0.01 / projector->vectorNorm();
  double alpha_w = .0001 / projector->vectorNorm();
  double gamma_tp1 = 0.99;
  double beta_tp1 = 1.0 - gamma_tp1;
  double lambda_t = 0.1;
  GQ<double>* gq = new GQ<double>(alpha_v, alpha_w, beta_tp1, lambda_t, e);
  //double epsilon = 0.01;
  /*Policy<double>* behavior = new EpsilonGreedy<double>(gq,
   problem->getDiscreteActionList(), 0.01);*/
  Policy<double>* behavior = new RandomPolicy<double>(problem->getDiscreteActionList());
  Policy<double>* target = new Greedy<double>(gq, problem->getDiscreteActionList());
  OffPolicyControlLearner<double, float>* control = new GreedyGQ<double, float>(target, behavior,
      problem->getDiscreteActionList(), toStateAction, gq);

  Simulator<double, float>* sim = new Simulator<double, float>(control, problem, 5000, 50001, 1);
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
  delete sim;
}

void ContinuousGridworldTest::testOffPACContinuousGridworld()
{
  srand(time(0));
  Environment<float>* problem = new ContinuousGridworld;
  Hashing* hashing = new MurmurHashing;
  Projector<double, float>* projector = new TileCoderHashing<double, float>(1000000, 10, true,
      hashing);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<double, float>(
      projector, problem->getDiscreteActionList());

  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_w = 0.0001 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.4;
  Trace<double>* critice = new ATrace<double>(projector->dimension());
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma, lambda, critice);
  double alpha_u = 0.001 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(projector->dimension(),
      problem->getDiscreteActionList());

  Trace<double>* actore = new ATrace<double>(projector->dimension());
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore);
  ActorOffPolicy<double, float>* actor = new ActorLambdaOffPolicy<double, float>(alpha_u, gamma,
      lambda, target, actoreTraces);

  Policy<double>* behavior = new RandomPolicy<double>(problem->getDiscreteActionList());
  OffPolicyControlLearner<double, float>* control = new OffPAC<double, float>(behavior, critic,
      actor, toStateAction, projector);

  Simulator<double, float>* sim = new Simulator<double, float>(control, problem, 5000, 2000, 5);
  sim->setTestEpisodesAfterEachRun(true);
  //sim->setVerbose(false);
  sim->run();
  sim->computeValueFunction();

  control->persist("visualization/cgw_offpac.data");

  control->reset();
  control->resurrect("visualization/cgw_offpac.data");
  sim->setEpisodes(100);
  sim->setEvaluate(true);
  sim->run();

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
  delete sim;
}

void ContinuousGridworldTest::testOffPACContinuousGridworld2()
{
  srand(time(0));
  Environment<float>* problem = new ContinuousGridworld;
  Projector<double, float>* projector = new TileCoderHashing<double, float>(1000000, 10, true);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<double, float>(
      projector, problem->getDiscreteActionList());

  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_w = 0.0001 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.4;
  Trace<double>* critice = new ATrace<double>(projector->dimension());
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma, lambda, critice);
  double alpha_u = 0.001 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(projector->dimension(),
      problem->getDiscreteActionList());

  Trace<double>* actore = new ATrace<double>(projector->dimension());
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore);
  ActorOffPolicy<double, float>* actor = new ActorLambdaOffPolicy<double, float>(alpha_u, gamma,
      lambda, target, actoreTraces);

  Policy<double>* behavior = new RandomPolicy<double>(problem->getDiscreteActionList());
  OffPolicyControlLearner<double, float>* control = new OffPAC<double, float>(behavior, critic,
      actor, toStateAction, projector);

  Simulator<double, float>* sim = new Simulator<double, float>(control, problem, 5000, 2000, 1);
  //sim->run(5, 5000, 3000);
  sim->setTestEpisodesAfterEachRun(true);
  sim->run();
  //sim->computeValueFunction();

  //control->persist("visualization/cgw_offpac.data");

  //control->reset();
  //control->resurrect("visualization/cgw_offpac.data");
  //sim->test(100, 2000);

  delete problem;
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
  delete sim;
}

void ContinuousGridworldTest::testOffPACOnPolicyContinuousGridworld()
{
  srand(time(0));
  Environment<float>* problem = new ContinuousGridworld;
  Projector<double, float>* projector = new TileCoderHashing<double, float>(1000000, 10, true);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<double, float>(
      projector, problem->getDiscreteActionList());

  double alpha_v = 0.01 / projector->vectorNorm();
  double alpha_w = 0.0001 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda_critic = 0.3;
  double lambda_actor = 0.3;
  Trace<double>* critice = new RTrace<double>(projector->dimension());
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma, lambda_critic,
      critice);
  double alpha_u = 0.001 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(projector->dimension(),
      problem->getDiscreteActionList());

  Trace<double>* actore = new RTrace<double>(projector->dimension());
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore);
  ActorOffPolicy<double, float>* actor = new ActorLambdaOffPolicy<double, float>(alpha_u, gamma,
      lambda_actor, target, actoreTraces);

  Policy<double>* behavior = new BoltzmannDistributionPerturbed<double>(target->parameters()->at(0),
      problem->getDiscreteActionList(), 0.01f, 1.0f);
  OffPolicyControlLearner<double, float>* control = new OffPAC<double, float>(behavior, critic,
      actor, toStateAction, projector);

  Simulator<double, float>* sim = new Simulator<double, float>(control, problem, 5000, 8000, 1);
  sim->run();
  sim->computeValueFunction();

  control->persist("visualization/cgw_offpac.data");

  control->reset();
  control->resurrect("visualization/cgw_offpac.data");
  sim->setEpisodes(100);
  sim->setEvaluate(true);
  sim->run();

  delete problem;
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
  delete sim;
}

void ContinuousGridworldTest::testOffPACContinuousGridworldOPtimized()
{
  srand(time(0));
  Environment<float>* problem = new ContinuousGridworld;
  Projector<double, float>* projector = new TileCoderHashing<double, float>(1000000, 10, true);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<double, float>(
      projector, problem->getDiscreteActionList());

  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_w = 0.0001 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.4;
  Trace<double>* critice = new ATrace<double>(projector->dimension());
  Trace<double>* criticeML = new MaxLengthTrace<double>(critice, 1000);
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma, lambda, criticeML);
  double alpha_u = 0.001 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(projector->dimension(),
      problem->getDiscreteActionList());

  Trace<double>* actore = new ATrace<double>(projector->dimension());
  Trace<double>* actoreML = new MaxLengthTrace<double>(actore, 1000);
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actoreML);
  ActorOffPolicy<double, float>* actor = new ActorLambdaOffPolicy<double, float>(alpha_u, gamma,
      lambda, target, actoreTraces);

  Policy<double>* behavior = new RandomPolicy<double>(problem->getDiscreteActionList());
  OffPolicyControlLearner<double, float>* control = new OffPAC<double, float>(behavior, critic,
      actor, toStateAction, projector);

  Simulator<double, float>* sim = new Simulator<double, float>(control, problem, 5000, 5000, 1);
  sim->run();
  sim->computeValueFunction();

  control->persist("visualization/cgw_offpac.data");

  control->reset();
  control->resurrect("visualization/cgw_offpac.data");
  sim->setEpisodes(100);
  sim->setEvaluate(true);
  sim->run();

  delete problem;
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

