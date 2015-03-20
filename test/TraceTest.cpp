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
 * TraceTest.cpp
 *
 *  Created on: Dec 18, 2012
 *      Author: sam
 */

#include "TraceTest.h"

RLLIB_TEST_MAKE(TraceTest)

void TraceTest::runTest(Random<double>* random, RLProblem<double>* problem,
    Projector<double>* projector, Trace<double>* trace)
{
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());
  double alpha = 0.2 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.3;
  double epsilon = 0.01;
  Sarsa<double>* sarsa = new Sarsa<double>(alpha, gamma, lambda, trace);
  Policy<double>* acting = new EpsilonGreedy<double>(random, problem->getDiscreteActions(), sarsa,
      epsilon);
  OnPolicyControlLearner<double>* control = new SarsaControl<double>(acting, toStateAction, sarsa);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  RLRunner<double>* sim = new RLRunner<double>(agent, problem, 5000, 500, 1);
  RLRunner<double>::Event* performanceVerifier = new PerformanceVerifier<double>();
  sim->onEpisodeEnd.push_back(performanceVerifier);
  sim->setVerbose(false);
  sim->run();

  delete toStateAction;
  delete sarsa;
  delete acting;
  delete control;
  delete agent;
  delete sim;
  delete performanceVerifier;
}

void TraceTest::testSarsaOnMountainCarSVectorTraces()
{
  Random<double>* random = new Random<double>;
  RLProblem<double>* problem = new MountainCar<double>();
  Hashing<double>* hashing = new UNH<double>(random, 10000);
  Projector<double>* projector = new TileCoderHashing<double>(hashing, problem->dimension(), 9, 10,
      false);

  Trace<double>* e = new ATrace<double>(projector->dimension());
  runTest(random, problem, projector, e);
  delete e;

  e = new AMaxTrace<double>(projector->dimension());
  runTest(random, problem, projector, e);
  delete e;

  e = new RTrace<double>(projector->dimension());
  runTest(random, problem, projector, e);
  delete e;

  delete random;
  delete problem;
  delete hashing;
  delete projector;

}

void TraceTest::testSarsaOnMountainCarMaxLengthTraces()
{
  Random<double>* random = new Random<double>;
  RLProblem<double>* problem = new MountainCar<double>();
  Hashing<double>* hashing = new MurmurHashing<double>(random, 10000);
  Projector<double>* projector = new TileCoderHashing<double>(hashing, problem->dimension(), 9, 10,
      false);
  Trace<double>* e = new ATrace<double>(projector->dimension());
  Trace<double>* trace = new MaxLengthTrace<double>(e, 100);
  runTest(random, problem, projector, e);
  delete trace;
  delete e;

  e = new AMaxTrace<double>(projector->dimension());
  trace = new MaxLengthTrace<double>(e, 100);
  runTest(random, problem, projector, e);
  delete trace;
  delete e;

  e = new RTrace<double>(projector->dimension());
  trace = new MaxLengthTrace<double>(e, 100);
  runTest(random, problem, projector, e);
  delete trace;
  delete e;

  delete random;
  delete problem;
  delete hashing;
  delete projector;
}

void TraceTest::run()
{
  testATrace();
  testRTrace();
  testAMaxTrace();
  testSarsaOnMountainCarSVectorTraces();
  testSarsaOnMountainCarMaxLengthTraces();
}

