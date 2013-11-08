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
 * ExtendedProblemsTest.cpp
 *
 *  Created on: May 15, 2013
 *      Author: sam
 */

#include "ExtendedProblemsTest.h"

RLLIB_TEST_MAKE(ExtendedProblemsTest)

void ExtendedProblemsTest::testOffPACMountainCar3D_1()
{
  srand(time(0));
  Environment<>* problem = new MCar3D;
  //Projector<double>* projector = new TileCoderHashing<double>(100000, 10, true);
  Projector<double>* projector = new MountainCar3DTilesProjector<double>();
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(
      projector, problem->getDiscreteActionList());

  double alpha_v = 0.01 / projector->vectorNorm();
  double alpha_w = .0001 / projector->vectorNorm();
  double gamma = 0.99;
  Trace<double>* critice = new ATrace<double>(projector->dimension());
  Trace<double>* criticeML = new MaxLengthTrace<double>(critice, 1000);
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma, 0.4, criticeML);
  double alpha_u = 0.5 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(projector->dimension(),
      problem->getDiscreteActionList());

  Trace<double>* actore = new ATrace<double>(projector->dimension());
  Trace<double>* actoreML = new MaxLengthTrace<double>(actore, 1000);
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actoreML);
  ActorOffPolicy<double>* actor = new ActorLambdaOffPolicy<double>(alpha_u, gamma,
      0.4, target, actoreTraces);

  Policy<double>* behavior = new BoltzmannDistributionPerturbed<double>(target->parameters()->at(0),
      problem->getDiscreteActionList(), 0.0f, 0.0f);
  OffPolicyControlLearner<double>* control = new OffPAC<double>(behavior, critic,
      actor, toStateAction, projector);

  Simulator<double>* sim = new Simulator<double>(control, problem, 5000, 100, 1);
  sim->run();
  sim->setEvaluate(true);
  sim->run();

  delete problem;
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
  delete sim;
}

void ExtendedProblemsTest::testGreedyGQMountainCar3D()
{
  srand(time(0));
  Environment<>* problem = new MCar3D;
  Projector<double>* projector = new MountainCar3DTilesProjector<double>();
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(
      projector, problem->getDiscreteActionList());
  Trace<double>* e = new ATrace<double>(projector->dimension(), 0.001);
  Trace<double>* eML = new MaxLengthTrace<double>(e, 2000);
  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_w = .0001 / projector->vectorNorm();
  double gamma_tp1 = 0.99;
  double beta_tp1 = 1.0 - gamma_tp1;
  double lambda_t = 0.8;
  GQ<double>* gq = new GQ<double>(alpha_v, alpha_w, beta_tp1, lambda_t, eML);
  //double epsilon = 0.01;
  Policy<double>* behavior = new EpsilonGreedy<double>(gq, problem->getDiscreteActionList(), 0.1);
  /*Policy<double>* behavior = new RandomPolicy<double>(
   &problem->getDiscreteActionList());*/
  Policy<double>* target = new Greedy<double>(gq, problem->getDiscreteActionList());
  OffPolicyControlLearner<double>* control = new GreedyGQ<double>(target, behavior,
      problem->getDiscreteActionList(), toStateAction, gq);

  Simulator<double>* sim = new Simulator<double>(control, problem, 5000, 300, 1);
  sim->run();
  //sim->computeValueFunction();
  control->persist("visualization/mcar3d_greedy_gq.data");
  control->reset();
  control->resurrect("visualization/mcar3d_greedy_gq.data");
  sim->setEvaluate(true);
  sim->setEpisodes(20);
  sim->run();

  delete problem;
  delete projector;
  delete toStateAction;
  delete e;
  delete eML;
  delete gq;
  delete behavior;
  delete target;
  delete control;
  delete sim;
}

// 3D
void ExtendedProblemsTest::testSarsaMountainCar3D()
{
  srand(time(0));
  Environment<>* problem = new MCar3D;
  Projector<double>* projector = new MountainCar3DTilesProjector<double>();
  //Projector<double>* projector = new TileCoderHashing<double>(100000, 10, true);
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(
      projector, problem->getDiscreteActionList());

  Trace<double>* e = new ATrace<double>(projector->dimension());
  //Trace<double>* e = new RTrace<double>(projector->dimension(), 0.01);
  Trace<double>* eML = new MaxLengthTrace<double>(e, 2000);
  double gamma = 0.99;
  double lambda = 0.9;
  Sarsa<double>* sarsa = new SarsaAlphaBound<double>(gamma, lambda, eML);
  double epsilon = 0.1;
  Policy<double>* acting = new EpsilonGreedy<double>(sarsa, problem->getDiscreteActionList(),
      epsilon);
  OnPolicyControlLearner<double>* control = new SarsaControl<double>(acting,
      toStateAction, sarsa);

  Simulator<double>* sim = new Simulator<double>(control, problem, 5000, 300, 1);
  sim->run();
  sim->setEvaluate(true);
  sim->run();

  delete problem;
  delete projector;
  delete toStateAction;
  delete e;
  delete eML;
  delete sarsa;
  delete acting;
  delete control;
  delete sim;
}

void ExtendedProblemsTest::testOffPACMountainCar3D_2()
{
  srand(time(0));
  Environment<>* problem = new MCar3D;
  Projector<double>* projector = new TileCoderHashing<double>(100000, 10, true);
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(
      projector, problem->getDiscreteActionList());

  double alpha_v = 0.01 / projector->vectorNorm();
  double alpha_w = .001 / projector->vectorNorm();
  double gamma = 0.99;
  Trace<double>* critice = new ATrace<double>(projector->dimension());
  Trace<double>* criticeML = new MaxLengthTrace<double>(critice, 1000);
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma, 0.1, criticeML);
  double alpha_u = 1.0 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(projector->dimension(),
      problem->getDiscreteActionList());

  Trace<double>* actore = new AMaxTrace<double>(projector->dimension());
  Trace<double>* actoreML = new MaxLengthTrace<double>(actore, 1000);
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actoreML);
  ActorOffPolicy<double>* actor = new ActorLambdaOffPolicy<double>(alpha_u, gamma,
      0.1, target, actoreTraces);

  Policy<double>* behavior = new RandomPolicy<double>(problem->getDiscreteActionList());
  OffPolicyControlLearner<double>* control = new OffPAC<double>(behavior, critic,
      actor, toStateAction, projector);

  Simulator<double>* sim = new Simulator<double>(control, problem, 5000, 1000, 1);
  sim->run();
  sim->computeValueFunction();

  delete problem;
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
  delete sim;
}

void ExtendedProblemsTest::testOffPACAcrobot()
{
  srand(time(0));
  Environment<>* problem = new Acrobot;
  Projector<double>* projector = new AcrobotTilesProjector<double>();
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(
      projector, problem->getDiscreteActionList());

  double alpha_v = 0.01 / projector->vectorNorm();
  double alpha_w = .0001 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.4;
  Trace<double>* critice = new AMaxTrace<double>(projector->dimension());
  Trace<double>* criticeML = new MaxLengthTrace<double>(critice, 1000);
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma, lambda, criticeML);
  double alpha_u = 0.001 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(projector->dimension(),
      problem->getDiscreteActionList());

  Trace<double>* actore = new AMaxTrace<double>(projector->dimension());
  Trace<double>* actoreML = new MaxLengthTrace<double>(actore, 1000);
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actoreML);
  ActorOffPolicy<double>* actor = new ActorLambdaOffPolicy<double>(alpha_u, gamma,
      lambda, target, actoreTraces);

  //Policy<double>* behavior = new RandomPolicy<double>(&problem->getDiscreteActionList());
  //Policy<double>* behavior = new RandomBiasPolicy<double>(&problem->getDiscreteActionList());
  Policy<double>* behavior = new BoltzmannDistributionPerturbed<double>(target->parameters()->at(0),
      problem->getDiscreteActionList(), 0.0f, 0.0f);
  OffPolicyControlLearner<double>* control = new OffPAC<double>(behavior, critic,
      actor, toStateAction, projector);

  Simulator<double>* sim = new Simulator<double>(control, problem, 5000, 300, 1);
  sim->run();
  sim->setEvaluate(true);
  sim->run();
  sim->computeValueFunction();

  delete problem;
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
  delete sim;
}

void ExtendedProblemsTest::testGreedyGQAcrobot()
{
  srand(time(0));
  Environment<>* problem = new Acrobot;
  Projector<double>* projector = new AcrobotTilesProjector<double>();
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(
      projector, problem->getDiscreteActionList());
  Trace<double>* e = new ATrace<double>(projector->dimension(), 0.001);
  Trace<double>* eML = new MaxLengthTrace<double>(e, 1000);
  double alpha_v = 0.2 / projector->vectorNorm();
  double alpha_w = .001 / projector->vectorNorm();
  double gamma_tp1 = 0.99;
  double beta_tp1 = 1.0 - gamma_tp1;
  double lambda_t = 0.8;
  GQ<double>* gq = new GQ<double>(alpha_v, alpha_w, beta_tp1, lambda_t, eML);
  //double epsilon = 0.01;
  Policy<double>* behavior = new EpsilonGreedy<double>(gq, problem->getDiscreteActionList(), 0.1);
  /*Policy<double>* behavior = new RandomPolicy<double>(
   &problem->getDiscreteActionList());*/
  Policy<double>* target = new Greedy<double>(gq, problem->getDiscreteActionList());
  OffPolicyControlLearner<double>* control = new GreedyGQ<double>(target, behavior,
      problem->getDiscreteActionList(), toStateAction, gq);

  Simulator<double>* sim = new Simulator<double>(control, problem, 5000, 500, 1);
  sim->run();
  sim->setEvaluate(true);
  sim->run();
  sim->computeValueFunction();

  delete problem;
  delete projector;
  delete toStateAction;
  delete e;
  delete eML;
  delete gq;
  delete behavior;
  delete target;
  delete control;
  delete sim;
}

void ExtendedProblemsTest::testPoleBalancingPlant()
{
  srand(time(0));
  PoleBalancing poleBalancing;
  VectorXd x(4);
  VectorXd k(4);
  k << 10, 15, -90, -25;

  ActionList* actions = poleBalancing.getContinuousActionList();

  for (int r = 0; r < 1; r++)
  {
    cout << "*** start *** " << endl;
    poleBalancing.initialize();
    int round = 0;
    do
    {
      const DenseVector<double>& vars = *poleBalancing.getObservations();
      for (int i = 0; i < vars.dimension(); i++)
        x[i] = vars[i];
      cout << "x=" << x.transpose() << endl;

      // **** action ***
      VectorXd noise(1);
      noise(0) = Probabilistic::nextNormalGaussian() * 0.1;
      VectorXd u = k.transpose() * x + noise;
      actions->update(0, 0, (float) u(0));

      poleBalancing.step(actions->at(0));
      cout << "r=" << poleBalancing.r() << endl;
      ++round;
      cout << "round=" << round << endl;
    } while (!poleBalancing.endOfEpisode());
  }
}

void ExtendedProblemsTest::testPersistResurrect()
{
  srand(time(0));
  SVector<float> a(20);
  for (int i = 0; i < 10; i++)
    a.insertEntry(i, Probabilistic::nextDouble());
  cout << a << endl;
  a.persist(string("testsv.dat"));

  SVector<float> b(20);
  b.resurrect(string("testsv.dat"));
  cout << b << endl;

  PVector<float> d(20);
  for (int i = 0; i < 10; i++)
    d[i] = Probabilistic::nextDouble();
  cout << d << endl;
  d.persist(string("testdv.dat"));

  PVector<float> e(20);
  e.resurrect(string("testdv.dat"));
  cout << e << endl;
}

void ExtendedProblemsTest::testMatrix()
{

  RLLib::Matrix m(2, 2);
  m(0, 0) = 3;
  m(1, 0) = 2.5;
  m(0, 1) = -1;
  m(1, 1) = m(1, 0) + m(0, 1);
  cout << m << endl;

}

void ExtendedProblemsTest::testTorquedPendulum()
{
  double m = 1.0;
  double l = 1.0;
  double mu = 0.01;
  double dt = 0.01;

  cout << "TorquedPendulum u TMAX" << endl;

  double d4 = 2.0;
  double d5 = 50;

  TorquedPendulum torquedPendulum(m, l, mu, dt);

  torquedPendulum.setu(d4);
  torquedPendulum.setx(0, M_PI_2);

  while (torquedPendulum.gett() < d5)
  {
    cout << torquedPendulum.gett() << " " << torquedPendulum.getx(0) << " "
        << torquedPendulum.getx(1) << endl;
    torquedPendulum.nextstep();
    torquedPendulum.nextcopy();
  }
}

void ExtendedProblemsTest::run()
{
  testOffPACMountainCar3D_1();
  testGreedyGQMountainCar3D();
  testSarsaMountainCar3D();
  testOffPACMountainCar3D_2();
  testOffPACAcrobot();
  testGreedyGQAcrobot();

  testMatrix();
  testPoleBalancingPlant();
  //testPersistResurrect();
  testTorquedPendulum();

}

