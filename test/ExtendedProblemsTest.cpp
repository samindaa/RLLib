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
  Env<float>* problem = new MCar3D;
  Projector<double, float>* projector = new TileCoderHashing<double, float>(1000000, 10, true);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<double, float>(
      projector, &problem->getDiscreteActionList());

  double alpha_v = 0.05 / projector->vectorNorm();
  double alpha_w = .0001 / projector->vectorNorm();
  double gamma = 0.99;
  Trace<double>* critice = new ATrace<double>(projector->dimension());
  Trace<double>* criticeML = new MaxLengthTrace<double>(critice, 1000);
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma, 0.4, criticeML);
  double alpha_u = 1.0 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(projector->dimension(),
      &problem->getDiscreteActionList());

  Trace<double>* actore = new ATrace<double>(projector->dimension());
  Trace<double>* actoreML = new MaxLengthTrace<double>(actore, 1000);
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actoreML);
  ActorOffPolicy<double, float>* actor = new ActorLambdaOffPolicy<double, float>(alpha_u, gamma,
      0.4, target, actoreTraces);

  //Policy<double>* behavior = new RandomPolicy<double>(
  //    &problem->getActionList());
  Policy<double>* behavior = new BoltzmannDistribution<double>(projector->dimension(),
      &problem->getDiscreteActionList());
  OffPolicyControlLearner<double, float>* control = new OffPAC<double, float>(behavior, critic,
      actor, toStateAction, projector, gamma);

  Simulator<double, float>* sim = new Simulator<double, float>(control, problem);
  sim->run(20, 5000, 100);

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
  Env<float>* problem = new MCar3D;
  /*Projector<double, float>* projector = new FullTilings<double, float>(1000000,
   10, true);*/
  Projector<double, float>* projector = new MountainCar3DTilesProjector<double, float>();
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<double, float>(
      projector, &problem->getDiscreteActionList());
  Trace<double>* e = new ATrace<double>(projector->dimension(), 0.001);
  Trace<double>* eML = new MaxLengthTrace<double>(e, 2000);
  double alpha_v = 0.2 / projector->vectorNorm();
  double alpha_w = .0001 / projector->vectorNorm();
  double gamma_tp1 = 0.99;
  double beta_tp1 = 1.0 - gamma_tp1;
  double lambda_t = 0.8;
  GQ<double>* gq = new GQ<double>(alpha_v, alpha_w, beta_tp1, lambda_t, eML);
  //double epsilon = 0.01;
  Policy<double>* behavior = new EpsilonGreedy<double>(gq, &problem->getDiscreteActionList(), 0.1);
  /*Policy<double>* behavior = new RandomPolicy<double>(
   &problem->getDiscreteActionList());*/
  Policy<double>* target = new Greedy<double>(gq, &problem->getDiscreteActionList());
  OffPolicyControlLearner<double, float>* control = new GreedyGQ<double, float>(target, behavior,
      &problem->getDiscreteActionList(), toStateAction, gq);

  Simulator<double, float>* sim = new Simulator<double, float>(control, problem);
  sim->run(1, 5000, 3000);
  //sim->computeValueFunction();
  control->persist("visualization/mcar3d_greedy_gq.data");
  control->reset();
  control->resurrect("visualization/mcar3d_greedy_gq.data");
  sim->test(20, 5000);

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
  Env<float>* problem = new MCar3D;
  /*Projector<double, float>* projector = new FullTilings<double, float>(1000000,
   10, false);*/
  Projector<double, float>* projector = new MountainCar3DTilesProjector<double, float>();
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<double, float>(
      projector, &problem->getDiscreteActionList());

  Trace<double>* e = new RTrace<double>(projector->dimension(), 0.001);
  Trace<double>* eML = new MaxLengthTrace<double>(e, 1000);
  double alpha = 0.01 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.6;
  Sarsa<double>* sarsa = new Sarsa<double>(alpha, gamma, lambda, eML);
  double epsilon = 0.1;
  Policy<double>* acting = new EpsilonGreedy<double>(sarsa, &problem->getDiscreteActionList(),
      epsilon);
  OnPolicyControlLearner<double, float>* control = new SarsaControl<double, float>(acting,
      toStateAction, sarsa);

  Simulator<double, float>* sim = new Simulator<double, float>(control, problem);
  sim->run(1, 5000, 1000);

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
  Env<float>* problem = new MCar3D;
  Projector<double, float>* projector = new TileCoderHashing<double, float>(1000000, 10, true);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<double, float>(
      projector, &problem->getDiscreteActionList());

  double alpha_v = 0.01 / projector->vectorNorm();
  double alpha_w = .001 / projector->vectorNorm();
  double gamma = 0.99;
  Trace<double>* critice = new ATrace<double>(projector->dimension());
  Trace<double>* criticeML = new MaxLengthTrace<double>(critice, 1000);
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma, 0.1, criticeML);
  double alpha_u = 1.0 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(projector->dimension(),
      &problem->getDiscreteActionList());

  Trace<double>* actore = new AMaxTrace<double>(projector->dimension());
  Trace<double>* actoreML = new MaxLengthTrace<double>(actore, 1000);
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actoreML);
  ActorOffPolicy<double, float>* actor = new ActorLambdaOffPolicy<double, float>(alpha_u, gamma,
      0.1, target, actoreTraces);

  Policy<double>* behavior = new RandomPolicy<double>(&problem->getDiscreteActionList());
  OffPolicyControlLearner<double, float>* control = new OffPAC<double, float>(behavior, critic,
      actor, toStateAction, projector, gamma);

  Simulator<double, float>* sim = new Simulator<double, float>(control, problem);
  sim->run(1, 5000, 1000);
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
  Env<float>* problem = new Acrobot;
  Projector<double, float>* projector = new AcrobotTilesProjector<double, float>();
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<double, float>(
      projector, &problem->getDiscreteActionList());

  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_w = .0001 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.4;
  Trace<double>* critice = new AMaxTrace<double>(projector->dimension());
  Trace<double>* criticeML = new MaxLengthTrace<double>(critice, 1000);
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma, lambda, criticeML);
  double alpha_u = 0.0001 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(projector->dimension(),
      &problem->getDiscreteActionList());

  Trace<double>* actore = new AMaxTrace<double>(projector->dimension());
  Trace<double>* actoreML = new MaxLengthTrace<double>(actore, 1000);
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actoreML);
  ActorOffPolicy<double, float>* actor = new ActorLambdaOffPolicy<double, float>(alpha_u, gamma,
      lambda, target, actoreTraces);

  /*Policy<double>* behavior = new RandomPolicy<double>(
   &problem->getDiscreteActionList());*/
  Policy<double>* behavior = new RandomBiasPolicy<double>(&problem->getDiscreteActionList());
  OffPolicyControlLearner<double, float>* control = new OffPAC<double, float>(behavior, critic,
      actor, toStateAction, projector, gamma);

  Simulator<double, float>* sim = new Simulator<double, float>(control, problem);
  sim->run(1, 5000, 500);
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
  Env<float>* problem = new Acrobot;
  /*Projector<double, float>* projector = new FullTilings<double, float>(1000000,
   10, true);*/
  Projector<double, float>* projector = new AcrobotTilesProjector<double, float>();
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<double, float>(
      projector, &problem->getDiscreteActionList());
  Trace<double>* e = new ATrace<double>(projector->dimension(), 0.001);
  Trace<double>* eML = new MaxLengthTrace<double>(e, 1000);
  double alpha_v = 0.2 / projector->vectorNorm();
  double alpha_w = .001 / projector->vectorNorm();
  double gamma_tp1 = 0.99;
  double beta_tp1 = 1.0 - gamma_tp1;
  double lambda_t = 0.8;
  GQ<double>* gq = new GQ<double>(alpha_v, alpha_w, beta_tp1, lambda_t, eML);
  //double epsilon = 0.01;
  Policy<double>* behavior = new EpsilonGreedy<double>(gq, &problem->getDiscreteActionList(), 0.1);
  /*Policy<double>* behavior = new RandomPolicy<double>(
   &problem->getDiscreteActionList());*/
  Policy<double>* target = new Greedy<double>(gq, &problem->getDiscreteActionList());
  OffPolicyControlLearner<double, float>* control = new GreedyGQ<double, float>(target, behavior,
      &problem->getDiscreteActionList(), toStateAction, gq);

  Simulator<double, float>* sim = new Simulator<double, float>(control, problem);
  sim->run(1, 5000, 500);
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

  ActionList& actions = poleBalancing.getContinuousActionList();

  for (int r = 0; r < 1; r++)
  {
    cout << "*** start *** " << endl;
    poleBalancing.initialize();
    int round = 0;
    do
    {
      const DenseVector<float>& vars = poleBalancing.getVars();
      for (int i = 0; i < vars.dimension(); i++)
        x[i] = vars[i];
      cout << "x=" << x.transpose() << endl;

      // **** action ***
      VectorXd noise(1);
      noise(0) = Random::nextNormalGaussian() * 0.1;
      VectorXd u = k.transpose() * x + noise;
      actions.update(0, 0, (float) u(0));

      poleBalancing.step(actions.at(0));
      cout << "r=" << poleBalancing.r() << endl;
      ++round;
      cout << "round=" << round << endl;
    } while (!poleBalancing.endOfEpisode());
  }
}

void ExtendedProblemsTest::testPersistResurrect()
{
  srand(time(0));
  SparseVector<float> a(20);
  for (int i = 0; i < 10; i++)
    a.insertEntry(i, Random::nextDouble());
  cout << a << endl;
  a.persist(string("testsv.dat"));

  SparseVector<float> b(20);
  b.resurrect(string("testsv.dat"));
  cout << b << endl;

  DenseVector<float> d(20);
  for (int i = 0; i < 10; i++)
    d[i] = Random::nextDouble();
  cout << d << endl;
  d.persist(string("testdv.dat"));

  DenseVector<float> e(20);
  e.resurrect(string("testdv.dat"));
  cout << e << endl;
}

void ExtendedProblemsTest::testExp()
{
  //cout << exp(200) << endl;
  cout << std::numeric_limits<float>::epsilon() << endl;
  cout << 1e3 << endl;
}

void ExtendedProblemsTest::testEigen3()
{
  using Eigen::MatrixXf;
  using Eigen::VectorXf;
  MatrixXf m(2, 2);
  m(0, 0) = 3;
  m(1, 0) = 2.5;
  m(0, 1) = -1;
  m(1, 1) = m(1, 0) + m(0, 1);
  cout << m << endl;

  //
  cout << "*** diagonal ***" << endl;
  VectorXf d;
  d.resize(4);
  cout << d << endl;
  d << 0.1, 0.1, 0.1, 0.1;
  MatrixXf sigma0 = d.asDiagonal();
  cout << sigma0 << endl;

  //
  cout << "*** sample from multivariate normal ***" << endl;

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

void ExtendedProblemsTest::testCubicSpline()
{
  Spline spline;

  for (double x = -5; x <= 5; x++)
    spline.addPoint(x, sin(x));
  spline.setLowBC(Spline::FIXED_1ST_DERIV_BC, 0);
  spline.setHighBC(Spline::FIXED_1ST_DERIV_BC, 0);

  std::ofstream of("/home/sam/Tmp/CSplines/spline.natural.dat");
  for (double x = -5; x <= 5; x += 0.1)
    of << x << " " << spline(x) << "\n";
  of.close();
}

void ExtendedProblemsTest::testMotion()
{
  Spline spline;
  std::ifstream inf("/home/sam/projects/workspace_robocanes/RLLib/visualization/j_balance_1");

  spline.setLowBC(Spline::FIXED_1ST_DERIV_BC, 0);
  spline.setHighBC(Spline::FIXED_1ST_DERIV_BC, 0);

  map<int, vector<double> > motions;

  std::string str;
  while (std::getline(inf, str))
  {
    //std::cout << str << '\n';
    std::stringstream strstr(str);

    // use stream iterators to copy the stream to the vector as whitespace separated strings
    std::istream_iterator<std::string> it(strstr);
    std::istream_iterator<std::string> end;
    std::vector<std::string> results(it, end);

    std::vector<double> resultsToDouble;
    for (std::vector<std::string>::iterator iter = results.begin(); iter != results.end(); ++iter)
    {
      stringstream ss(*iter);
      double tmp;
      ss >> tmp;
      resultsToDouble.push_back(tmp);
    }
    motions.insert(make_pair(motions.size(), resultsToDouble));
  }

  cout << "*** write some down *** " << endl;
  std::ofstream of("/home/sam/Tmp/CSplines/motions.natural.dat");
  std::ofstream of2("/home/sam/Tmp/CSplines/motions.dat");
  int cTime = 0;
  for (map<int, vector<double> >::const_iterator iter = motions.begin(); iter != motions.end();
      ++iter)
  {
    cTime += iter->second[0];
    spline.addPoint(cTime, iter->second[3]);
  }

  // every 20ms
  int startTime = motions[0][0];
  for (int tt = startTime; tt <= cTime; tt += 20)
  {
    of << tt << " " << spline(tt) << endl;
  }

  cTime = 0;
  for (map<int, vector<double> >::const_iterator iter = motions.begin(); iter != motions.end();
      ++iter)
  {
    cTime += iter->second[0];
    of2 << cTime << " " << iter->second[3] << " " << spline(cTime) << endl;
  }
  of.close();
  of2.close();

}

void ExtendedProblemsTest::run()
{
  testOffPACMountainCar3D_1();
  testGreedyGQMountainCar3D();
  testSarsaMountainCar3D();
  testOffPACMountainCar3D_2();
  testOffPACAcrobot();
  testGreedyGQAcrobot();

  testPoleBalancingPlant();
  testPersistResurrect();
  testExp();
  testEigen3();
  testTorquedPendulum();

  testCubicSpline();
  testMotion();
}

