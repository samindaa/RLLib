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
  Probabilistic::srand(0);
  RLProblem<double>* problem = new MountainCar3D<double>;
  //Projector<double>* projector = new TileCoderHashing<double>(100000, 10, true);
  Projector<double>* projector = new MountainCar3DTilesProjector<double>();
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());

  double alpha_v = 0.01 / projector->vectorNorm();
  double alpha_w = .0001 / projector->vectorNorm();
  double gamma = 0.99;
  Trace<double>* critice = new ATrace<double>(projector->dimension());
  Trace<double>* criticeML = new MaxLengthTrace<double>(critice, 1000);
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma, 0.4, criticeML);
  double alpha_u = 0.5 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(projector->dimension(),
      problem->getDiscreteActions());

  Trace<double>* actore = new ATrace<double>(projector->dimension());
  Trace<double>* actoreML = new MaxLengthTrace<double>(actore, 1000);
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actoreML);
  ActorOffPolicy<double>* actor = new ActorLambdaOffPolicy<double>(alpha_u, gamma, 0.4, target,
      actoreTraces);

  Policy<double>* behavior = new BoltzmannDistributionPerturbed<double>(target->parameters()->at(0),
      problem->getDiscreteActions(), 0.0f, 0.0f);
  OffPolicyControlLearner<double>* control = new OffPAC<double>(behavior, critic, actor,
      toStateAction, projector);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  Simulator<double>* sim = new Simulator<double>(agent, problem, 5000, 100, 1);
  sim->run();
  sim->runEvaluate();

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
  delete agent;
  delete sim;
}

void ExtendedProblemsTest::testGreedyGQMountainCar3D()
{
  Probabilistic::srand(0);
  RLProblem<double>* problem = new MountainCar3D<double>;
  Projector<double>* projector = new MountainCar3DTilesProjector<double>();
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());
  Trace<double>* e = new ATrace<double>(projector->dimension(), 0.001);
  Trace<double>* eML = new MaxLengthTrace<double>(e, 2000);
  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_w = .0001 / projector->vectorNorm();
  double gamma_tp1 = 0.99;
  double beta_tp1 = 1.0 - gamma_tp1;
  double lambda_t = 0.8;
  GQ<double>* gq = new GQ<double>(alpha_v, alpha_w, beta_tp1, lambda_t, eML);
  //double epsilon = 0.01;
  Policy<double>* behavior = new EpsilonGreedy<double>(gq, problem->getDiscreteActions(), 0.1);
  /*Policy<double>* behavior = new RandomPolicy<double>(
   &problem->getDiscreteActions());*/
  Policy<double>* target = new Greedy<double>(gq, problem->getDiscreteActions());
  OffPolicyControlLearner<double>* control = new GreedyGQ<double>(target, behavior,
      problem->getDiscreteActions(), toStateAction, gq);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  Simulator<double>* sim = new Simulator<double>(agent, problem, 5000, 300, 1);
  sim->run();
  //sim->computeValueFunction();
  control->persist("visualization/mcar3d_greedy_gq.data");
  control->reset();
  control->resurrect("visualization/mcar3d_greedy_gq.data");
  sim->runEvaluate();

  delete problem;
  delete projector;
  delete toStateAction;
  delete e;
  delete eML;
  delete gq;
  delete behavior;
  delete target;
  delete control;
  delete agent;
  delete sim;
}

// 3D
void ExtendedProblemsTest::testSarsaMountainCar3D()
{
  Probabilistic::srand(0);
  RLProblem<double>* problem = new MountainCar3D<double>;
  Projector<double>* projector = new MountainCar3DTilesProjector<double>();
  //Projector<double>* projector = new TileCoderHashing<double>(100000, 10, true);
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());

  Trace<double>* e = new ATrace<double>(projector->dimension());
  //Trace<double>* e = new RTrace<double>(projector->dimension(), 0.01);
  Trace<double>* eML = new MaxLengthTrace<double>(e, 2000);
  double gamma = 0.99;
  double lambda = 0.9;
  Sarsa<double>* sarsa = new SarsaAlphaBound<double>(gamma, lambda, eML);
  double epsilon = 0.1;
  Policy<double>* acting = new EpsilonGreedy<double>(sarsa, problem->getDiscreteActions(), epsilon);
  OnPolicyControlLearner<double>* control = new SarsaControl<double>(acting, toStateAction, sarsa);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  Simulator<double>* sim = new Simulator<double>(agent, problem, 5000, 300, 1);
  sim->run();
  sim->runEvaluate();

  delete problem;
  delete projector;
  delete toStateAction;
  delete e;
  delete eML;
  delete sarsa;
  delete acting;
  delete control;
  delete agent;
  delete sim;
}

void ExtendedProblemsTest::testOffPACMountainCar3D_2()
{
  Probabilistic::srand(0);
  RLProblem<double>* problem = new MountainCar3D<double>;
  Hashing* hashing = new UNH(100000);
  Projector<double>* projector = new TileCoderHashing<double>(hashing, 10, true);
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());

  double alpha_v = 0.01 / projector->vectorNorm();
  double alpha_w = .001 / projector->vectorNorm();
  double gamma = 0.99;
  Trace<double>* critice = new ATrace<double>(projector->dimension());
  Trace<double>* criticeML = new MaxLengthTrace<double>(critice, 1000);
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma, 0.1, criticeML);
  double alpha_u = 1.0 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(projector->dimension(),
      problem->getDiscreteActions());

  Trace<double>* actore = new AMaxTrace<double>(projector->dimension());
  Trace<double>* actoreML = new MaxLengthTrace<double>(actore, 1000);
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actoreML);
  ActorOffPolicy<double>* actor = new ActorLambdaOffPolicy<double>(alpha_u, gamma, 0.1, target,
      actoreTraces);

  Policy<double>* behavior = new RandomPolicy<double>(problem->getDiscreteActions());
  OffPolicyControlLearner<double>* control = new OffPAC<double>(behavior, critic, actor,
      toStateAction, projector);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  Simulator<double>* sim = new Simulator<double>(agent, problem, 5000, 1000, 1);
  sim->run();
  sim->computeValueFunction();

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

void ExtendedProblemsTest::testOffPACAcrobot()
{
  Probabilistic::srand(0);
  RLProblem<double>* problem = new Acrobot<double>;
  Projector<double>* projector = new AcrobotTilesProjector<double>();
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());

  double alpha_v = 0.01 / projector->vectorNorm();
  double alpha_w = .0001 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.4;
  Trace<double>* critice = new AMaxTrace<double>(projector->dimension());
  Trace<double>* criticeML = new MaxLengthTrace<double>(critice, 1000);
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma, lambda, criticeML);
  double alpha_u = 0.001 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(projector->dimension(),
      problem->getDiscreteActions());

  Trace<double>* actore = new AMaxTrace<double>(projector->dimension());
  Trace<double>* actoreML = new MaxLengthTrace<double>(actore, 1000);
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actoreML);
  ActorOffPolicy<double>* actor = new ActorLambdaOffPolicy<double>(alpha_u, gamma, lambda, target,
      actoreTraces);

  //Policy<double>* behavior = new RandomPolicy<double>(&problem->getDiscreteActions());
  //Policy<double>* behavior = new RandomBiasPolicy<double>(&problem->getDiscreteActions());
  Policy<double>* behavior = new BoltzmannDistributionPerturbed<double>(target->parameters()->at(0),
      problem->getDiscreteActions(), 0.0f, 0.0f);
  OffPolicyControlLearner<double>* control = new OffPAC<double>(behavior, critic, actor,
      toStateAction, projector);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  Simulator<double>* sim = new Simulator<double>(agent, problem, 5000, 300, 1);
  sim->run();
  sim->runEvaluate();
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
  delete agent;
  delete sim;
}

void ExtendedProblemsTest::testGreedyGQAcrobot()
{
  Probabilistic::srand(0);
  RLProblem<double>* problem = new Acrobot<double>;
  Projector<double>* projector = new AcrobotTilesProjector<double>();
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());
  Trace<double>* e = new ATrace<double>(projector->dimension(), 0.001);
  Trace<double>* eML = new MaxLengthTrace<double>(e, 1000);
  double alpha_v = 0.2 / projector->vectorNorm();
  double alpha_w = .001 / projector->vectorNorm();
  double gamma_tp1 = 0.99;
  double beta_tp1 = 1.0 - gamma_tp1;
  double lambda_t = 0.8;
  GQ<double>* gq = new GQ<double>(alpha_v, alpha_w, beta_tp1, lambda_t, eML);
  //double epsilon = 0.01;
  Policy<double>* behavior = new EpsilonGreedy<double>(gq, problem->getDiscreteActions(), 0.1);
  /*Policy<double>* behavior = new RandomPolicy<double>(
   &problem->getDiscreteActions());*/
  Policy<double>* target = new Greedy<double>(gq, problem->getDiscreteActions());
  OffPolicyControlLearner<double>* control = new GreedyGQ<double>(target, behavior,
      problem->getDiscreteActions(), toStateAction, gq);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  Simulator<double>* sim = new Simulator<double>(agent, problem, 5000, 500, 1);
  sim->run();
  sim->runEvaluate();
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
  delete agent;
  delete sim;
}

void ExtendedProblemsTest::testPoleBalancingPlant()
{
  Probabilistic::srand(0);
  PoleBalancing poleBalancing;
  VectorXd x(4);
  VectorXd k(4);
  k << 10, 15, -90, -25;

  Actions<double>* actions = poleBalancing.getContinuousActions();

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
  Probabilistic::srand(0);
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

void ExtendedProblemsTest::testSupervisedProjector()
{
  /**
   * This test uses regression on three problems, f(x) = sin(x), f(x) = sinc(x), and f(x) = x / pi,
   * to interpret the output, y_tp1, based on the parameter changes.  The independent variable, x_t,
   * has confined to the range [-pi, pi], and it is normalized to [0,1]. We have used a tile coding
   * feature extractor with parameters memory, nbTiling, and resolution. The curve fitting uses LMS rule
   * with a learning rate of alpha / ||norm||, such that ||norm|| is the number of active features.
   * The tile coding function approximator uses Murmur hashing function. We have added a constant bias
   * unit to the features. y_tp1 is perturb with Gaussian with zero mean and standard deviation one.
   *
   * visualization/dout.data contains the outputs and the user can use the following Gnuplot script
   * to visualize the data.
   *
   *
   unset key

   set term wxt 2
   plot "dout.data" using 1:2 with linespoints lt 1, "dout.data" using 1:5 with linespoints lt 2

   set term wxt 3
   plot "dout.data" using 1:3 with linespoints lt 1, "dout.data" using 1:6 with linespoints lt 2

   set term wxt 4
   plot "dout.data" using 1:4 with linespoints lt 1, "dout.data" using 1:7 with linespoints lt 2

   pause -1 "Hit return to continue"

   */
  Probabilistic::srand(time(0));
  // Parameters:
  int memory = 512 / 4;
  int nbTiling = 8;
  double alpha = 0.01;
  double resolution = 4.0f;
  int nbTraingExamples = 200;
  int nbRuns = 8;

  Hashing* hashing = new MurmurHashing(memory);
  Projector<double>* projector = new TileCoderHashing<double>(hashing, nbTiling, true);

  VectorXd trainX = VectorXd::Zero(nbTraingExamples);
  MatrixXd trainY = MatrixXd::Zero(trainX.rows(), 3);

  // Training set: [sin(x), sinc(x), line(x)]
  for (int i = 0; i < trainX.rows(); i++)
  {
    trainX(i) = -M_PI + 2 * M_PI * Probabilistic::nextDouble();
    trainY(i, 0) = sin(trainX(i)) + Probabilistic::nextNormalGaussian();
    trainY(i, 1) = sin(M_PI * trainX(i)) / (M_PI * trainX(i)) + Probabilistic::nextNormalGaussian();
    trainY(i, 2) = trainX(i) / M_PI + Probabilistic::nextNormalGaussian();
  }

  std::vector<LearningAlgorithm<double>*> predictors;
  for (int i = 0; i < 3; i++)
    predictors.push_back(
        new Adaline<double>(hashing->getMemorySize(), alpha / projector->vectorNorm()));

  // Passes through the training set
  Vector<double>* x_t = new PVector<double>(1);
  for (int runs = 0; runs < nbRuns; runs++)
  {
    for (int i = 0; i < trainX.rows(); i++)
    {
      x_t->setEntry(0, resolution * (trainX(i) + M_PI) / (2.0 * M_PI));
      for (int j = 0; j < 3; j++)
        predictors[j]->learn(projector->project(x_t, j), trainY(i, j));
    }
  }

  std::ofstream dout("visualization/dout.data");
  for (double x = -M_PI; x < M_PI; x += 0.05)
  {
    x_t->setEntry(0, resolution * (x + M_PI) / (2.0 * M_PI));
    dout << (x_t->getEntry(0) / resolution) << " " << sin(x) << " " << (sin(M_PI * x) / (M_PI * x))
        << " " << (x / M_PI) << " ";
    for (int j = 0; j < 3; j++)
      dout << predictors[j]->predict(projector->project(x_t, j)) << " ";
    dout << std::endl;
    dout.flush();
  }
  dout.close();

  delete hashing;
  delete projector;
  for (int i = 0; i < 3; i++)
    delete predictors[i];
  delete x_t;

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
  testTorquedPendulum();
  testSupervisedProjector();
}

