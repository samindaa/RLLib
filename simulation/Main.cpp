//============================================================================
// Name        : RLLib.cpp
// Author      : Sam Abeyruwan
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <map>
#include <fstream>

// From the RLLib
#include "../src/Vector.h"
#include "../src/Trace.h"
#include "../src/Projector.h"
#include "../src/ControlAlgorithm.h"
#include "../src/Representation.h"
#include "../src/SupervisedAlgorithm.h"
#include "Simulator.h"
#include "MCar2D.h"
#include "MCar3D.h"
#include "SwingPendulum.h"
#include "ContinuousGridworld.h"

using namespace std;
using namespace RLLib;

void testFullVector()
{
  DenseVector<float> v(10);
  cout << v << endl;
  srand48(time(0));
  for (int i = 0; i < v.dimension(); i++)
  {
    double k = drand48();
    v[i] = k;
    cout << k << " ";
  }
  cout << endl;
  cout << v << endl;
  DenseVector<float> d;
  cout << d << endl;
  d = v;
  d * 100;
  cout << d << endl;
  cout << d.maxNorm() << endl;

  DenseVector<float> i(5);
  i[0] = 1.0;
  cout << i << endl;
  cout << i.maxNorm() << endl;
  cout << i.euclideanNorm() << endl;
}

void testSparseVector()
{
  srand48(time(0));
  /*SparseVector<> s(16);
   cout << s << endl;

   for (int i = 0; i < s.dimension() ; i++)
   {
   double k = drand48();
   s.insertEntry(i, k);
   cout << "[i=" << i << " v=" << k << "] ";
   }
   cout << endl;
   cout << s << endl;*/

  SparseVector<float> a(20);
  SparseVector<float> b(20);
  for (int i = 0; i < 5; i++)
  {
    a.insertEntry(i, 1);
    b.insertEntry(i, 2);
  }

  cout << a << endl;
  cout << b << endl;
  cout << a.numActiveEntries() << " " << b.numActiveEntries() << endl;
  b.removeEntry(2);
  cout << a.numActiveEntries() << " " << b.numActiveEntries() << endl;
  cout << a << endl;
  cout << b << endl;
  cout << "dot=" << a.dot(b) << endl;
  cout << a.addToSelf(b) << endl;
  a.clear();
  b.clear();
  cout << a << endl;
  cout << b << endl;

}

void testProjector()
{
  srand48(time(0));

  int numObservations = 2;
  int memorySize = 512;
  int numTiling = 32;
  SparseVector<double> w(memorySize);
  for (int t = 0; t < 50; t++)
    w.insertEntry(rand() % memorySize, drand48());
  FullTilings<double, float> coder(memorySize, numTiling, true);
  DenseVector<float> x(numObservations);
  for (int p = 0; p < 5; p++)
  {
    for (int o = 0; o < numObservations; o++)
      x[o] = drand48() / 0.25;
    const SparseVector<double>& vect = coder.project(x);
    cout << w << endl;
    cout << vect << endl;
    cout << w.dot(vect) << endl;
    cout << "---------" << endl;
  }
}

void testProjectorMachineLearning()
{
  // simple sine curve estimation
  // training samples
  srand48(time(0));
  multimap<double, double> X;
  for (int i = 0; i < 100; i++)
  {
    double x = -M_PI_2 + 2 * M_PI * drand48(); // @@>> input noise?
    double y = sin(2 * x); // @@>> output noise?
    X.insert(make_pair(x, y));
  }

  // train
  int numObservations = 1;
  int memorySize = 512;
  int numTiling = 32;
  FullTilings<double, float> coder(memorySize, numTiling, true);
  DenseVector<float> x(numObservations);
  Adaline<double> lms(coder.dimension(), 0.1 / coder.vectorNorm());
  int traininCounter = 0;
  while (++traininCounter < 100)
  {
    for (multimap<double, double>::const_iterator iter = X.begin();
        iter != X.end(); ++iter)
    {
      x[0] = iter->first / (2 * M_PI) / 0.25; // normalized and unit generalized
      const SparseVector<double>& phi = coder.project(x);
      lms.learn(phi, iter->second);
    }
  }

  // output
  ofstream outFile("visualization/mest.dat");
  for (multimap<double, double>::const_iterator iter = X.begin();
      iter != X.end(); ++iter)
  {
    x[0] = iter->first / (2 * M_PI) / 0.25;
    const SparseVector<double>& phi = coder.project(x);
    if (outFile.is_open())
      outFile << iter->first << " " << iter->second << " " << lms.predict(phi)
          << endl;
  }
  outFile.close();
}

void testSarsaMountainCar()
{
  Env<float>* problem = new MCar2D;
  Projector<double, float>* projector = new FullTilings<double, float>(1000000,
      10, false);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<
      double, float>(projector, &problem->getDiscreteActionList());
  Trace<double>* e = new RTrace<double>(projector->dimension());
  double alpha = 0.25 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.9;
  Sarsa<double>* sarsa = new Sarsa<double>(alpha, gamma, lambda, e);
  double epsilon = 0.01;
  Policy<double>* acting = new EpsilonGreedy<double>(sarsa,
      &problem->getDiscreteActionList(), epsilon);
  OnPolicyControlLearner<double, float>* control = new SarsaControl<double,
      float>(acting, toStateAction, sarsa);

  Simulator<double, float>* sim = new Simulator<double, float>(control,
      problem);
  sim->run(10, 5000, 100);
  sim->computeValueFunction();

  delete problem;
  delete projector;
  delete toStateAction;
  delete e;
  delete sarsa;
  delete acting;
  delete control;
  delete sim;
}

void testExpectedSarsaMountainCar()
{
  Env<float>* problem = new MCar2D;
  Projector<double, float>* projector = new FullTilings<double, float>(1000000,
      10, false);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<
      double, float>(projector, &problem->getDiscreteActionList());
  Trace<double>* e = new RTrace<double>(projector->dimension());
  double alpha = 0.1 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.9;
  Sarsa<double>* sarsa = new Sarsa<double>(alpha, gamma, lambda, e);
  double epsilon = 0.01;
  Policy<double>* acting = new EpsilonGreedy<double>(sarsa,
      &problem->getDiscreteActionList(), epsilon);
  OnPolicyControlLearner<double, float>* control = new ExpectedSarsaControl<
      double, float>(acting, toStateAction, sarsa,
      &problem->getDiscreteActionList());

  Simulator<double, float>* sim = new Simulator<double, float>(control,
      problem);
  sim->run(5, 5000, 100);
  sim->computeValueFunction();

  delete problem;
  delete projector;
  delete toStateAction;
  delete e;
  delete sarsa;
  delete acting;
  delete control;
  delete sim;
}

void testGreedyGQMountainCar()
{
  srand(time(0));
  srand48(time(0));
  Env<float>* problem = new MCar2D;
  Projector<double, float>* projector = new FullTilings<double, float>(1000000,
      10, true);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<
      double, float>(projector, &problem->getDiscreteActionList());
  Trace<double>* e = new ATrace<double>(projector->dimension());
  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_w = .0001 / projector->vectorNorm();
  double gamma_tp1 = 0.99;
  double beta_tp1 = 1.0 - gamma_tp1;
  double lambda_t = 0.4;
  GQ<double>* gq = new GQ<double>(alpha_v, alpha_w, beta_tp1, lambda_t, e);
  //double epsilon = 0.01;
  //Policy<double>* behavior = new EpsilonGreedy<double>(gq,
  //    &problem->getActionList(), epsilon);
  Policy<double>* behavior = new RandomPolicy<double>(
      &problem->getDiscreteActionList());
  Policy<double>* target = new Greedy<double>(gq,
      &problem->getDiscreteActionList());
  OffPolicyControlLearner<double, float>* control = new GreedyGQ<double, float>(
      target, behavior, &problem->getDiscreteActionList(), toStateAction, gq);

  Simulator<double, float>* sim = new Simulator<double, float>(control,
      problem);
  sim->run(10, 5000, 100);
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

void testOffPACMountainCar()
{
  srand(time(0));
  srand48(time(0));
  Env<float>* problem = new MCar2D;
  Projector<double, float>* projector = new FullTilings<double, float>(1000000,
      10, true);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<
      double, float>(projector, &problem->getDiscreteActionList());

  double alpha_v = 0.05 / projector->vectorNorm();
  double alpha_w = .0001 / projector->vectorNorm();
  double lambda = 0.4;
  double gamma = 0.99;
  Trace<double>* critice = new ATrace<double>(projector->dimension());
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma,
      lambda, critice);
  double alpha_u = 1.0 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(
      projector->dimension(), &problem->getDiscreteActionList());

  Trace<double>* actore = new ATrace<double>(projector->dimension());
  ActorOffPolicy<double, float>* actor =
      new ActorLambdaOffPolicy<double, float>(alpha_u, gamma, lambda, target,
          actore);

  Policy<double>* behavior = new RandomPolicy<double>(
      &problem->getDiscreteActionList());

  OffPolicyControlLearner<double, float>* control = new OffPAC<double, float>(
      behavior, critic, actor, toStateAction, projector, gamma);

  Simulator<double, float>* sim = new Simulator<double, float>(control,
      problem);
  sim->run(1, 5000, 100);
  sim->computeValueFunction();
  control->persist("visualization/mcar_offpac.data");

  control->reset();
  control->resurrect("visualization/mcar_offpac.data");
  sim->test(20, 5000);

  delete problem;
  delete projector;
  delete toStateAction;
  delete critice;
  delete critic;
  delete actore;
  delete actor;
  delete behavior;
  delete target;
  delete control;
  delete sim;
}

void testGreedyGQContinuousGridworld()
{
  srand(time(0));
  srand48(time(0));
  Env<float>* problem = new ContinuousGridworld;
  Projector<double, float>* projector = new FullTilings<double, float>(1000000,
      10, false);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<
      double, float>(projector, &problem->getDiscreteActionList());
  Trace<double>* e = new ATrace<double>(projector->dimension());
  double alpha_v = 0.01 / projector->vectorNorm();
  double alpha_w = .0001 / projector->vectorNorm();
  double gamma_tp1 = 0.99;
  double beta_tp1 = 1.0 - gamma_tp1;
  double lambda_t = 0.1;
  GQ<double>* gq = new GQ<double>(alpha_v, alpha_w, beta_tp1, lambda_t, e);
  //double epsilon = 0.01;
  /*Policy<double>* behavior = new EpsilonGreedy<double>(gq,
   &problem->getDiscreteActionList(), 0.01);*/
  Policy<double>* behavior = new RandomPolicy<double>(
      &problem->getDiscreteActionList());
  Policy<double>* target = new Greedy<double>(gq,
      &problem->getDiscreteActionList());
  OffPolicyControlLearner<double, float>* control = new GreedyGQ<double, float>(
      target, behavior, &problem->getDiscreteActionList(), toStateAction, gq);

  Simulator<double, float>* sim = new Simulator<double, float>(control,
      problem);
  sim->run(1, 5000, 5000);
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

void testOffPACContinuousGridworld()
{
  srand(time(0));
  srand48(time(0));
  Env<float>* problem = new ContinuousGridworld;
  Projector<double, float>* projector = new FullTilings<double, float>(1000000,
      10, true);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<
      double, float>(projector, &problem->getDiscreteActionList());

  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_w = 0.0001 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.4;
  Trace<double>* critice = new ATrace<double>(projector->dimension());
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma,
      lambda, critice);
  double alpha_u = 0.001 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(
      projector->dimension(), &problem->getDiscreteActionList());

  Trace<double>* actore = new ATrace<double>(projector->dimension());
  ActorOffPolicy<double, float>* actor =
      new ActorLambdaOffPolicy<double, float>(alpha_u, gamma, lambda, target,
          actore);

  Policy<double>* behavior = new RandomPolicy<double>(
      &problem->getDiscreteActionList());
  OffPolicyControlLearner<double, float>* control = new OffPAC<double, float>(
      behavior, critic, actor, toStateAction, projector, gamma);

  Simulator<double, float>* sim = new Simulator<double, float>(control,
      problem);
  sim->run(1, 5000, 5000);
  sim->computeValueFunction();

  delete problem;
  delete projector;
  delete toStateAction;
  delete critice;
  delete critic;
  delete actore;
  delete actor;
  delete behavior;
  delete target;
  delete control;
  delete sim;
}

void testOffPACMountainCar2()
{
  srand(time(0));
  srand48(time(0));
  Env<float>* problem = new MCar2D;
  Projector<double, float>* projector = new FullTilings<double, float>(1000000,
      10, true);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<
      double, float>(projector, &problem->getDiscreteActionList());

  double alpha_v = 0.05 / projector->vectorNorm();
  double alpha_w = .0001 / projector->vectorNorm();
  double gamma = 0.99;
  Trace<double>* critice = new ATrace<double>(projector->dimension());
  Trace<double>* criticeML = new MaxLengthTrace<double>(critice, 1000);
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma,
      0.4, criticeML);
  double alpha_u = 1.0 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(
      projector->dimension(), &problem->getDiscreteActionList());

  Trace<double>* actore = new ATrace<double>(projector->dimension());
  Trace<double>* actoreML = new MaxLengthTrace<double>(actore, 1000);
  ActorOffPolicy<double, float>* actor =
      new ActorLambdaOffPolicy<double, float>(alpha_u, gamma, 0.4, target,
          actoreML);

  //Policy<double>* behavior = new RandomPolicy<double>(
  //    &problem->getActionList());
  Policy<double>* behavior = new BoltzmannDistribution<double>(
      projector->dimension(), &problem->getDiscreteActionList());
  OffPolicyControlLearner<double, float>* control = new OffPAC<double, float>(
      behavior, critic, actor, toStateAction, projector, gamma);

  Simulator<double, float>* sim = new Simulator<double, float>(control,
      problem);
  sim->run(20, 5000, 100);

  delete problem;
  delete projector;
  delete toStateAction;
  delete critice;
  delete criticeML;
  delete critic;
  delete actore;
  delete actoreML;
  delete actor;
  delete behavior;
  delete target;
  delete control;
  delete sim;
}

void testGreedyGQMountainCar2()
{
  srand(time(0));
  srand48(time(0));
  Env<float>* problem = new MCar3D;
  Projector<double, float>* projector = new FullTilings<double, float>(1000000,
      10, true);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<
      double, float>(projector, &problem->getDiscreteActionList());
  Trace<double>* e = new ATrace<double>(projector->dimension(), 0.001);
  Trace<double>* eML = new MaxLengthTrace<double>(e, 1000);
  double alpha_v = 0.125 / projector->vectorNorm();
  double alpha_w = .0001 / projector->vectorNorm();
  double gamma_tp1 = 0.99;
  double beta_tp1 = 1.0 - gamma_tp1;
  double lambda_t = 0.4;
  GQ<double>* gq = new GQ<double>(alpha_v, alpha_w, beta_tp1, lambda_t, eML);
  //double epsilon = 0.01;
  Policy<double>* behavior = new EpsilonGreedy<double>(gq,
      &problem->getDiscreteActionList(), 0.1);
  /*Policy<double>* behavior = new RandomPolicy<double>(
   &problem->getDiscreteActionList());*/
  Policy<double>* target = new Greedy<double>(gq,
      &problem->getDiscreteActionList());
  OffPolicyControlLearner<double, float>* control = new GreedyGQ<double, float>(
      target, behavior, &problem->getDiscreteActionList(), toStateAction, gq);

  Simulator<double, float>* sim = new Simulator<double, float>(control,
      problem);
  sim->run(1, 2000, 5000);
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

// 3D
void testSarsaMountainCar3D()
{
  Env<float>* problem = new MCar3D;
  Projector<double, float>* projector = new FullTilings<double, float>(1000000,
      10, false);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<
      double, float>(projector, &problem->getDiscreteActionList());

  Trace<double>* e = new RTrace<double>(projector->dimension(), 0.001);
  Trace<double>* eML = new MaxLengthTrace<double>(e, 1000);
  double alpha = 0.15 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.8;
  Sarsa<double>* sarsa = new Sarsa<double>(alpha, gamma, lambda, eML);
  double epsilon = 0.1;
  Policy<double>* acting = new EpsilonGreedy<double>(sarsa,
      &problem->getDiscreteActionList(), epsilon);
  OnPolicyControlLearner<double, float>* control = new SarsaControl<double,
      float>(acting, toStateAction, sarsa);

  Simulator<double, float>* sim = new Simulator<double, float>(control,
      problem);
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

void testOffPACMountainCar3D()
{
  srand(time(0));
  srand48(time(0));
  Env<float>* problem = new MCar3D;
  Projector<double, float>* projector = new FullTilings<double, float>(1000000,
      10, true);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<
      double, float>(projector, &problem->getDiscreteActionList());

  double alpha_v = 0.01 / projector->vectorNorm();
  double alpha_w = .001 / projector->vectorNorm();
  double gamma = 0.99;
  Trace<double>* critice = new ATrace<double>(projector->dimension());
  Trace<double>* criticeML = new MaxLengthTrace<double>(critice, 1000);
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma,
      0.1, criticeML);
  double alpha_u = 1.0 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(
      projector->dimension(), &problem->getDiscreteActionList());

  Trace<double>* actore = new AMaxTrace<double>(projector->dimension());
  Trace<double>* actoreML = new MaxLengthTrace<double>(actore, 1000);
  ActorOffPolicy<double, float>* actor =
      new ActorLambdaOffPolicy<double, float>(alpha_u, gamma, 0.1, target,
          actoreML);

  Policy<double>* behavior = new RandomPolicy<double>(
      &problem->getDiscreteActionList());
  OffPolicyControlLearner<double, float>* control = new OffPAC<double, float>(
      behavior, critic, actor, toStateAction, projector, gamma);

  Simulator<double, float>* sim = new Simulator<double, float>(control,
      problem);
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
  delete actor;
  delete behavior;
  delete target;
  delete control;
  delete sim;
}

void testOffPACSwingPendulum()
{
  srand(time(0));
  srand48(time(0));
  Env<float>* problem = new SwingPendulum;
  Projector<double, float>* projector = new FullTilings<double, float>(1000000,
      10, true);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<
      double, float>(projector, &problem->getDiscreteActionList());

  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_w = .0001 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.4;
  Trace<double>* critice = new AMaxTrace<double>(projector->dimension());
  Trace<double>* criticeML = new MaxLengthTrace<double>(critice, 1000);
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma,
      lambda, criticeML);
  double alpha_u = 0.5 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(
      projector->dimension(), &problem->getDiscreteActionList());

  Trace<double>* actore = new AMaxTrace<double>(projector->dimension());
  Trace<double>* actoreML = new MaxLengthTrace<double>(actore, 1000);
  ActorOffPolicy<double, float>* actor =
      new ActorLambdaOffPolicy<double, float>(alpha_u, gamma, lambda, target,
          actoreML);

  /*Policy<double>* behavior = new RandomPolicy<double>(
   &problem->getDiscreteActionList());*/
  Policy<double>* behavior = new RandomBiasPolicy<double>(
      &problem->getDiscreteActionList());
  OffPolicyControlLearner<double, float>* control = new OffPAC<double, float>(
      behavior, critic, actor, toStateAction, projector, gamma);

  Simulator<double, float>* sim = new Simulator<double, float>(control,
      problem);
  sim->run(1, 3000, 1);
  sim->computeValueFunction();

  delete problem;
  delete projector;
  delete toStateAction;
  delete critice;
  delete criticeML;
  delete critic;
  delete actore;
  delete actoreML;
  delete actor;
  delete behavior;
  delete target;
  delete control;
  delete sim;
}

void testOnPolicyCar()
{
  srand(time(0));
  srand48(time(0));
  Env<float>* problem = new MCar2D;
  Projector<double, float>* projector = new FullTilings<double, float>(1000000,
      10, true);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<
      double, float>(projector, &problem->getContinuousActionList());

  double alpha_v = 1.0 / projector->vectorNorm();
  double gamma = 1.0;
  double lambda = 0.9;
  Trace<double>* critice = new AMaxTrace<double>(projector->dimension());
  TDLambda<double>* critic = new TDLambda<double>(alpha_v, gamma, lambda,
      critice);
  double alpha_u = 0.001 / projector->vectorNorm();
  PolicyDistribution<double>* acting = new NormalDistribution<double>(0, 1.0,
      projector->dimension(), &problem->getContinuousActionList());

  Trace<double>* actore = new AMaxTrace<double>(2 * projector->dimension());
  ActorOnPolicy<double, float>* actor = new Actor<double, float>(alpha_u, gamma,
      lambda, acting, actore);

  OnPolicyControlLearner<double, float>* control = new AverageRewardActorCritic<
      double, float>(critic, actor, toStateAction, 0);

  Simulator<double, float>* sim = new Simulator<double, float>(control,
      problem);
  sim->run(1, 5000, 100);
  sim->computeValueFunction();

  delete problem;
  delete projector;
  delete toStateAction;
  delete critice;
  delete critic;
  delete actore;
  delete actor;
  delete acting;
  delete control;
  delete sim;
}

void testOnPolicySwingPendulum()
{
  srand(time(0));
  srand48(time(0));
  Env<float>* problem = new SwingPendulum;
  Projector<double, float>* projector = new FullTilings<double, float>(1000000,
      10, true);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<
      double, float>(projector, &problem->getContinuousActionList());

  double alpha_v = 0.1 / projector->vectorNorm();
  double gamma = 1.0;
  double lambda = 0.5;
  Trace<double>* critice = new AMaxTrace<double>(projector->dimension());
  TDLambda<double>* critic = new TDLambda<double>(alpha_v, gamma, lambda,
      critice);
  double alpha_u = 0.001 / projector->vectorNorm();
  PolicyDistribution<double>* acting = new NormalDistribution<double>(0, 1.0,
      projector->dimension(), &problem->getContinuousActionList());

  Trace<double>* actore = new AMaxTrace<double>(2 * projector->dimension());
  ActorOnPolicy<double, float>* actor = new Actor<double, float>(alpha_u, gamma,
      lambda, acting, actore);

  OnPolicyControlLearner<double, float>* control = new AverageRewardActorCritic<
      double, float>(critic, actor, toStateAction, .0001);

  Simulator<double, float>* sim = new Simulator<double, float>(control,
      problem);
  sim->run(1, 1000, 1000);

  delete problem;
  delete projector;
  delete toStateAction;
  delete critice;
  delete critic;
  delete actore;
  delete actor;
  delete acting;
  delete control;
  delete sim;
}

void testOffPACSwingPendulum2()
{
  srand(time(0));
  srand48(time(0));
  Env<float>* problem = new SwingPendulum;
  Projector<double, float>* projector = new FullTilings<double, float>(1000000,
      10, true);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<
      double, float>(projector, &problem->getDiscreteActionList());

  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_w = .005 / projector->vectorNorm();
  double gamma = 0.99;
  Trace<double>* critice = new AMaxTrace<double>(projector->dimension());
  Trace<double>* criticeML = new MaxLengthTrace<double>(critice, 1000);
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma,
      0.4, criticeML);
  double alpha_u = 0.5 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(
      projector->dimension(), &problem->getDiscreteActionList());

  Trace<double>* actore = new AMaxTrace<double>(projector->dimension());
  Trace<double>* actoreML = new MaxLengthTrace<double>(actore, 1000);
  ActorOffPolicy<double, float>* actor =
      new ActorLambdaOffPolicy<double, float>(alpha_u, gamma, 0.4, target,
          actoreML);

  /*Policy<double>* behavior = new RandomPolicy<double>(
   &problem->getActionList());*/
  Policy<double>* behavior = new BoltzmannDistribution<double>(
      projector->dimension(), &problem->getDiscreteActionList());
  OffPolicyControlLearner<double, float>* control = new OffPAC<double, float>(
      behavior, critic, actor, toStateAction, projector, gamma);

  Simulator<double, float>* sim = new Simulator<double, float>(control,
      problem);
  sim->run(1, 5000, 200);

  delete problem;
  delete projector;
  delete toStateAction;
  delete critice;
  delete criticeML;
  delete critic;
  delete actore;
  delete actoreML;
  delete actor;
  delete behavior;
  delete target;
  delete control;
  delete sim;
}

void testSimple()
{
  double a = 2.0 / 10;
  double b = 1.0 / 2.0 / 10;
  double c = 1.0 / a;
  cout << b << endl;
  cout << c << endl;
}

void testPersistResurrect()
{
  srand(time(0));
  srand48(time(0));
  SparseVector<float> a(20);
  for (int i = 0; i < 10; i++)
    a.insertEntry(i, drand48());
  cout << a << endl;
  a.persist(string("testsv.dat"));

  SparseVector<float> b(20);
  b.resurrect(string("testsv.dat"));
  cout << b << endl;

  DenseVector<float> d(20);
  for (int i = 0; i < 10; i++)
    d[i] = drand48();
  cout << d << endl;
  d.persist(string("testdv.dat"));

  DenseVector<float> e(20);
  e.resurrect(string("testdv.dat"));
  cout << e << endl;
}

int main(int argc, char** argv)
{
  cout << "## start" << endl; // prints @@ start
//  testSparseVector();
//  testProjector();
//  testProjectorMachineLearning();
//  testSarsaMountainCar();
//  testExpectedSarsaMountainCar();
//  testGreedyGQMountainCar();
  testOffPACMountainCar();
//  testGreedyGQContinuousGridworld();
//  testOffPACContinuousGridworld();
//  testOffPACMountainCar2();

//  testGreedyGQMountainCar2();
//  testSarsaMountainCar3D();
//  testOffPACMountainCar3D();
//  testOffPACSwingPendulum();
//  testOffPACSwingPendulum2();

//  testOnPolicySwingPendulum();
//  testOnPolicyCar();
//  testSimple();
//  testPersistResurrect();
  cout << endl;
  cout << "## end" << endl;
  return 0;
}
