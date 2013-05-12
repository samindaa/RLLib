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
#include <string>
#include <map>
#include <fstream>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <algorithm>

// From the RLLib
#include "Vector.h"
#include "Trace.h"
#include "Projector.h"
#include "ControlAlgorithm.h"
#include "Representation.h"
// Eigen
//#include "../Eigen/Dense"

// Simulation
#include "Simulator.h"
#include "MCar2D.h"
#include "MCar3D.h"
#include "SwingPendulum.h"
#include "ContinuousGridworld.h"
#include "Acrobot.h"
#include "RandlovBike.h"
#include "PoleBalancing.h"
#include "TorquedPendulum.h"

#include "util/Spline.h"

using namespace std;
using namespace RLLib;

void testProjector()
{
  srand(time(0));

  int numObservations = 2;
  int memorySize = 512;
  int numTiling = 32;
  SparseVector<double> w(memorySize);
  for (int t = 0; t < 50; t++)
    w.insertEntry(rand() % memorySize, Random::nextDouble());
  TileCoderHashing<double, float> coder(memorySize, numTiling, true);
  DenseVector<float> x(numObservations);
  for (int p = 0; p < 5; p++)
  {
    for (int o = 0; o < numObservations; o++)
      x[o] = Random::nextDouble() / 0.25;
    const SparseVector<double>& vect = coder.project(x);
    cout << w << endl;
    cout << vect << endl;
    cout << w.dot(vect) << endl;
    cout << "---------" << endl;
  }
}

void testSarsaTabularActionMountainCar()
{
  srand(time(0));
  cout << "time=" << time(0) << endl;
  Env<float>* problem = new MCar2D;
  Projector<double, float>* projector = new TileCoderNoHashing<double, float>(
      1000, 10, true);
  StateToStateAction<double, float>* toStateAction = new TabularAction<double,
      float>(projector, &problem->getDiscreteActionList(), true);
  Trace<double>* e = new RTrace<double>(toStateAction->dimension());

  cout << "|phi_sa|=" << toStateAction->dimension() << endl;
  cout << "||.||=" << toStateAction->vectorNorm() << endl;

  double alpha = 0.15 / toStateAction->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.3;
  Sarsa<double>* sarsa = new Sarsa<double>(alpha, gamma, lambda, e);
  double epsilon = 0.01;
  Policy<double>* acting = new EpsilonGreedy<double>(sarsa,
      &problem->getDiscreteActionList(), epsilon);
  OnPolicyControlLearner<double, float>* control = new SarsaControl<double,
      float>(acting, toStateAction, sarsa);

  Simulator<double, float>* sim = new Simulator<double, float>(control,
      problem);
  sim->run(1, 5000, 300);
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

void testOnPolicyBoltzmannRTraceTabularActionCar()
{
  srand(time(0));
  Env<float>* problem = new MCar2D;

  Projector<double, float>* projector = new TileCoderHashing<double, float>(
      1000, 10, false);
  StateToStateAction<double, float>* toStateAction = new TabularAction<double,
      float>(projector, &problem->getDiscreteActionList(), false);

  cout << "|x_t|=" << projector->dimension() << endl;
  cout << "||.||=" << projector->vectorNorm() << endl;
  cout << "|phi_sa|=" << toStateAction->dimension() << endl;
  cout << "||.||=" << toStateAction->vectorNorm() << endl;

  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_u = 0.01 / projector->vectorNorm();
  double lambda = 0.3;
  double gamma = 0.99;

  Trace<double>* critice = new RTrace<double>(projector->dimension());
  TDLambda<double>* critic = new TDLambda<double>(alpha_v, gamma, lambda,
      critice);

  PolicyDistribution<double>* acting = new BoltzmannDistribution<double>(
      toStateAction->dimension(), &problem->getDiscreteActionList());

  Trace<double>* actore = new RTrace<double>(toStateAction->dimension());
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore);
  ActorOnPolicy<double, float>* actor = new ActorLambda<double, float>(alpha_u,
      gamma, lambda, acting, actoreTraces);

  OnPolicyControlLearner<double, float>* control =
      new ActorCritic<double, float>(critic, actor, projector, toStateAction);

  Simulator<double, float>* sim = new Simulator<double, float>(control,
      problem);
  sim->run(1, 5000, 300);
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
  delete sim;
}

void testSarsaMountainCar()
{
  srand(time(0));
  Env<float>* problem = new MCar2D;
  Projector<double, float>* projector = new TileCoderHashing<double, float>(
      10000, 10, true);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<
      double, float>(projector, &problem->getDiscreteActionList());
  Trace<double>* e = new RTrace<double>(projector->dimension());
  double alpha = 0.15 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.3;
  Sarsa<double>* sarsa = new Sarsa<double>(alpha, gamma, lambda, e);
  double epsilon = 0.01;
  Policy<double>* acting = new EpsilonGreedy<double>(sarsa,
      &problem->getDiscreteActionList(), epsilon);
  OnPolicyControlLearner<double, float>* control = new SarsaControl<double,
      float>(acting, toStateAction, sarsa);

  Simulator<double, float>* sim = new Simulator<double, float>(control,
      problem);
  sim->run(1, 5000, 300);
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
  Projector<double, float>* projector = new TileCoderHashing<double, float>(
      10000, 10, true);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<
      double, float>(projector, &problem->getDiscreteActionList());
  Trace<double>* e = new RTrace<double>(projector->dimension());
  double alpha = 0.2 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.1;
  Sarsa<double>* sarsa = new Sarsa<double>(alpha, gamma, lambda, e);
  double epsilon = 0.01;
  Policy<double>* acting = new EpsilonGreedy<double>(sarsa,
      &problem->getDiscreteActionList(), epsilon);
  OnPolicyControlLearner<double, float>* control = new ExpectedSarsaControl<
      double, float>(acting, toStateAction, sarsa,
      &problem->getDiscreteActionList());

  Simulator<double, float>* sim = new Simulator<double, float>(control,
      problem);
  sim->run(5, 5000, 300);
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

void testGreedyGQOnPolicyMountainCar()
{
  srand(time(0));
  Env<float>* problem = new MCar2D;
  Projector<double, float>* projector = new TileCoderHashing<double, float>(
      10000, 10, true);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<
      double, float>(projector, &problem->getDiscreteActionList());
  Trace<double>* e = new ATrace<double>(projector->dimension());
  double alpha_v = 0.05 / projector->vectorNorm();
  double alpha_w = 0.0 / projector->vectorNorm();
  double gamma_tp1 = 0.9;
  double beta_tp1 = 1.0 - gamma_tp1;
  double lambda_t = 0.1;
  GQ<double>* gq = new GQ<double>(alpha_v, alpha_w, beta_tp1, lambda_t, e);
  //double epsilon = 0.01;
  Policy<double>* acting = new EpsilonGreedy<double>(gq,
      &problem->getDiscreteActionList(), 0.01);

  OffPolicyControlLearner<double, float>* control = new GQOnPolicyControl<
      double, float>(acting, &problem->getDiscreteActionList(), toStateAction,
      gq);

  Simulator<double, float>* sim = new Simulator<double, float>(control,
      problem);
  sim->run(1, 5000, 300);
  sim->computeValueFunction();

  delete problem;
  delete projector;
  delete toStateAction;
  delete e;
  delete gq;
  delete acting;
  delete control;
  delete sim;
}

void testGreedyGQMountainCar()
{
  srand(time(0));
  Env<float>* problem = new MCar2D;
  Projector<double, float>* projector = new TileCoderHashing<double, float>(
      1000000, 10, true);
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
  Env<float>* problem = new MCar2D;
  Projector<double, float>* projector = new TileCoderHashing<double, float>(
      1000000, 10, true);
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
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore);
  ActorOffPolicy<double, float>* actor =
      new ActorLambdaOffPolicy<double, float>(alpha_u, gamma, lambda, target,
          actoreTraces);

  Policy<double>* behavior = new RandomPolicy<double>(
      &problem->getDiscreteActionList());

  OffPolicyControlLearner<double, float>* control = new OffPAC<double, float>(
      behavior, critic, actor, toStateAction, projector, gamma);

  Simulator<double, float>* sim = new Simulator<double, float>(control,
      problem);
  sim->run(10, 5000, 100);
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
  delete actoreTraces;
  delete actor;
  delete behavior;
  delete target;
  delete control;
  delete sim;
}

void testGreedyGQContinuousGridworld()
{
  srand(time(0));
  Env<float>* problem = new ContinuousGridworld;
  Projector<double, float>* projector = new TileCoderHashing<double, float>(
      1000000, 10, false);
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
  Env<float>* problem = new ContinuousGridworld;
  Projector<double, float>* projector = new TileCoderHashing<double, float>(
      1000000, 10, true);
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
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore);
  ActorOffPolicy<double, float>* actor =
      new ActorLambdaOffPolicy<double, float>(alpha_u, gamma, lambda, target,
          actoreTraces);

  Policy<double>* behavior = new RandomPolicy<double>(
      &problem->getDiscreteActionList());
  OffPolicyControlLearner<double, float>* control = new OffPAC<double, float>(
      behavior, critic, actor, toStateAction, projector, gamma);

  Simulator<double, float>* sim = new Simulator<double, float>(control,
      problem);
  sim->run(5, 5000, 3000);
  sim->computeValueFunction();

  control->persist("visualization/cgw_offpac.data");

  control->reset();
  control->resurrect("visualization/cgw_offpac.data");
  sim->test(100, 2000);

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

/*
 void testOffPACOnPolicyContinuousGridworld()
 {
 srand(time(0));
 Env<float>* problem = new ContinuousGridworld;
 Projector<double, float>* projector = new FullTilings<double, float>(1000000,
 10, true);
 StateToStateAction<double, float>* toStateAction = new StateActionTilings<
 double, float>(projector, &problem->getDiscreteActionList());

 double alpha_v = 0.1 / projector->vectorNorm();
 double alpha_w = 0.0001 / projector->vectorNorm();
 double gamma = 0.99;
 double lambda = 0.3;
 Trace<double>* critice = new ATrace<double>(projector->dimension());
 GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma,
 lambda, critice);
 double alpha_u = 0.001 / projector->vectorNorm();
 PolicyDistribution<double>* target = new BoltzmannDistribution<double>(
 projector->dimension(), &problem->getDiscreteActionList());

 Trace<double>* actore = new ATrace<double>(projector->dimension());
 MultiTrace<double>* actoreTraces = new MultiTrace<double>();
 actoreTraces->push_back(actore);
 ActorOffPolicy<double, float>* actor =
 new ActorLambdaOffPolicy<double, float>(alpha_u, gamma, lambda, target,
 actoreTraces);

 //Policy<double>* behavior = new RandomPolicy<double>(
 //    &problem->getDiscreteActionList());
 Policy<double>* behavior = new EpsilonGreedy<double>(critic,
 &problem->getDiscreteActionList(), 0.01);
 OffPolicyControlLearner<double, float>* control = new OffPAC<double, float>(
 behavior, critic, actor, toStateAction, projector, gamma);

 Simulator<double, float>* sim = new Simulator<double, float>(control,
 problem);
 sim->run(1, 5000, 3000);
 sim->computeValueFunction();

 control->persist("visualization/cgw_offpac.data");

 control->reset();
 control->resurrect("visualization/cgw_offpac.data");
 sim->test(100, 2000);

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
 */

void testOffPACContinuousGridworldOPtimized()
{
  srand(time(0));
  Env<float>* problem = new ContinuousGridworld;
  Projector<double, float>* projector = new TileCoderHashing<double, float>(
      1000000, 10, true);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<
      double, float>(projector, &problem->getDiscreteActionList());

  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_w = 0.0001 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.4;
  Trace<double>* critice = new ATrace<double>(projector->dimension());
  Trace<double>* criticeML = new MaxLengthTrace<double>(critice, 1000);
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma,
      lambda, criticeML);
  double alpha_u = 0.001 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(
      projector->dimension(), &problem->getDiscreteActionList());

  Trace<double>* actore = new ATrace<double>(projector->dimension());
  Trace<double>* actoreML = new MaxLengthTrace<double>(actore, 1000);
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actoreML);
  ActorOffPolicy<double, float>* actor =
      new ActorLambdaOffPolicy<double, float>(alpha_u, gamma, lambda, target,
          actoreTraces);

  Policy<double>* behavior = new RandomPolicy<double>(
      &problem->getDiscreteActionList());
  OffPolicyControlLearner<double, float>* control = new OffPAC<double, float>(
      behavior, critic, actor, toStateAction, projector, gamma);

  Simulator<double, float>* sim = new Simulator<double, float>(control,
      problem);
  sim->run(1, 5000, 5000);
  sim->computeValueFunction();

  control->persist("visualization/cgw_offpac.data");

  control->reset();
  control->resurrect("visualization/cgw_offpac.data");
  sim->test(100, 2000);

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

// ====================== Advanced projector ===================================
template<class T, class O>
class AdvancedTilesProjector: public Projector<T, O>
{
  protected:
    SparseVector<double>* vector;
    int* activeTiles;
    Tiles* tiles;

  public:
    AdvancedTilesProjector() :
        vector(new SparseVector<T>(100000 + 1)), activeTiles(new int[48]), tiles(
            new Tiles)
    {
      // Consistent hashing
      int dummy_tiles[1];
      float dummy_vars[1];
      srand(0);
      tiles->tiles(dummy_tiles, 1, 1, dummy_vars, 0); // initializes tiling code
      srand(time(0));
    }

    virtual ~AdvancedTilesProjector()
    {
      delete vector;
      delete[] activeTiles;
      delete tiles;
    }

  public:
    const SparseVector<T>& project(const DenseVector<O>& x, int h1)
    {
      vector->clear();
      // all 4
      tiles->tiles(&activeTiles[0], 12, vector->dimension() - 1, x(),
          x.dimension(), h1);
      // 3 of 4
      static DenseVector<O> x3(3);
      static int x3o[4][3] =
      {
      { 0, 1, 2 },
      { 1, 2, 3 },
      { 2, 3, 0 },
      { 1, 3, 0 } };
      for (int i = 0; i < 4; i++)
      {
        for (int j = 0; j < 3; j++)
          x3[j] = x[x3o[i][j]];
        tiles->tiles(&activeTiles[12 + i * 3], 3, vector->dimension() - 1, x3(),
            x3.dimension(), h1);
      }
      // 2 of 6
      static DenseVector<O> x2(2);
      static int x2o[6][2] =
      {
      { 0, 1 },
      { 1, 2 },
      { 2, 3 },
      { 0, 3 },
      { 0, 2 },
      { 1, 3 } };
      for (int i = 0; i < 6; i++)
      {
        for (int j = 0; j < 2; j++)
          x2[j] = x[x2o[i][j]];
        tiles->tiles(&activeTiles[24 + i * 2], 2, vector->dimension() - 1, x2(),
            x2.dimension(), h1);
      }

      // 3 of 4 of 1
      static DenseVector<O> x1(1);
      static int x1o[4] =
      { 0, 1, 2, 3 };
      for (int i = 0; i < 4; i++)
      {
        x1[0] = x[x1o[i]];
        tiles->tiles(&activeTiles[36 + i * 3], 3, vector->dimension() - 1, x1(),
            x1.dimension(), h1);
      }

      // bias
      vector->insertLast(1.0);
      for (int* i = activeTiles; i < activeTiles + 48; ++i)
        vector->insertEntry(*i, 1.0);

      return *vector;
    }
    const SparseVector<T>& project(const DenseVector<O>& x)
    {

      vector->clear();
      // all 4
      tiles->tiles(&activeTiles[0], 12, vector->dimension() - 1, x(),
          x.dimension());
      // 3 of 4
      static DenseVector<O> x3(3);
      static int x3o[4][3] =
      {
      { 0, 1, 2 },
      { 1, 2, 3 },
      { 2, 3, 0 },
      { 1, 3, 0 } };
      for (int i = 0; i < 4; i++)
      {
        for (int j = 0; j < 3; j++)
          x3[j] = x[x3o[i][j]];
        tiles->tiles(&activeTiles[12 + i * 3], 3, vector->dimension() - 1, x3(),
            x3.dimension());
      }
      // 2 of 6
      static DenseVector<O> x2(2);
      static int x2o[6][2] =
      {
      { 0, 1 },
      { 1, 2 },
      { 2, 3 },
      { 0, 3 },
      { 0, 2 },
      { 1, 3 } };
      for (int i = 0; i < 6; i++)
      {
        for (int j = 0; j < 2; j++)
          x2[j] = x[x2o[i][j]];
        tiles->tiles(&activeTiles[24 + i * 2], 2, vector->dimension() - 1, x2(),
            x2.dimension());
      }

      // 4 of 1
      static DenseVector<O> x1(1);
      static int x1o[4] =
      { 0, 1, 2, 3 };
      for (int i = 0; i < 4; i++)
      {
        x1[0] = x[x1o[i]];
        tiles->tiles(&activeTiles[36 + i], 3, vector->dimension() - 1, x1(),
            x1.dimension());
      }

      for (int* i = activeTiles; i < activeTiles + 48; ++i)
        vector->insertEntry(*i, 1.0);

      // bias
      vector->insertLast(1.0);

      return *vector;
    }

    double vectorNorm() const
    {
      return 48 + 1;
    }
    int dimension() const
    {
      return vector->dimension();
    }
};

// ====================== Mountain Car 3D =====================================
// Mountain Car 3D projector
template<class T, class O>
class MountainCar3DTilesProjector: public AdvancedTilesProjector<T, O>
{
  public:
};

void testOffPACMountainCar3D_1()
{
  srand(time(0));
  Env<float>* problem = new MCar2D;
  Projector<double, float>* projector = new TileCoderHashing<double, float>(
      1000000, 10, true);
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
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actoreML);
  ActorOffPolicy<double, float>* actor =
      new ActorLambdaOffPolicy<double, float>(alpha_u, gamma, 0.4, target,
          actoreTraces);

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
  delete actoreTraces;
  delete actor;
  delete behavior;
  delete target;
  delete control;
  delete sim;
}

void testGreedyGQMountainCar3D()
{
  srand(time(0));
  Env<float>* problem = new MCar3D;
  /*Projector<double, float>* projector = new FullTilings<double, float>(1000000,
   10, true);*/
  Projector<double, float>* projector = new MountainCar3DTilesProjector<double,
      float>();
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<
      double, float>(projector, &problem->getDiscreteActionList());
  Trace<double>* e = new ATrace<double>(projector->dimension(), 0.001);
  Trace<double>* eML = new MaxLengthTrace<double>(e, 2000);
  double alpha_v = 0.2 / projector->vectorNorm();
  double alpha_w = .0001 / projector->vectorNorm();
  double gamma_tp1 = 0.99;
  double beta_tp1 = 1.0 - gamma_tp1;
  double lambda_t = 0.8;
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
void testSarsaMountainCar3D()
{
  srand(time(0));
  Env<float>* problem = new MCar3D;
  /*Projector<double, float>* projector = new FullTilings<double, float>(1000000,
   10, false);*/
  Projector<double, float>* projector = new MountainCar3DTilesProjector<double,
      float>();
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<
      double, float>(projector, &problem->getDiscreteActionList());

  Trace<double>* e = new RTrace<double>(projector->dimension(), 0.001);
  Trace<double>* eML = new MaxLengthTrace<double>(e, 1000);
  double alpha = 0.01 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.6;
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

void testOffPACMountainCar3D_2()
{
  srand(time(0));
  Env<float>* problem = new MCar3D;
  Projector<double, float>* projector = new TileCoderHashing<double, float>(
      1000000, 10, true);
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
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actoreML);
  ActorOffPolicy<double, float>* actor =
      new ActorLambdaOffPolicy<double, float>(alpha_u, gamma, 0.1, target,
          actoreTraces);

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
  delete actoreTraces;
  delete actor;
  delete behavior;
  delete target;
  delete control;
  delete sim;
}

// ====================== Mountain Car 3D =====================================

void testOffPACSwingPendulum()
{
  srand(time(0));
  Env<float>* problem = new SwingPendulum;
  Projector<double, float>* projector = new TileCoderHashing<double, float>(
      1000000, 10, true);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<
      double, float>(projector, &problem->getDiscreteActionList());

  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_w = .0001 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.4;

  Trace<double>* critice = new ATrace<double>(projector->dimension());
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma,
      lambda, critice);
  double alpha_u = 0.5 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(
      projector->dimension(), &problem->getDiscreteActionList());

  Trace<double>* actore = new ATrace<double>(projector->dimension());
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore);
  ActorOffPolicy<double, float>* actor =
      new ActorLambdaOffPolicy<double, float>(alpha_u, gamma, lambda, target,
          actoreTraces);

  Policy<double>* behavior = new RandomPolicy<double>(
      &problem->getDiscreteActionList());
  /*Policy<double>* behavior = new RandomBiasPolicy<double>(
   &problem->getDiscreteActionList());*/
  OffPolicyControlLearner<double, float>* control = new OffPAC<double, float>(
      behavior, critic, actor, toStateAction, projector, gamma);

  Simulator<double, float>* sim = new Simulator<double, float>(control,
      problem);
  sim->run(1, 5000, 200);
  sim->computeValueFunction();

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

void testOnPolicyContinousActionCar(const int& nbMemory, const double& lambda,
    const double& gamma, double alpha_v, double alpha_u)
{
  srand(time(0));
  Env<float>* problem = new MCar2D;
  Projector<double, float>* projector = new TileCoderHashing<double, float>(
      nbMemory, 10, false);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<
      double, float>(projector, &problem->getContinuousActionList());

  alpha_v /= projector->vectorNorm();
  alpha_u /= projector->vectorNorm();

  Trace<double>* critice = new RTrace<double>(projector->dimension());
  TDLambda<double>* critic = new TDLambda<double>(alpha_v, gamma, lambda,
      critice);

  PolicyDistribution<double>* acting = new NormalDistributionScaled<double>(0,
      1.0, projector->dimension(), &problem->getContinuousActionList());

  Trace<double>* actore1 = new RTrace<double>(projector->dimension());
  Trace<double>* actore2 = new RTrace<double>(projector->dimension());
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore1);
  actoreTraces->push_back(actore2);
  ActorOnPolicy<double, float>* actor = new ActorLambda<double, float>(alpha_u,
      gamma, lambda, acting, actoreTraces);

  OnPolicyControlLearner<double, float>* control = new AverageRewardActorCritic<
      double, float>(critic, actor, projector, toStateAction, 0);

  Simulator<double, float>* sim = new Simulator<double, float>(control,
      problem);
  sim->run(1, 5000, 200);
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
  delete sim;
}

void testOnPolicyBoltzmannATraceCar()
{
  srand(time(0));
  Env<float>* problem = new MCar2D;

  Projector<double, float>* projector = new TileCoderHashing<double, float>(
      10000, 10, false);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<
      double, float>(projector, &problem->getDiscreteActionList());

  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_u = 0.01 / projector->vectorNorm();
  double lambda = 0.3;
  double gamma = 0.99;

  Trace<double>* critice = new ATrace<double>(projector->dimension());
  TDLambda<double>* critic = new TDLambda<double>(alpha_v, gamma, lambda,
      critice);

  PolicyDistribution<double>* acting = new BoltzmannDistribution<double>(
      projector->dimension(), &problem->getDiscreteActionList());

  Trace<double>* actore = new ATrace<double>(projector->dimension());
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore);
  ActorOnPolicy<double, float>* actor = new ActorLambda<double, float>(alpha_u,
      gamma, lambda, acting, actoreTraces);

  OnPolicyControlLearner<double, float>* control =
      new ActorCritic<double, float>(critic, actor, projector, toStateAction);

  Simulator<double, float>* sim = new Simulator<double, float>(control,
      problem);
  sim->run(1, 5000, 300);
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
  delete sim;
}

void testOnPolicyBoltzmannRTraceCar()
{
  srand(time(0));
  Env<float>* problem = new MCar2D;

  Projector<double, float>* projector = new TileCoderHashing<double, float>(
      10000, 10, false);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<
      double, float>(projector, &problem->getDiscreteActionList());

  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_u = 0.01 / projector->vectorNorm();
  double lambda = 0.3;
  double gamma = 0.99;

  Trace<double>* critice = new RTrace<double>(projector->dimension());
  TDLambda<double>* critic = new TDLambda<double>(alpha_v, gamma, lambda,
      critice);

  PolicyDistribution<double>* acting = new BoltzmannDistribution<double>(
      projector->dimension(), &problem->getDiscreteActionList());

  Trace<double>* actore = new RTrace<double>(projector->dimension());
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore);
  ActorOnPolicy<double, float>* actor = new ActorLambda<double, float>(alpha_u,
      gamma, lambda, acting, actoreTraces);

  OnPolicyControlLearner<double, float>* control =
      new ActorCritic<double, float>(critic, actor, projector, toStateAction);

  Simulator<double, float>* sim = new Simulator<double, float>(control,
      problem);
  sim->run(1, 5000, 300);
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
  delete sim;
}

void testOnPolicyContinousActionCar()
{
  testOnPolicyContinousActionCar(10000, 0.4, 0.99, 0.1, 0.001);
}

void testOnPolicySwingPendulum()
{
  srand(time(0));
  Env<float>* problem = new SwingPendulum;
  Projector<double, float>* projector = new TileCoderHashing<double, float>(
      1000, 10, false);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<
      double, float>(projector, &problem->getContinuousActionList());

  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_u = 0.001 / projector->vectorNorm();
  double alpha_r = .0001;
  double gamma = 1.0;
  double lambda = 0.5;

  Trace<double>* critice = new ATrace<double>(projector->dimension());
  TDLambda<double>* critic = new TDLambda<double>(alpha_v, gamma, lambda,
      critice);

  PolicyDistribution<double>* policyDistribution = new NormalDistributionScaled<
      double>(0, 1.0, projector->dimension(),
      &problem->getContinuousActionList());
  Range<double> policyRange(-2.0, 2.0);
  Range<double> problemRange(-2.0, 2.0);
  PolicyDistribution<double>* acting = new ScaledPolicyDistribution<double>(
      &problem->getContinuousActionList(), policyDistribution, &policyRange,
      &problemRange);

  Trace<double>* actore1 = new ATrace<double>(projector->dimension());
  Trace<double>* actore2 = new ATrace<double>(projector->dimension());
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore1);
  actoreTraces->push_back(actore2);
  ActorOnPolicy<double, float>* actor = new ActorLambda<double, float>(alpha_u,
      gamma, lambda, acting, actoreTraces);

  OnPolicyControlLearner<double, float>* control = new AverageRewardActorCritic<
      double, float>(critic, actor, projector, toStateAction, alpha_r);

  Simulator<double, float>* sim = new Simulator<double, float>(control,
      problem);
  sim->run(1, 5000, 100, false);
  sim->test(1, 1000);
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
  delete policyDistribution;
  delete acting;
  delete control;
  delete sim;
}

void testOffPACSwingPendulum2()
{
  srand(time(0));
  Env<float>* problem = new SwingPendulum;
  Projector<double, float>* projector = new TileCoderHashing<double, float>(
      1000000, 10, true);
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
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actoreML);
  ActorOffPolicy<double, float>* actor =
      new ActorLambdaOffPolicy<double, float>(alpha_u, gamma, 0.4, target,
          actoreTraces);

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
  delete actoreTraces;
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

// ====================== Acrobot projector ===================================

template<class T, class O>
class AcrobotTilesProjector: public AdvancedTilesProjector<T, O>
{
  public:
};

void testOffPACAcrobot()
{
  srand(time(0));
  Env<float>* problem = new Acrobot;
  Projector<double, float>* projector =
      new AcrobotTilesProjector<double, float>();
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
  double alpha_u = 0.0001 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(
      projector->dimension(), &problem->getDiscreteActionList());

  Trace<double>* actore = new AMaxTrace<double>(projector->dimension());
  Trace<double>* actoreML = new MaxLengthTrace<double>(actore, 1000);
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actoreML);
  ActorOffPolicy<double, float>* actor =
      new ActorLambdaOffPolicy<double, float>(alpha_u, gamma, lambda, target,
          actoreTraces);

  /*Policy<double>* behavior = new RandomPolicy<double>(
   &problem->getDiscreteActionList());*/
  Policy<double>* behavior = new RandomBiasPolicy<double>(
      &problem->getDiscreteActionList());
  OffPolicyControlLearner<double, float>* control = new OffPAC<double, float>(
      behavior, critic, actor, toStateAction, projector, gamma);

  Simulator<double, float>* sim = new Simulator<double, float>(control,
      problem);
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

void testGreedyGQAcrobot()
{
  srand(time(0));
  Env<float>* problem = new Acrobot;
  /*Projector<double, float>* projector = new FullTilings<double, float>(1000000,
   10, true);*/
  Projector<double, float>* projector =
      new AcrobotTilesProjector<double, float>();
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<
      double, float>(projector, &problem->getDiscreteActionList());
  Trace<double>* e = new ATrace<double>(projector->dimension(), 0.001);
  Trace<double>* eML = new MaxLengthTrace<double>(e, 1000);
  double alpha_v = 0.2 / projector->vectorNorm();
  double alpha_w = .001 / projector->vectorNorm();
  double gamma_tp1 = 0.99;
  double beta_tp1 = 1.0 - gamma_tp1;
  double lambda_t = 0.8;
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

void testPoleBalancingPlant()
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

void testPersistResurrect()
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

void testExp()
{
  //cout << exp(200) << endl;
  cout << std::numeric_limits<float>::epsilon() << endl;
  cout << 1e3 << endl;
}

void testEigen3()
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

void testTorquedPendulum()
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

void testCubicSpline()
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

void testMotion()
{
  Spline spline;
  std::ifstream inf(
      "/home/sam/projects/workspace_robocanes/RLLib/visualization/j_balance_1");

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
    for (std::vector<std::string>::iterator iter = results.begin();
        iter != results.end(); ++iter)
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
  for (map<int, vector<double> >::const_iterator iter = motions.begin();
      iter != motions.end(); ++iter)
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
  for (map<int, vector<double> >::const_iterator iter = motions.begin();
      iter != motions.end(); ++iter)
  {
    cTime += iter->second[0];
    of2 << cTime << " " << iter->second[3] << " " << spline(cTime) << endl;
  }
  of.close();
  of2.close();

}

int main(int argc, char** argv)
{
  cout << "## start" << endl; // prints @@ start
//  testSparseVector();
//  testProjector();
//  testProjectorMachineLearning();
//  testSarsaMountainCar();
//  testSarsaTabularActionMountainCar();
//  testOnPolicyBoltzmannRTraceTabularActionCar();
//  testExpectedSarsaMountainCar();
//  testGreedyGQOnPolicyMountainCar();
//  testGreedyGQMountainCar();
//  testOffPACMountainCar();
//  testGreedyGQContinuousGridworld();
  testOffPACContinuousGridworld();
//  testOffPACContinuousGridworldOPtimized();
//  testOffPACMountainCar3D_1();
//  testOffPACOnPolicyContinuousGridworld();

//  testGreedyGQMountainCar3D();
//  testSarsaMountainCar3D();
//  testOffPACMountainCar3D_2();
//  testOffPACSwingPendulum();
//  testOffPACSwingPendulum2();
//  testOffPACAcrobot();
//  testGreedyGQAcrobot();

//  testOnPolicySwingPendulum();
//  testOnPolicyContinousActionCar();
//  testOnPolicyBoltzmannATraceCar();
//  testOnPolicyBoltzmannRTraceCar();

//  testPoleBalancingPlant();
//  testTorquedPendulum();

// some simple stuff
//  testSimple();
//  testPersistResurrect();
//  testExp();
//  testEigen3();
//  testCubicSpline();
//  testMotion();
  cout << endl;
  cout << "## end" << endl;
  return 0;
}
