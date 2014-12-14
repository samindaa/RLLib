/*
 * BicycleTest.cpp
 *
 *  Created on: Dec 2, 2013
 *      Author: sam
 */

#include "BicycleTest.h"

BicycleProjector::BicycleProjector(const int& nbVars)
{
  /**
   * Number of inputs | Tiling type | Number of intervals | Number of tilings
   *            5/7   |     1D      |       8             |       8
   *                  |     1D      |       4             |       4
   *                  |     2D      |       4             |       4
   *                  |     2D + 1  |       4             |       4
   *                  |     2D + 2  |       4             |       4
   */
  nbTiles = nbVars * (8 + 4 + 4 + 4 + 4);
  memory = nbVars * (8 * 8 + 4 * 4 + 4 * 4 * 4 + 4 * 4 * 4 + 4 * 4 * 4) * 9/*nbActions*/
  * 8/*to hash*/;
  vector = new SVector<double>(memory + 1/*bias unit*/, nbTiles + 1/*bias unit*/);
  random = new Random<double>;
  hashing = new MurmurHashing<double>(random, memory);
  tiles = new Tiles<double>(hashing);

  std::cout << "nbTiles=" << nbTiles << " memory=" << vector->dimension() << std::endl;
}

BicycleProjector::~BicycleProjector()
{
  delete vector;
  delete random;
  delete hashing;
  delete tiles;
}

const Vector<double>* BicycleProjector::project(const Vector<double>* x, const int& h1)
{
  vector->clear();
  if (x->empty())
    return vector;

  int h2 = 0;
  // IRdistance
  for (int i = 0; i < x->dimension(); i++)
  {
    tiles->tiles1(vector, 8, memory, x->getEntry(i) * 8, h1, h2++);
    tiles->tiles1(vector, 4, memory, x->getEntry(i) * 4, h1, h2++);

    int j = (i + 1) % x->dimension();
    tiles->tiles2(vector, 4, memory, x->getEntry(i) * 4, x->getEntry(j) * 4, h1, h2++);

    j = (i + 2) % x->dimension();
    tiles->tiles2(vector, 4, memory, x->getEntry(i) * 4, x->getEntry(j) * 4, h1, h2++);

    j = (i + 3) % x->dimension();
    tiles->tiles2(vector, 4, memory, x->getEntry(i) * 4, x->getEntry(j) * 4, h1, h2++);
  }

  vector->setEntry(vector->dimension() - 1, 1.0f);
  return vector;
}

const Vector<double>* BicycleProjector::project(const Vector<double>* x)
{
  vector->clear();
  if (x->empty())
    return vector;

  int h2 = 0;
  for (int i = 0; i < x->dimension(); i++)
  {
    tiles->tiles1(vector, 8, memory, x->getEntry(i) * 8, h2++);
    tiles->tiles1(vector, 4, memory, x->getEntry(i) * 4, h2++);

    int j = (i + 1) % x->dimension();
    tiles->tiles2(vector, 4, memory, x->getEntry(i) * 4, x->getEntry(j) * 4, h2++);

    j = (i + 2) % x->dimension();
    tiles->tiles2(vector, 4, memory, x->getEntry(i) * 4, x->getEntry(j) * 4, h2++);

    j = (i + 3) % x->dimension();
    tiles->tiles2(vector, 4, memory, x->getEntry(i) * 4, x->getEntry(j) * 4, h2++);
  }

  vector->setEntry(vector->dimension() - 1, 1.0f);
  return vector;
}

double BicycleProjector::vectorNorm() const
{
  return nbTiles + 1;
}

int BicycleProjector::dimension() const
{
  return vector->dimension();
}

void BicycleTest::testBicycleBalance()
{
  Random<double>* random = new Random<double>;
  RandlovBike<double>* problem = new RandlovBike<double>(random, false);
  Projector<double>* projector = new BicycleProjector(problem->dimension());
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());
  Trace<double>* e = new ATrace<double>(projector->dimension());
  double alpha = 0.1 / projector->vectorNorm();
  double gamma = problem->getGamma();
  double lambda = 0.96;
  Sarsa<double>* sarsa = new SarsaTrue<double>(alpha, gamma, lambda, e);
  double epsilon = 0.01;
  Policy<double>* acting = new EpsilonGreedy<double>(random, problem->getDiscreteActions(), sarsa,
      epsilon);
  OnPolicyControlLearner<double>* control = new SarsaControl<double>(acting, toStateAction, sarsa);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  RLRunner<double>* sim = new RLRunner<double>(agent, problem, 100000, 130, 1);
  sim->run();
  control->persist("visualization/bicycle_balance.dat");
  control->reset();
  control->resurrect("visualization/bicycle_balance.dat");
  sim->runEvaluate(10);

  delete random;
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

void BicycleTest::testBicycleGoToTarget()
{
  Random<double>* random = new Random<double>;
  RandlovBike<double>* problem = new RandlovBike<double>(random, true);
  Projector<double>* projector = new BicycleProjector(problem->dimension());
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());
  Trace<double>* e = new ATrace<double>(projector->dimension());
  double alpha = 0.1 / projector->vectorNorm();
  double gamma = problem->getGamma();
  double lambda = 0.9;
  Sarsa<double>* sarsa = new SarsaAlphaBound<double>(alpha, gamma, lambda, e);
  double epsilon = 0.01;
  Policy<double>* acting = new EpsilonGreedy<double>(random, problem->getDiscreteActions(), sarsa,
      epsilon);
  OnPolicyControlLearner<double>* control = new SarsaControl<double>(acting, toStateAction, sarsa);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  RLRunner<double>* sim = new RLRunner<double>(agent, problem, 100000, 1000, 1);
  sim->run();
  control->persist("visualization/bicycle_goToTarget.dat");
  control->reset();
  control->resurrect("visualization/bicycle_goToTarget.dat");
  sim->runEvaluate(10);

  delete random;
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

void BicycleTest::testBicycleGoToTargetEvaluate()
{
  Random<double>* random = new Random<double>;
  RandlovBike<double>* problem = new RandlovBike<double>(random, true);
  Projector<double>* projector = new BicycleProjector(problem->dimension());
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());
  Trace<double>* e = new ATrace<double>(projector->dimension());
  Sarsa<double>* sarsa = new SarsaTrue<double>(0, 0, 0, e);
  Policy<double>* acting = new Greedy<double>(problem->getDiscreteActions(), sarsa);
  OnPolicyControlLearner<double>* control = new SarsaControl<double>(acting, toStateAction, sarsa);
  control->reset();
  control->resurrect("visualization/bicycle_goToTarget.dat");

  RLAgent<double>* agent = new ControlAgent<double>(control);
  RLRunner<double>* sim = new RLRunner<double>(agent, problem, 100000, 10, 1);
  sim->run();

  delete random;
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

void BicycleTest::run()
{
  testBicycleBalance();
  testBicycleGoToTarget();
  testBicycleGoToTargetEvaluate();
}

RLLIB_TEST_MAKE(BicycleTest)
