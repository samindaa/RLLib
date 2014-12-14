/*
 * CartPoleBalancingTest.cpp
 *
 *  Created on: Nov 22, 2013
 *      Author: sam
 */

#include "CartPoleBalancingTest.h"

void CartPoleBalancingTest::run()
{
  testCartPole();
  testNonMarkovPoleBalancing();
  testNonMarkovPoleBalancingCMAES();
  testCartPoleCMAES();
}

void CartPoleBalancingTest::testCartPole()
{
  Random<double> random;
  CartPole pb(&random);
  pb.initialize();
  Action<double> action(0);
  action.push_back(1.0f);
  int i = 0;
  for (i = 0; i < 1000; i++)
  {
    pb.step(&action);
    const TRStep<double>* step = pb.getTRStep();
    if (step->endOfEpisode)
      break;
  }
  Assert::assertPasses(i < 200);
}

void CartPoleBalancingTest::testNonMarkovPoleBalancing()
{
  Random<double> random;
  NonMarkovPoleBalancing<double> pb(&random);
  pb.initialize();
  Action<double> action(0);
  action.push_back(0.0f);
  int i = 0;
  for (i = 0; i < 1000; i++)
  {
    pb.step(&action);
    const TRStep<double>* step = pb.getTRStep();
    if (step->endOfEpisode)
      break;
  }
  Assert::assertPasses(i < 200);
}

void CartPoleBalancingTest::testNonMarkovPoleBalancingCMAES()
{
  Random<double>* random = new Random<double>;
  RLProblem<double>* problem = new NonMarkovPoleBalancing<double>(random, 2);
  RLAgent<double>* agent = new CMAESAgent(problem, false);
  RLRunner<double>* sim = new RLRunner<double>(agent, problem, 5000, 5000, 1);
  sim->setVerbose(false);
  sim->run();

  delete random;
  delete problem;
  delete agent;
  delete sim;
}

void CartPoleBalancingTest::testCartPoleCMAES()
{
  Random<double>* random = new Random<double>;
  RLProblem<double>* problem = new CartPole(random);
  RLAgent<double>* agent = new CMAESAgent(problem, false);
  RLRunner<double>* sim = new RLRunner<double>(agent, problem, 5000, 200, 1);
  sim->setVerbose(false);
  sim->run();

  delete problem;
  delete agent;
  delete sim;
}

// ==
CMAESAgent::CMAESAgent(RLProblem<double>* problem, const bool& evaluation) :
    RLAgent<double>(0), problem(problem), evo(new cmaes_t), x(
        new PVector<double>(problem->dimension())), phi(new PVector<double>(problem->dimension())), xstart(
        new PVector<double>(problem->dimension())), stddev(
        new PVector<double>(problem->dimension())), initialized(false), evaluation(evaluation), nbEvaluations(
        0), nbPopEvaluations(0), lambda(0), arFunvals(0), pop(0)
{
}

CMAESAgent::~CMAESAgent()
{
  cmaes_exit(evo);
  delete evo;
  delete x;
  delete phi;
  delete xstart;
  delete stddev;
}

const Action<double>* CMAESAgent::initialize(const TRStep<double>* step)
{
  if (!evaluation)
  {
    if (!initialized)
    {
      initialized = true;
      xstart->set(0.0f);
      stddev->set(0.01f); // FixMe:
      lambda = 30; // FixMe:

      arFunvals = cmaes_init(evo, x->dimension(), xstart->getValues(), stddev->getValues(), 0,
          lambda, "non");
      pop = cmaes_SamplePopulation(evo);
      nbPopEvaluations = -1;
    }

    ++nbPopEvaluations;
    if (nbPopEvaluations == lambda)
    {
      cmaes_UpdateDistribution(evo, arFunvals);
      const double* xfinal = cmaes_GetPtr(evo, "xmean");
      // Persist
      PVector<double> xf(x->dimension());
      for (int i = 0; i < x->dimension(); i++)
        xf.setEntry(i, xfinal[i]);
      xf.persist("xmean.dat");

      pop = cmaes_SamplePopulation(evo);
      nbPopEvaluations = 0;
    }

    ASSERT(nbPopEvaluations < lambda);
    double const *nextX = pop[nbPopEvaluations];
    for (int j = 0; j < x->dimension(); j++)
      x->setEntry(j, nextX[j]);
    arFunvals[nbPopEvaluations] = 0.0f;
    ++nbEvaluations;
  }
  else
  {
    if (!initialized)
    {
      initialized = true;
      x->resurrect("xmean.dat");
    }
  }

  problem->getContinuousActions()->update(0, 0, x->dot(getPhi(step->o_tp1)));
  return problem->getContinuousActions()->getEntry(0);
}

const Action<double>* CMAESAgent::getAtp1(const TRStep<double>* step)
{
  if (!evaluation)
    arFunvals[nbPopEvaluations] -= fabs(step->r_tp1);
  problem->getContinuousActions()->update(0, 0, x->dot(getPhi(step->o_tp1)));
  return problem->getContinuousActions()->getEntry(0);
}

void CMAESAgent::reset()
{
// Not used:
}

const Vector<double>* CMAESAgent::getPhi(const Vector<double>* x_tp1)
{
  for (int i = 0; i < x_tp1->dimension(); i++)
    phi->setEntry(i, x->getEntry(i));
  return phi;
}

RLLIB_TEST_MAKE(CartPoleBalancingTest)
