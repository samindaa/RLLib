/*
 * CartPoleBalancingTest.h
 *
 *  Created on: Nov 22, 2013
 *      Author: sam
 */

#ifndef CARTPOLEBALANCINGTEST_H_
#define CARTPOLEBALANCINGTEST_H_

#include "Test.h"
#include "PoleBalancing.h"
#include "CartPole.h"
#include "NonMarkovPoleBalancing.h"
#include "util/cma/cmaes_interface.h"

RLLIB_TEST(CartPoleBalancingTest)
class CartPoleBalancingTest: public CartPoleBalancingTestBase
{
  public:
    void run();

  private:
    void testCartPole();
    void testNonMarkovPoleBalancing();
    void testNonMarkovPoleBalancingCMAES();
    void testCartPoleCMAES();
};

class CMAESAgent: public RLAgent<double>
{
  protected:
    RLProblem<double>* problem;
    cmaes_t* evo;
    Vector<double>* x;
    Vector<double>* phi;
    Vector<double>* xstart;
    Vector<double>* stddev;
    bool initialized;
    bool evaluation;

    // CMAES
    int nbEvaluations;
    int nbPopEvaluations;
    int lambda;
    double* arFunvals;
    double* const * pop;

  public:
    CMAESAgent(RLProblem<double>* problem, const bool& evaluation);
    virtual ~CMAESAgent();
    const Action<double>* initialize(const TRStep<double>* step);
    const Action<double>* getAtp1(const TRStep<double>* step);
    void reset();

  private:
    const Vector<double>* getPhi(const Vector<double>* x_tp1);
};

#endif /* CARTPOLEBALANCINGTEST_H_ */
