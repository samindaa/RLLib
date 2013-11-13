/*
 * OnOffPolicyPredictionTest.h
 *
 *  Created on: Oct 26, 2013
 *      Author: sam
 */

#ifndef ONOFFPOLICYPREDICTIONTEST_H_
#define ONOFFPOLICYPREDICTIONTEST_H_

#include "HeaderTest.h"

RLLIB_TEST(OnOffPolicyPredictionTest)

class OnOffPolicyPredictionTest: public OnOffPolicyPredictionTestBase
{
  protected:
    void testTD(FSGAgentState* agentState, FiniteStateGraph* graph, OnPolicyTD<double>* td,
        const int& nbEpisodeMax, const double& precision);

    void testOffPolicyGTD(FSGAgentState* agentState, RandomWalk* problem, OffPolicyTD<double>* gtd,
        const int& nbEpisodeMax, const double& precision, const double& targetLeftProbability,
        const double& behaviourLeftProbability);

    void testOnLineProblem();
    void testOnRandomWalkProblem();

    void testOffPolicy();
    void testOffPolicyWithLambda();

  public:
    void run();
};

#endif /* ONOFFPOLICYPREDICTIONTEST_H_ */
