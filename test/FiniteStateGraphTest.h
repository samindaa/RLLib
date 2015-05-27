/*
 * FiniteStateGraphTest.h
 *
 *  Created on: Oct 25, 2013
 *      Author: sam
 */

#ifndef FINITESTATEGRAPHTEST_H_
#define FINITESTATEGRAPHTEST_H_

#include "Test.h"
//
#include "StateGraph.h"

RLLIB_TEST(FiniteStateGraphTest)

class FiniteStateGraphTest: public FiniteStateGraphTestBase
{
  protected:
    void testSimpleProblemTrajectory();
    void testRandomWalkRightTrajectory();
    void testRandomWalkLeftTrajectory();
    void testComputeSolution();

  public:
    void run();
};
#endif /* FINITESTATEGRAPHTEST_H_ */
