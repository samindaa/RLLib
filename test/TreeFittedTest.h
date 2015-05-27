/*
 * TreeFittedTest.h
 *
 *  Created on: Nov 21, 2014
 *      Author: sam
 */

#ifndef TEST_TREEFITTEDTEST_H_
#define TEST_TREEFITTEDTEST_H_

#include "Test.h"
// Matrix
#include "util/Eigen/Dense"
using Eigen::MatrixXd;

RLLIB_TEST(TreeFittedTest)
class TreeFittedTest: public TreeFittedTestBase
{
  public:
    TreeFittedTest();
    ~TreeFittedTest();
    void run();

  private:
    void testRosenbrock();
    void testRastrigin();
    void testCigar();
    void testRegularizedLinearRegression();
};

#endif /* TEST_TREEFITTEDTEST_H_ */
