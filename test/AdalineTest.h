/*
 * AdalineTest.h
 *
 *  Created on: Nov 19, 2013
 *      Author: sam
 */

#ifndef ADALINETEST_H_
#define ADALINETEST_H_

#include "HeaderTest.h"

RLLIB_TEST(AdalineTest)
class AdalineTest: public AdalineTestBase
{
  public:
    void run();

  private:
    void testAdaline();
    void testAdalineOnTracking();
    void learnTarget(const Vector<double>* targetWeights, Adaline<double>* learner);
    void updateFeatures(Vector<double>* features);
};

#endif /* ADALINETEST_H_ */
