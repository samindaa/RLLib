/*
 * LearningAlgorithmTest.h
 *
 *  Created on: May 12, 2013
 *      Author: sam
 */

#ifndef LEARNINGALGORITHMTEST_H_
#define LEARNINGALGORITHMTEST_H_

#include "HeaderTest.h"
#include "Projector.h"
#include "SupervisedAlgorithm.h"

RLLIB_TEST(SupervisedAlgorithmTest)

class SupervisedAlgorithmTest: public SupervisedAlgorithmTestBase
{
  public:
    SupervisedAlgorithmTest()
    {
      srand(time(0));
    }

    virtual ~SupervisedAlgorithmTest()
    {
    }
    void run();
};

#endif /* LEARNINGALGORITHMTEST_H_ */
