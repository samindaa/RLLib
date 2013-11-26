/*
 * NAOTest.h
 *
 *  Created on: Nov 15, 2013
 *      Author: sam
 */

#ifndef NAOTEST_H_
#define NAOTEST_H_

#include "Test.h"

RLLIB_TEST(NAOTest)
class NAOTest: public NAOTestBase
{
  public:
    NAOTest();
    ~NAOTest();
    void run();

  protected:
    void testTrain();
    void testEvaluate();
};

#endif /* NAOTEST_H_ */
