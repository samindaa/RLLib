/*
 * FourierBasisTest.h
 *
 *  Created on: Aug 5, 2016
 *      Author: sabeyruw
 */

#ifndef TEST_FOURIERBASISTEST_H_
#define TEST_FOURIERBASISTEST_H_

#include "Test.h"
#include "FourierBasis.h"

RLLIB_TEST(FourierBasisTest)

class FourierBasisTest: public FourierBasisTestBase
{
  public:
    FourierBasisTest()
    {
    }

    virtual ~FourierBasisTest()
    {
    }
    void run();

  private:
    void testFourierBasis1();
    void testFourierBasis2();
    void testFourierBasis3();
};

#endif /* TEST_FOURIERBASISTEST_H_ */
