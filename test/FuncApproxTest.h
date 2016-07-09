/*
 * FuncApproxTest.h
 *
 *  Created on: Jun 28, 2016
 *      Author: sabeyruw
 */

#ifndef TEST_FUNCAPPROXTEST_H_
#define TEST_FUNCAPPROXTEST_H_

#include "Test.h"

RLLIB_TEST(FuncApproxTest)
class FuncApproxTest: public FuncApproxTestBase
{
  public:
    FuncApproxTest();
    virtual ~FuncApproxTest();
    void run();
};

#endif /* TEST_FUNCAPPROXTEST_H_ */
