/*
 * TraceTest.cpp
 *
 *  Created on: Dec 18, 2012
 *      Author: sam
 */

#include "TraceTest.h"

RLLIB_TEST_MAKE(TraceTest)

void TraceTest::run()
{
  testATrace();
  testRTrace();
  testAMaxTrace();
}

