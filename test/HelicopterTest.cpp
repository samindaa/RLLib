/*
 * HelicopterTest.cpp
 *
 *  Created on: May 13, 2013
 *      Author: sam
 */

#include "HelicopterTest.h"

RLLIB_TEST_MAKE(HelicopterTest)

void HelicopterTest::run()
{
  testEndEpisode();
  testCrash();
}

