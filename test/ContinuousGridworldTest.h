/*
 * ContinuousGridworldTest.h
 *
 *  Created on: May 15, 2013
 *      Author: sam
 */

#ifndef CONTINUOUSGRIDWORLDTEST_H_
#define CONTINUOUSGRIDWORLDTEST_H_

#include "HeaderTest.h"

// From the RLLib
#include "Vector.h"
#include "Trace.h"
#include "Projector.h"
#include "ControlAlgorithm.h"
#include "Representation.h"

// From the simulation
#include "ContinuousGridworld.h"
#include "Simulator.h"

RLLIB_TEST(ContinuousGridworldTest)

class ContinuousGridworldTest: public ContinuousGridworldTestBase
{
  public:
    ContinuousGridworldTest() {}

    virtual ~ContinuousGridworldTest() {}
    void run();

  private:
    void testGreedyGQContinuousGridworld();
    void testOffPACContinuousGridworld();

    void testOffPACOnPolicyContinuousGridworld();

    void testOffPACContinuousGridworldOPtimized();
};

#endif /* CONTINUOUSGRIDWORLDTEST_H_ */
