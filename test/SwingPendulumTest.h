/*
 * SwingPendulumTest.h
 *
 *  Created on: May 15, 2013
 *      Author: sam
 */

#ifndef SWINGPENDULUMTEST_H_
#define SWINGPENDULUMTEST_H_

#include "HeaderTest.h"

// From the RLLib
#include "Vector.h"
#include "Trace.h"
#include "Projector.h"
#include "ControlAlgorithm.h"
#include "Representation.h"

// From the simulation
#include "SwingPendulum.h"
#include "Simulator.h"

RLLIB_TEST(SwingPendulumTest)

class SwingPendulumTest: public SwingPendulumTestBase
{
  public:
    SwingPendulumTest() {}

    virtual ~SwingPendulumTest() {}
    void run();

  private:
    void testOffPACSwingPendulum();
    void testOnPolicySwingPendulum();
    void testOffPACSwingPendulum2();

};

#endif /* SWINGPENDULUMTEST_H_ */
