/*
 * Copyright 2013 Saminda Abeyruwan (saminda@cs.miami.edu)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
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
    void testOffPACOnPolicySwingPendulum();

};

#endif /* SWINGPENDULUMTEST_H_ */
