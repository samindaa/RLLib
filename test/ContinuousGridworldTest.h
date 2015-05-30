/*
 * Copyright 2015 Saminda Abeyruwan (saminda@cs.miami.edu)
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
 * ContinuousGridworldTest.h
 *
 *  Created on: May 15, 2013
 *      Author: sam
 */

#ifndef CONTINUOUSGRIDWORLDTEST_H_
#define CONTINUOUSGRIDWORLDTEST_H_

#include "Test.h"

// From the RLLib
#include "Vector.h"
#include "Trace.h"
#include "Projector.h"
#include "ControlAlgorithm.h"

// From the simulation
#include "ContinuousGridworld.h"

RLLIB_TEST(ContinuousGridworldTest)

class ContinuousGridworldTest: public ContinuousGridworldTestBase
{
  public:
    ContinuousGridworldTest()
    {
    }

    virtual ~ContinuousGridworldTest()
    {
    }
    void run();

  private:
    void testGreedyGQContinuousGridworld();
    void testOffPACContinuousGridworld();
    void testOffPACContinuousGridworld2();
    void testOffPACOnPolicyContinuousGridworld();
    void testOffPACContinuousGridworldOPtimized();
};

#endif /* CONTINUOUSGRIDWORLDTEST_H_ */
