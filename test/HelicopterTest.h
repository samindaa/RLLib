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
 * HelicopterTest.h
 *
 *  Created on: May 13, 2013
 *      Author: sam
 */

#ifndef HELICOPTERTEST_H_
#define HELICOPTERTEST_H_

#include "Test.h"
#include "Helicopter.h"

RLLIB_TEST(HelicopterTest)

class HelicopterTest: public HelicopterTestBase
{
  public:
    HelicopterTest()
    {
      srand(time(0));
    }

    virtual ~HelicopterTest()
    {
    }

    void run();

  private:

    void testCrash()
    {
      Random<double> random;
      Helicopter<double> helicopter(&random);
      ASSERT(4 == helicopter.getContinuousActions()->getEntry(0)->dimension());
      ASSERT(12 == helicopter.dimension());

      for (int nb = 0; nb < 100; nb++)
      {
        helicopter.initialize();
        while (!helicopter.endOfEpisode())
        {
          helicopter.step(helicopter.getContinuousActions()->getEntry(0));
          ASSERT(helicopter.r() <= 0);
          for (int i = 0; i < helicopter.dimension(); i++)
            ASSERT(helicopter.heliDynamics.ObservationRanges[i].in( //
                helicopter.getTRStep()->observation_tp1->getEntry(i)));
        }
      }
    }

    void testEndEpisode()
    {
      Random<double> random;
      Helicopter<double> helicopter(&random, 2);
      ASSERT(4 == helicopter.getContinuousActions()->getEntry(0)->dimension());
      helicopter.initialize();
      while (!helicopter.endOfEpisode())
      {
        helicopter.step(helicopter.getContinuousActions()->getEntry(0));
        ASSERT(helicopter.r() <= 0);
      }
      ASSERT(2 == helicopter.step_time);
    }

};

#endif /* HELICOPTERTEST_H_ */
