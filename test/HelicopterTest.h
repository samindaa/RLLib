/*
 * HelicopterTest.h
 *
 *  Created on: May 13, 2013
 *      Author: sam
 */

#ifndef HELICOPTERTEST_H_
#define HELICOPTERTEST_H_

#include "HeaderTest.h"
#include "Helicopter.h"

RLLIB_TEST(HelicopterTest)

class HelicopterTest: public HelicopterTestBase
{
  public:
    HelicopterTest()
    {
      srand(time(0));
      RLLibTestRegistory::registerInstance(this);
    }

    virtual ~HelicopterTest()
    {
    }

    void run();

  private:

    void testCrash()
    {
      Helicopter helicopter;
      assert(4 == helicopter.getContinuousActionList().at(0).dimension());
      assert(12 == helicopter.getVars().dimension());

      for (int nb = 0; nb < 100; nb++)
      {
        helicopter.initialize();
        while (!helicopter.endOfEpisode())
        {
          helicopter.step(helicopter.getContinuousActionList().at(0));
          assert(helicopter.r() <= 0);
          for (int i = 0; i < helicopter.getVars().dimension(); i++)
            assert( helicopter.heliDynamics.ObservationRanges[i].in(helicopter.getVars()[i]));
        }
      }
    }

    void testEndEpisode()
    {
      Helicopter helicopter(2);
      assert(4 == helicopter.getContinuousActionList().at(0).dimension());
      helicopter.initialize();
      while (!helicopter.endOfEpisode())
      {
        helicopter.step(helicopter.getContinuousActionList().at(0));
        assert(helicopter.r() <= 0);
      }
      assert(2 == helicopter.step_time);
    }

};

#endif /* HELICOPTERTEST_H_ */
