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
 * MountainCar3D.h
 *
 *  Created on: Dec 2, 2013
 *      Author: sam
 */

#ifndef MOUNTAINCAR3D_H_
#define MOUNTAINCAR3D_H_

#include "RL.h"
/******************************************************************************
 *  Author: Sam Abeyruwan
 *
 *  Based on MontainCar3DSym.cc, created by Matthew Taylor
 *           (Based on MountainCar.cc, created by Adam White,
 *            created on March 29 2007.)
 *
 *    Episodic Task
 *    Reward: -1 per step
 *    Actions: Discrete
 *          0 - coast
 *          1 - left
 *          2 - right
 *          3 - down
 *          4 - up
 *
 *    State: 3D Continuous
 *          car's x-position (-1.2 to .6)
 *          car's y-position (-1.2 to .6)
 *          car's x-velocity (-.07 to .07)
 *          car's y-velocity (-.07 to .07)
 *
 ******************************************************************************
 */
using namespace RLLib;

template<typename T>
class MountainCar3D: public RLProblem<T>
{
    typedef RLProblem<T> Base;
  protected:
    float xposition;
    float yposition;
    float xvelocity;
    float yvelocity;

    float offset;
    float targetPosition;

    Range<T>* positionRange;
    Range<T>* velocityRange;

  public:
    MountainCar3D(Random<T>* random) :
        RLProblem<T>(random, 4, 5, 1), xposition(0), yposition(0), xvelocity(0), yvelocity(0), //
        offset(0), targetPosition(0.5), positionRange(new Range<T>(-1.2, 0.5)), //
        velocityRange(new Range<T>(-0.07, 0.07))
    {

      for (int a = 0; a < Base::discreteActions->dimension(); a++)
        Base::discreteActions->push_back(a, a);
      // not used
      Base::continuousActions->push_back(0, 0.0);

      Base::observationRanges->push_back(positionRange);
      Base::observationRanges->push_back(velocityRange);
    }

    virtual ~MountainCar3D()
    {
      delete positionRange;
      delete velocityRange;
    }

  private:

    void set_initial_position_random()
    {
      xposition = positionRange->min()
          + Base::random->nextReal() * ((positionRange->max() - 0.2f) - positionRange->min());
      yposition = positionRange->min()
          + Base::random->nextReal() * ((positionRange->max() - 0.2f) - positionRange->min());
      xvelocity = 0.0;
      yvelocity = 0.0;
    }

    void set_initial_position_at_bottom()
    {
      xposition = 0.; //-M_PI / 6.0 + offset;
      yposition = 0.; //-M_PI / 6.0 + offset;
      xvelocity = 0.;
      yvelocity = 0.;
    }

    void update_velocity(const Action<double>* act)
    {

      switch (act->id())
      {
        case 0:
          xvelocity += cos(3 * xposition) * (-0.0025);
          yvelocity += cos(3 * yposition) * (-0.0025);
          break;
        case 1:
          xvelocity += -0.001 + cos(3 * xposition) * (-0.0025);
          yvelocity += cos(3 * yposition) * (-0.0025);
          break;
        case 2:
          xvelocity += +0.001 + cos(3 * xposition) * (-0.0025);
          yvelocity += cos(3 * yposition) * (-0.0025);
          break;
        case 3:
          xvelocity += cos(3 * xposition) * (-0.0025);
          yvelocity += -0.001 + cos(3 * yposition) * (-0.0025);
          break;
        case 4:
          xvelocity += cos(3 * xposition) * (-0.0025);
          yvelocity += +0.001 + cos(3 * yposition) * (-0.0025);
          break;
      }

      //xvelocity *= get_gaussian(1.0,std_dev_eff);
      //yvelocity *= get_gaussian(1.0,std_dev_eff);
      xvelocity = velocityRange->bound(xvelocity);
      yvelocity = velocityRange->bound(yvelocity);
    }

    void update_position()
    {
      xposition += xvelocity;
      yposition += yvelocity;
      xposition = positionRange->bound(xposition);
      yposition = positionRange->bound(yposition);
    }

    void updateTRStep()
    {
      Base::output->o_tp1->setEntry(0, positionRange->toUnit(xposition));
      Base::output->o_tp1->setEntry(1, positionRange->toUnit(yposition));
      Base::output->o_tp1->setEntry(2, velocityRange->toUnit(xvelocity));
      Base::output->o_tp1->setEntry(3, velocityRange->toUnit(yvelocity));

      Base::output->observation_tp1->setEntry(0, xposition);
      Base::output->observation_tp1->setEntry(1, yposition);
      Base::output->observation_tp1->setEntry(2, xvelocity);
      Base::output->observation_tp1->setEntry(3, yvelocity);
    }

  public:
    void initialize()
    {
//      set_initial_position_at_bottom();
      set_initial_position_random();
    }

    void step(const Action<double>* a)
    {
      update_velocity(a);
      update_position();
    }

    bool endOfEpisode() const
    {
      return ((xposition >= targetPosition) && (yposition >= targetPosition));
    }

    T r() const
    {
      return -1.0f;
    }

    T z() const
    {
      return 0.0f;
    }

};

#endif /* MOUNTAINCAR3D_H_ */
