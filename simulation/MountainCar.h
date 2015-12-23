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
 * MountainCar.h
 *
 *  Created on: Nov 8, 2013
 *      Author: sam
 */

#ifndef MOUNTAINCAR_H_
#define MOUNTAINCAR_H_

#include "RL.h"

template<typename T>
class MountainCar: public RLLib::RLProblem<T>
{
  private:
    typedef RLLib::RLProblem<T> Base;
  protected:
    // Global variables:
    float position;
    float velocity;

    RLLib::Range<T>* positionRange;
    RLLib::Range<T>* velocityRange;
    RLLib::Range<float>* actionRange;

    float targetPosition;
    float throttleFactor;

  public:
    MountainCar(RLLib::Random<T>* random = 0) :
        RLLib::RLProblem<T>(random, 2, 3, 1), position(0), velocity(0), //
        positionRange(new RLLib::Range<T>(-1.2, 0.6)), //
        velocityRange(new RLLib::Range<T>(-0.07, 0.07)), //
        actionRange(new RLLib::Range<float>(-1.0, 1.0)), targetPosition(positionRange->max()), //
        throttleFactor(1.0)
    {
      Base::discreteActions->push_back(0, actionRange->min());
      Base::discreteActions->push_back(1, 0.0);
      Base::discreteActions->push_back(2, actionRange->max());

      // subject to change
      Base::continuousActions->push_back(0, 0.0);

      Base::observationRanges->push_back(positionRange);
      Base::observationRanges->push_back(velocityRange);
    }

    virtual ~MountainCar()
    {
      delete positionRange;
      delete velocityRange;
      delete actionRange;
    }

    void updateTRStep()
    {
      Base::output->o_tp1->setEntry(0, positionRange->toUnit(position));
      Base::output->o_tp1->setEntry(1, velocityRange->toUnit(velocity));

      Base::output->observation_tp1->setEntry(0, position);
      Base::output->observation_tp1->setEntry(1, velocity);

    }

    // Profiles
    void initialize()
    {
      if (Base::random)
      {
        position = positionRange->choose(Base::random);
        velocity = velocityRange->choose(Base::random);
      }
      else
      {
        position = -0.5;
        velocity = 0.0;
      }
    }

    void step(const RLLib::Action<T>* a)
    {
      float throttle = actionRange->bound(a->getEntry()) * throttleFactor;
      velocity = velocityRange->bound(
          velocity + throttle * 0.001 + cos(3.0 * position) * (-0.0025));
      position += velocity;
      if (position < positionRange->min())
        velocity = 0.0;
      position = positionRange->bound(position);
    }

    bool endOfEpisode() const
    {
      return (position >= targetPosition);
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

#endif /* MOUNTAINCAR_H_ */
