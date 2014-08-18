/*
 * Copyright 2014 Saminda Abeyruwan (saminda@cs.miami.edu)
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
 * Acrobot.h
 *
 *  Created on: Sep 12, 2012
 *      Author: sam
 */

#ifndef ACROBOT_H_
#define ACROBOT_H_

#include "RL.h"

template<class T>
class Acrobot: public RLProblem<T>
{
    typedef RLProblem<T> Base;
  protected:
    Range<T>* thetaRange;
    Range<T>* theta1DotRange;
    Range<T>* theta2DotRange;
    Range<float>* actionRange;
    float m1, m2, l1, l2, lc1, lc2, I1, I2, g, dt;
    float targetPosition;
    float theta1, theta2, theta1Dot, theta2Dot;
    float transitionNoise;

  public:
    Acrobot(Random<T>* random) :
        RLProblem<T>(random, 4, 3, 1), thetaRange(new Range<T>(-M_PI, M_PI)), theta1DotRange(
            new Range<T>(-4.0 * M_PI, 4.0 * M_PI)), theta2DotRange(
            new Range<T>(-9.0 * M_PI, 9.0 * M_PI)), actionRange(new Range<float>(-1.0, 1.0)), m1(
            1.0), m2(1.0), l1(1.0), l2(1.0), lc1(0.5), lc2(0.5), I1(1.0), I2(1.0), g(9.8), dt(0.05), targetPosition(
            1.0), theta1(0), theta2(0), theta1Dot(0), theta2Dot(0), transitionNoise(0)
    {
      Base::discreteActions->push_back(0, actionRange->min());
      Base::discreteActions->push_back(1, 0.0);
      Base::discreteActions->push_back(2, actionRange->max());

      // subject to change
      Base::continuousActions->push_back(0, 0.0);

      Base::observationRanges->push_back(thetaRange);
      Base::observationRanges->push_back(theta1DotRange);
      Base::observationRanges->push_back(theta2DotRange);
    }

    ~Acrobot()
    {
      delete thetaRange;
      delete theta1DotRange;
      delete theta2DotRange;
      delete actionRange;
    }

    void initialize()
    {
      if (Base::random) // random
      {
        theta1 = (Base::random->nextReal() * (M_PI + fabs(-M_PI)) + (-M_PI)) * 0.1f;
        theta2 = (Base::random->nextReal() * (M_PI + fabs(-M_PI)) + (-M_PI)) * 0.1f;
        theta1Dot = (Base::random->nextReal() * (theta1DotRange->max() * 2.0f)
            - theta1DotRange->max()) * 0.1f;
        theta2Dot = (Base::random->nextReal() * (theta2DotRange->max() * 2.0f)
            - theta1DotRange->max()) * 0.1f;
      }
      else
        theta1 = theta2 = theta1Dot = theta2Dot = 0.0; // not random
    }

    void updateTRStep()
    {
      DenseVector<T>& vars = *Base::output->o_tp1;
      vars[0] = thetaRange->toUnit(theta1);
      vars[1] = thetaRange->toUnit(theta2);
      vars[2] = theta1DotRange->toUnit(theta1Dot);
      vars[3] = theta2DotRange->toUnit(theta2Dot);

      Base::observations->at(0) = theta1;
      Base::observations->at(1) = theta2;
      Base::observations->at(2) = theta1Dot;
      Base::observations->at(3) = theta2Dot;
    }

    void step(const Action<double>* action)
    {
      float torque = actionRange->bound(action->at(0));
      float d1, d2, phi2, phi1, theta2ddot, theta1ddot;

      //torque is in [-1,1]
      //We'll make noise equal to at most +/- 1
      float theNoise = Base::random ? transitionNoise * 2.0 * (Base::random->nextReal() - 0.5) : 0;

      torque += theNoise;

      int count = 0;
      while (!endOfEpisode() && count < 4)
      {
        count++;

        d1 = m1 * pow(lc1, 2) + m2 * (pow(l1, 2) + pow(lc2, 2) + 2 * l1 * lc2 * cos(theta2)) + I1
            + I2;
        d2 = m2 * (pow(lc2, 2) + l1 * lc2 * cos(theta2)) + I2;

        phi2 = m2 * lc2 * g * cos(theta1 + theta2 - M_PI_2);
        phi1 = -(m2 * l1 * lc2 * pow(theta2Dot, 2) * sin(theta2)
            - 2 * m2 * l1 * lc2 * theta1Dot * theta2Dot * sin(theta2))
            + (m1 * lc1 + m2 * l1) * g * cos(theta1 - M_PI_2) + phi2;

        theta2ddot = (torque + (d2 / d1) * phi1 - m2 * l1 * lc2 * pow(theta1Dot, 2) * sin(theta2)
            - phi2) / (m2 * pow(lc2, 2) + I2 - pow(d2, 2) / d1);
        theta1ddot = -(d2 * theta2ddot + phi1) / d1;

        theta1Dot += theta1ddot * dt;
        theta2Dot += theta2ddot * dt;

        theta1 += theta1Dot * dt;
        theta2 += theta2Dot * dt;
      }
      theta1Dot = theta1DotRange->bound(theta1Dot);
      theta2Dot = theta2DotRange->bound(theta2Dot);

      /* Put a hard constraint on the Acrobot physics, thetas MUST be in [-PI,+PI]
       * if they reach a top then angular velocity becomes zero
       */
      if (fabs(theta2) > M_PI)
      {
        theta2 = thetaRange->bound(theta2);
        theta2Dot = 0;
      }
      if (fabs(theta1) > M_PI)
      {
        theta1 = thetaRange->bound(theta1);
        theta1Dot = 0;
      }
    }

    bool endOfEpisode() const
    {
      float feetHeight = -(l1 * cos(theta1) + l2 * cos(theta2));
      float firstJointEndHeight = l1 * cos(theta1);
      float secondJointEndHeight = l2 * sin((M_PI / 2.0) - theta1 - theta2);
      feetHeight = -(firstJointEndHeight + secondJointEndHeight);
      return (feetHeight > targetPosition);
    }

    T r() const
    {
      return endOfEpisode() ? 0.0f : -1.0f;
    }

    T z() const
    {
      return 0.0f;
    }

};

#endif /* ACROBOT_H_ */
