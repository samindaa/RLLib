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

class Acrobot: public RLProblem<double>
{
  protected:
    Range<double>* thetaRange;
    Range<double>* theta1DotRange;
    Range<double>* theta2DotRange;
    double m1;
    double m2;
    double l1;
    double l2;
    double l1Square;
    double l2Square;
    double lc1;
    double lc2;
    double lc1Square;
    double lc2Square;
    double I1;
    double I2;
    double g;
    double delta_t;

    double theta1;
    double theta2;
    double theta1Dot;
    double theta2Dot;
    double targetPosition;
    double transitionNoise;

    Range<double>* actionRange;

  public:
    Acrobot(Random<double>* random) :
        RLProblem<double>(random, 4, 3, 1), thetaRange(new Range<double>(-M_PI, M_PI)), theta1DotRange(
            new Range<double>(-4.0 * M_PI, 4.0 * M_PI)), theta2DotRange(
            new Range<double>(-9.0 * M_PI, 9.0 * M_PI)), m1(1.0), m2(1.0), l1(1.0), l2(1.0), l1Square(
            l1 * l1), l2Square(l2 * l2), lc1(0.5), lc2(0.5), lc1Square(lc1 * lc1), lc2Square(
            lc2 * lc2), I1(1.0), I2(1.0), g(9.8), delta_t(0.05), theta1(0), theta2(0), theta1Dot(0), theta2Dot(
            0), targetPosition(1.0), transitionNoise(0), actionRange(new Range<double>(-1.0, 1.0))
    {
      discreteActions->push_back(0, actionRange->min());
      discreteActions->push_back(1, 0.0);
      discreteActions->push_back(2, actionRange->max());

      // subject to change
      continuousActions->push_back(0, 0.0);

      observationRanges->push_back(thetaRange);
      observationRanges->push_back(theta1DotRange);
      observationRanges->push_back(theta2DotRange);
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
      theta1 = theta2 = theta1Dot = theta2Dot = 0.0; // not random
    }

    void updateTRStep()
    {
      DenseVector<double>& vars = *output->o_tp1;
      vars[0] = thetaRange->toUnit(theta1);
      vars[1] = thetaRange->toUnit(theta2);
      vars[2] = theta1DotRange->toUnit(theta1Dot);
      vars[3] = theta2DotRange->toUnit(theta2Dot);

      observations->at(0) = theta1;
      observations->at(1) = theta2;
      observations->at(2) = theta1Dot;
      observations->at(3) = theta2Dot;
    }

    void step(const Action<double>* action)
    {
      double torque = actionRange->bound(action->getEntry(0));

      //torque is in [-1,1]
      //We'll make noise equal to at most +/- 1
      double theNoise = random ? transitionNoise * 2.0 * (random->nextReal() - 0.5) : 0;

      torque += theNoise;

      double d1 = m1 * lc1Square + m2 * (l1Square + lc2Square + 2 * l1 * lc2 * cos(theta2)) + I1
          + I2;
      double d2 = m2 * (lc2Square + l1 * lc2 * cos(theta2)) + I2;

      double phi2 = m2 * lc2 * g * cos(theta1 + theta2 - M_PI_2);
      double phi1 = -m2 * l1 * lc2 * theta2Dot * sin(theta2) * (theta2Dot - 2.0f * theta1Dot)
          + (m1 * lc1 + m2 * l1) * g * cos(theta1 - M_PI_2) + phi2;

      double accel2 = (torque + phi1 * (d2 / d1)
          - m2 * l1 * lc2 * theta1Dot * theta1Dot * sin(theta2) - phi2);
      accel2 = accel2 / (m2 * lc2Square + I2 - (d2 * d2 / d1));
      double accel1 = -(d2 * accel2 + phi1) / d1;

      // 4 time step advancement
      for (int i = 0; i < 4; i++)
      {
        theta1Dot += accel1 * delta_t;
        theta1Dot = theta1DotRange->bound(theta1Dot);
        theta1 += theta1Dot * delta_t;
        theta1 = thetaRange->bound(theta1);
        theta2Dot += accel2 * delta_t;
        theta2Dot = theta2DotRange->bound(theta2Dot);
        theta2 += theta2Dot * delta_t;
        theta2 = thetaRange->bound(theta2);
      }

    }

    bool endOfEpisode() const
    {
      double feetHeight = -(l1 * cos(theta1) + l2 * cos(theta2));
      double firstJointEndHeight = l1 * cos(theta1);
      double secondJointEndHeight = l2 * sin((M_PI / 2.0) - theta1 - theta2);
      feetHeight = -(firstJointEndHeight + secondJointEndHeight);
      return (feetHeight > targetPosition);
    }

    double r() const
    {
      return endOfEpisode() ? 0.0f : -1.0f;
    }

    double z() const
    {
      return 0.0f;
    }

};

#endif /* ACROBOT_H_ */
