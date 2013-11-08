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
 * Acrobot.h
 *
 *  Created on: Sep 12, 2012
 *      Author: sam
 */

#ifndef ACROBOT_H_
#define ACROBOT_H_

#include "Environment.h"

class Acrobot: public Environment<>
{
  protected:
    Range<float>* thetaRange;
    Range<float>* theta1DotRange;
    Range<float>* theta2DotRange;
    Range<float>* actionRange;
    float m1, m2, l1, l2, lc1, lc2, I1, I2, g, dt;
    float targetPosition;
    float theta1, theta2, theta1Dot, theta2Dot;
    float transitionNoise;

  public:
    Acrobot() :
        Environment<>(4, 3, 1), thetaRange(new Range<float>(-M_PI, M_PI)), theta1DotRange(
            new Range<float>(-4.0 * M_PI, 4.0 * M_PI)), theta2DotRange(
            new Range<float>(-9.0 * M_PI, 9.0 * M_PI)), actionRange(new Range<float>(-1.0, 1.0)), m1(
            1.0), m2(1.0), l1(1.0), l2(1.0), lc1(0.5), lc2(0.5), I1(1.0), I2(1.0), g(9.8), dt(0.05), targetPosition(
            1.0), theta1(0), theta2(0), theta1Dot(0), theta2Dot(0), transitionNoise(0)
    {
      discreteActions->push_back(0, actionRange->min());
      discreteActions->push_back(1, 0.0);
      discreteActions->push_back(2, actionRange->max());

      // subject to change
      continuousActions->push_back(0, 0.0);

      for (int i = 0; i < dimension(); i++)
        resolutions->at(i) = 6.0;
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
      bool isRandomStart = false;
      if (isRandomStart) // random
      {
        theta1 = Probabilistic::nextFloat() - 0.5;
        theta2 = Probabilistic::nextFloat() - 0.5;
        theta1Dot = Probabilistic::nextFloat() - 0.5;
        theta2Dot = Probabilistic::nextFloat() - 0.5;
      }
      else
        theta1 = theta2 = theta1Dot = theta2Dot = 0.0; // not random

      updateRTStep();

    }

    void updateRTStep()
    {
      DenseVector<double>& vars = *output->o_tp1;
      //std::cout << (theta * 180 / M_PI) << " " << xDot << std::endl;
      vars[0] = ((theta1 - thetaRange->min()) / thetaRange->length()) * resolutions->at(0);
      vars[1] = ((theta2 - thetaRange->min()) / thetaRange->length()) * resolutions->at(1);
      vars[2] = ((theta1Dot - theta1DotRange->min()) / theta1DotRange->length()) * resolutions->at(2);
      vars[3] = ((theta2Dot - theta2DotRange->min()) / theta2DotRange->length()) * resolutions->at(3);
      output->updateRTStep(r(), z(), endOfEpisode());

      observations->at(0) = theta1;
      observations->at(1) = theta2;
      observations->at(2) = theta1Dot;
      observations->at(3) = theta2Dot;
    }

    void step(const Action* action)
    {
      float torque = actionRange->bound(action->at(0));
      float d1, d2, phi2, phi1, theta2ddot, theta1ddot;

      //torque is in [-1,1]
      //We'll make noise equal to at most +/- 1
      float theNoise = transitionNoise * 2.0 * (Probabilistic::nextFloat() - 0.5);

      torque += theNoise;

      int count = 0;
      while (!endOfEpisode() && count < 4)
      {
        count++;

        d1 = m1 * pow(lc1, 2) + m2 * (pow(l1, 2) + pow(lc2, 2) + 2 * l1 * lc2 * cos(theta2)) + I1
            + I2;
        d2 = m2 * (pow(lc2, 2) + l1 * lc2 * cos(theta2)) + I2;

        phi2 = m2 * lc2 * g * cos(theta1 + theta2 - M_PI / 2.0);
        phi1 = -(m2 * l1 * lc2 * pow(theta2Dot, 2) * sin(theta2)
            - 2 * m2 * l1 * lc2 * theta1Dot * theta2Dot * sin(theta2))
            + (m1 * lc1 + m2 * l1) * g * cos(theta1 - M_PI / 2.0) + phi2;

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

      updateRTStep();
    }

    bool endOfEpisode() const
    {
      float feetHeight = -(l1 * cos(theta1) + l2 * cos(theta2));
      float firstJointEndHeight = l1 * cos(theta1);
      float secondJointEndHeight = l2 * sin((M_PI / 2.0) - theta1 - theta2);
      feetHeight = -(firstJointEndHeight + secondJointEndHeight);
      return (feetHeight > targetPosition);
    }

    float r() const
    {
      return endOfEpisode() ? 0.0 : -1.0;
    }

    float z() const
    {
      return 0;
    }

};

#endif /* ACROBOT_H_ */
