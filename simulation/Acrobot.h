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
 * Acrobot.h
 *
 *  Created on: Sep 12, 2012
 *      Author: sam
 */

#ifndef ACROBOT_H_
#define ACROBOT_H_

#include "RL.h"

class Acrobot: public RLLib::RLProblem<double>
{
  protected:
    RLLib::Range<double>* thetaRange;
    RLLib::Range<double>* theta1DotRange;
    RLLib::Range<double>* theta2DotRange;
    double m1;
    double m2;
    double l1;
    double l2;
    double lc1;
    double lc2;
    double I1;
    double I2;
    double g;
    double dt;
    double acrobotGoalPosition;

    double theta1;
    double theta2;
    double theta1Dot;
    double theta2Dot;
    double targetPosition;
    double transitionNoise;

    RLLib::Range<double>* actionRange;

  public:
    Acrobot(RLLib::Random<double>* random) :
        RLProblem<double>(random, 4, 3, 1), thetaRange(new RLLib::Range<double>(-M_PI, M_PI)), //
        theta1DotRange(new RLLib::Range<double>(-4.0 * M_PI, 4.0 * M_PI)), //
        theta2DotRange(new RLLib::Range<double>(-9.0 * M_PI, 9.0 * M_PI)), m1(1.0), m2(1.0), //
        l1(1.0), l2(1.0), lc1(0.5), lc2(0.5), I1(1.0), I2(1.0), g(9.8), dt(0.05), //
        acrobotGoalPosition(1.0), theta1(0), theta2(0), theta1Dot(0), theta2Dot(0), //
        targetPosition(1.0), transitionNoise(0), actionRange(new RLLib::Range<double>(-1.0, 1.0))
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
      output->o_tp1->setEntry(0, thetaRange->toUnit(theta1));
      output->o_tp1->setEntry(1, thetaRange->toUnit(theta2));
      output->o_tp1->setEntry(2, theta1DotRange->toUnit(theta1Dot));
      output->o_tp1->setEntry(3, theta2DotRange->toUnit(theta2Dot));

      output->observation_tp1->setEntry(0, theta1);
      output->observation_tp1->setEntry(1, theta2);
      output->observation_tp1->setEntry(2, theta1DotRange->bound(theta1Dot));
      output->observation_tp1->setEntry(3, theta2DotRange->bound(theta2Dot));
    }

    void step(const RLLib::Action<double>* action)
    {
      double torque = actionRange->bound(action->getEntry(0));

      //torque is in [-1,1]
      //We'll make noise equal to at most +/- 1
      double theNoise = random ? transitionNoise * 2.0 * (random->nextReal() - 0.5) : 0;
      torque += theNoise;

      int stepAdvancement = 0;
      double d1;
      double d2;
      double phi_2;
      double phi_1;
      double theta2_ddot;
      double theta1_ddot;

      while (!endOfEpisode() && (stepAdvancement++ < 4))
      {
        d1 = m1 * std::pow(lc1, 2)
            + m2 * (std::pow(l1, 2) + std::pow(lc2, 2) + 2 * l1 * lc2 * std::cos(theta2)) + I1 + I2;
        d2 = m2 * (std::pow(lc2, 2) + l1 * lc2 * std::cos(theta2)) + I2;

        phi_2 = m2 * lc2 * g * std::cos(theta1 + theta2 - M_PI_2);
        phi_1 = -(m2 * l1 * lc2 * std::pow(theta2Dot, 2) * std::sin(theta2)
            - 2 * m2 * l1 * lc2 * theta1Dot * theta2Dot * std::sin(theta2))
            + (m1 * lc1 + m2 * l1) * g * std::cos(theta1 - M_PI_2) + phi_2;

        theta2_ddot = (torque + (d2 / d1) * phi_1
            - m2 * l1 * lc2 * std::pow(theta1Dot, 2) * std::sin(theta2) - phi_2)
            / (m2 * std::pow(lc2, 2) + I2 - std::pow(d2, 2) / d1);
        theta1_ddot = -(d2 * theta2_ddot + phi_1) / d1;

        theta1Dot += theta1_ddot * dt;
        theta2Dot += theta2_ddot * dt;

        theta1 += theta1Dot * dt;
        theta2 += theta2Dot * dt;

        theta1Dot = theta1DotRange->bound(theta1Dot);
        theta2Dot = theta2DotRange->bound(theta2Dot);

        /* Put a hard constraint on the Acrobot physics, thetas MUST be in [-PI,+PI]
         * if they reach a top then angular velocity becomes zero
         */
        if (!thetaRange->in(theta1))
        {
          theta1 = thetaRange->bound(theta1);
          theta1Dot = 0;
        }

        if (!thetaRange->in(theta2))
        {
          theta2 = thetaRange->bound(theta2);
          theta2Dot = 0;
        }
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
