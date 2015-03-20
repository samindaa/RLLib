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
 * CartPole.h
 *
 *  Created on: Nov 21, 2013
 *      Author: sam
 *
 *  This is the original cart-pole problem from:
 *  http://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
 *
 *  This problem uses Euler's method to simulate the plant.
 *  The implementation regulates in a deterministic and stochastic environments.
 */

#ifndef CARTPOLE_H_
#define CARTPOLE_H_

#include "RL.h"

class CartPole: public RLLib::RLProblem<double>
{
  protected:
    double gravity;
    double massCart;
    double massPole;
    double totalMass;
    double length;
    double poleMassLength;
    double forceMag;
    double tau;
    double fourthirds;

    double x, x_dot, theta, theta_dot;
    RLLib::Range<double>* forceRange;
    RLLib::Range<double> *xRange, *xDotRange, *thetaRange, *thetaDotRange;

    float previousTheta, cumulatedRotation;
    bool overRotated;
    float overRotatedTime;
    int upTime;

  public:
    CartPole(Random<double>* random = 0) :
        RLLib::RLProblem<double>(random, 4, 3, 1), gravity(9.8), massCart(1.0), massPole(0.1), //
        totalMass(massPole + massCart), length(0.5/* actually half the pole's length */), //
        poleMassLength(massPole * length), forceMag(10.0), //
        tau(0.02/* seconds between state updates */), fourthirds(4.0f / 3.0f), //
        x(0/* cart position, meters */), x_dot(0/* cart velocity */), //
        theta(0/* pole angle, radians */), theta_dot(0/* pole angular velocity */), //
        forceRange(new RLLib::Range<double>(-forceMag, forceMag)), //
        xRange(new RLLib::Range<double>(-2.4, 2.4)), xDotRange(new RLLib::Range<double>(-2.4, 2.4)), //
        thetaRange(new RLLib::Range<double>(-M_PI, M_PI)), //
        thetaDotRange(new RLLib::Range<double>(-4.0 * M_PI, 4.0 * M_PI)), previousTheta(0), //
        cumulatedRotation(0), overRotated(false), overRotatedTime(0), upTime(0)
    {
      discreteActions->push_back(0, forceRange->min());
      discreteActions->push_back(1, 0.0);
      discreteActions->push_back(2, forceRange->max());

      // subject to change
      continuousActions->push_back(0, 0.0);

      observationRanges->push_back(xRange);
      observationRanges->push_back(xDotRange);
      observationRanges->push_back(thetaRange);
      observationRanges->push_back(thetaDotRange);
    }

    virtual ~CartPole()
    {
      delete forceRange;
      delete xRange;
      delete xDotRange;
      delete thetaRange;
      delete thetaDotRange;
    }

    void updateTRStep()
    {
      output->observation_tp1->setEntry(0, x);
      output->observation_tp1->setEntry(1, x_dot);
      output->observation_tp1->setEntry(2, theta);
      output->observation_tp1->setEntry(3, theta_dot);

      output->o_tp1->setEntry(0, xRange->toUnit(x));
      output->o_tp1->setEntry(1, xDotRange->toUnit(x_dot));
      output->o_tp1->setEntry(2, thetaRange->toUnit(theta));
      output->o_tp1->setEntry(3, thetaDotRange->toUnit(theta_dot));

    }

    // Profiles
    void initialize()
    {
      if (random)
      {
        x_dot = theta_dot = 0;
        x = xRange->choose(random);
        theta = thetaRange->choose(random);
      }
      else
        x = x_dot = theta = theta_dot = 0;

      previousTheta = theta;
      cumulatedRotation = theta;
      overRotated = false;
      overRotatedTime = 0;
      upTime = 0;
    }

    void step(const RLLib::Action<double>* a)
    {
      double xacc, thetaacc, force, costheta, sintheta, temp;
      force = forceRange->bound(a->getEntry(0));
      costheta = cos(theta);
      sintheta = sin(theta);
      temp = (force + poleMassLength * theta_dot * theta_dot * sintheta) / totalMass;
      thetaacc = (gravity * sintheta - costheta * temp)
          / (length * (fourthirds - massPole * costheta * costheta / totalMass));
      xacc = temp - poleMassLength * thetaacc * costheta / totalMass;

      x += tau * x_dot;
      x_dot += tau * xacc;
      x_dot = xDotRange->bound(x_dot);
      theta += tau * theta_dot;
      theta_dot += tau * thetaacc;
      theta_dot = thetaDotRange->bound(theta_dot);
      theta = RLLib::Angle::normalize(theta);

      float signAngleDifference = std::atan2(std::sin(theta - previousTheta),
          std::cos(theta - previousTheta));
      cumulatedRotation += signAngleDifference;
      if (!overRotated && std::abs(cumulatedRotation) > 5.0f * M_PI)
        overRotated = true;
      if (overRotated)
        overRotatedTime += 1;
      ++upTime;
      previousTheta = theta;
    }

    bool endOfEpisode() const
    {
      if (!xRange->in(x))
        return true;
      // Reinforcement Learning in Continuous Time and Space (Kenji Doya)
      if (overRotated && (overRotatedTime > (0.5 / tau)))
        return true;
      // Reinforcement Learning in Continuous Time and Space (Kenji Doya)
      //if (upTime >= (20.0/*seconds*// tau))
      //  return true;

      return false;
    }

    double r() const
    {
      // Reinforcement Learning in Continuous Time and Space (Kenji Doya)
      if (overRotated || !xRange->in(x))
        return -1.0f;
      else
        return (cos(theta) - 1.0) / 2.0;
    }

    double z() const
    {
      return 0.0f;
    }

};

#endif /* CARTPOLE_H_ */
