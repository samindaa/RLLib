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

using namespace RLLib;

template<class T>
class CartPole: public RLProblem<T>
{
    typedef RLProblem<T> Base;
  protected:
    float gravity;
    float massCart;
    float massPole;
    float totalMass;
    float length;
    float poleMassLength;
    float forceMag;
    float tau;
    float fourthirds;
    float twelveRadians;

    float x, x_dot, theta, theta_dot;
    Range<float>* forceRange;
    Range<T> *xRange, *thetaRange;
  public:
    CartPole(Random<T>* random) :
        RLProblem<T>(random, 4, 3, 1), gravity(9.8), massCart(1.0), massPole(0.1), totalMass(
            massPole + massCart), length(0.5), poleMassLength(massPole * length), forceMag(10.0), tau(
            0.02), fourthirds(4.0f / 3.0f), twelveRadians(-12.0f / 180.0f * M_PI), x(0), x_dot(0), theta(
            0), theta_dot(0), forceRange(new Range<float>(-forceMag, forceMag)), xRange(
            new Range<T>(-2.4, 2.4)), thetaRange(new Range<T>(-twelveRadians, twelveRadians))
    {
      Base::discreteActions->push_back(0, forceRange->min());
      Base::discreteActions->push_back(1, 0.0);
      Base::discreteActions->push_back(2, forceRange->max());

      // subject to change
      Base::continuousActions->push_back(0, 0.0);

      Base::observationRanges->push_back(xRange);
      Base::observationRanges->push_back(thetaRange);
    }

    virtual ~CartPole()
    {
      delete forceRange;
      delete xRange;
      delete thetaRange;
    }

    void updateTRStep()
    {
      DenseVector<T>& vars = *Base::output->o_tp1;
      Base::output->updateTRStep(r(), z(), endOfEpisode());

      Base::observations->at(0) = x;
      Base::observations->at(1) = x_dot;
      Base::observations->at(2) = theta;
      Base::observations->at(3) = theta_dot;

      vars[0] = xRange->toUnit(x);
      vars[1] = x_dot; // FixME
      vars[2] = thetaRange->toUnit(theta);
      vars[3] = theta_dot; // FixME

    }

    // Profiles
    void initialize()
    {
      if (Base::random)
      {
        Range<T> xs2(-0.2, 0.2);
        Range<T> thetas2(-0.2, 0.2);
        x_dot = theta_dot = 0;
        x = xs2.choose(Base::random);
        theta = thetas2.choose(Base::random);
      }
      else
        x = x_dot = theta = theta_dot = 0; //<< fixMe with noise
    }

    void step(const Action<T>* a)
    {
      float xacc, thetaacc, force, costheta, sintheta, temp;
      force = forceRange->bound(a->at(0));
      costheta = cos(theta);
      sintheta = sin(theta);
      temp = (force + poleMassLength * theta_dot * theta_dot * sintheta) / totalMass;
      thetaacc = (gravity * sintheta - costheta * temp)
          / (length * (fourthirds - massPole * costheta * costheta / totalMass));
      xacc = temp - poleMassLength * thetaacc * costheta / totalMass;

      x += tau * x_dot;
      x_dot += tau * xacc;
      theta += tau * theta_dot;
      theta_dot += tau * thetaacc;
    }

    bool endOfEpisode() const
    {
      return !(xRange->in(x) || thetaRange->in(theta));
    }

    T r() const
    {
      return cos(theta);
    }

    T z() const
    {
      return 0.0f;
    }

};

#endif /* CARTPOLE_H_ */
