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
 * NonMarkovPoleBalancing.h
 *
 *  Created on: Oct 19, 2013
 *      Author: sam
 */

#ifndef NONMARKOVPOLEBALANCING_H_
#define NONMARKOVPOLEBALANCING_H_

#include "RL.h"

using namespace RLLib;
/**
 * Incremental Evolution of Complex General Behavior
 * Faustino Gomez and Risto Miikkulainen, 1996
 *
 */

template<typename T>
class NonMarkovPoleBalancing: public RLProblem<T>
{
    typedef RLProblem<T> Base;
  protected:
    int nbPoles;
    float stepTime; // s
    float x; // m
    float xDot; // ms^{-1}
    float g; // ms^{-2}
    float M; // Kg
    float muc; // coefficient of friction of cart on track
    float threeFourth;
    float fifteenRadian;
    float twelveRadian;

    Range<float>* xRange;
    Range<float>* thetaRange;
    Range<float>* actionRange;

  public:
    Vector<float>* theta;
    Vector<float>* thetaDot;
    Vector<float>* length; // half length of i^{th} pole
    Vector<float>* effectiveForce; // effective force
    Vector<float>* mass; // mass of i^{th} pole
    Vector<float>* effectiveMass; // effective mass
    Vector<float>* mup; // coefficient of friction of i^{th} pole's hinge

  public:
    NonMarkovPoleBalancing(Random<T>* random, const int& nbPoles = 1) :
        RLProblem<T>(random, (1 + nbPoles) * 2, 3, 1), nbPoles(nbPoles), stepTime(0.02), x(0), //
        xDot(0), g(-9.81), M(1.0), muc(0), threeFourth(3.0f / 4.0f), //
        fifteenRadian(15.0f / 180.0f * M_PI), twelveRadian(12.0f / 180.0f * M_PI), //
        xRange(new Range<float>(-2.4f, 2.4f)), actionRange(new Range<float>(-10.0f, 10.0f)), //
        theta(new PVector<float>(nbPoles)), thetaDot(new PVector<float>(nbPoles)), //
        length(new PVector<float>(nbPoles)), effectiveForce(new PVector<float>(nbPoles)), //
        mass(new PVector<float>(nbPoles)), effectiveMass(new PVector<float>(nbPoles)), //
        mup(new PVector<float>(nbPoles))
    {
      assert(nbPoles <= 2);
      if (nbPoles == 2)
      {
        thetaRange = new Range<float>(-fifteenRadian, fifteenRadian);
        length->setEntry(0, 0.5);
        length->setEntry(1, 0.05);
        mass->setEntry(0, 0.1);
        mass->setEntry(1, 0.01);
        muc = 0.0005f;
        mup->setEntry(0, 0.000002f);
        mup->setEntry(1, 0.000002f);
      }
      else
      {
        thetaRange = new Range<float>(-twelveRadian, twelveRadian);
        length->setEntry(0, 0.5); // Kg
        mass->setEntry(0, 0.1); // Kg
        muc = 0.0;
      }

      Base::discreteActions->push_back(0, actionRange->min());
      Base::discreteActions->push_back(1, 0.0);
      Base::discreteActions->push_back(2, actionRange->max());

      // subject to change
      Base::continuousActions->push_back(0, 0.0);
    }

    ~NonMarkovPoleBalancing()
    {
      delete xRange;
      delete thetaRange;
      delete actionRange;
      delete theta;
      delete thetaDot;
      delete length;
      delete effectiveForce;
      delete mass;
      delete effectiveMass;
      delete mup;
    }

  private:
    void adjustTheta()
    {
      for (int i = 0; i < nbPoles; i++)
      {
        if (theta->getEntry(i) >= M_PI)
          theta->setEntry(i, theta->getEntry(i) - 2.0 * M_PI);
        if (theta->getEntry(i) < -M_PI)
          theta->setEntry(i, theta->getEntry(i) + 2.0 * M_PI);
      }
    }

  public:
    void updateTRStep()
    {
      Base::output->observation_tp1->setEntry(0, xRange->bound(x));
      Base::output->observation_tp1->setEntry(1, xDot);
      Base::output->o_tp1->setEntry(0, Base::output->observation_tp1->getEntry(0)); //<<FixMe: only for testing
      Base::output->o_tp1->setEntry(1, Base::output->observation_tp1->getEntry(1));

      for (int i = 0; i < nbPoles; i += 2)
      {
        Base::output->observation_tp1->setEntry(i + 2, theta->getEntry(i));
        Base::output->observation_tp1->setEntry(i + 3, thetaDot->getEntry(i));
        Base::output->o_tp1->setEntry(i + 2, Base::output->observation_tp1->getEntry(i + 2)); //<<FixMe: only for testing
        Base::output->o_tp1->setEntry(i + 3, Base::output->observation_tp1->getEntry(i + 3));
      }

    }

    void initialize()
    {
      if (Base::random)
      {
        Range<T> xs1(-2, 2);
        Range<T> thetas1(-0.6, 0.6);
        Range<T> xs2(-0.2, 0.2);
        Range<T> thetas2(-0.2, 0.2);

        //<<FixMe: S2
        x = xs2.choose(Base::random);
        for (int i = 0; i < nbPoles; i++)
          theta->setEntry(i, thetas2.choose(Base::random));
      }
      else
      {
        x = 0.0;
        for (int i = 0; i < nbPoles; i++)
          theta->setEntry(i, 0.0f);
      }

      xDot = 0;
      for (int i = 0; i < nbPoles; i++)
        thetaDot->setEntry(i, 0.0f);

      adjustTheta();
    }

    void step(const Action<T>* a)
    {
      float totalEffectiveForce = 0;
      float totalEffectiveMass = 0;
      for (int i = 0; i < nbPoles; i++)
      {
        double effForce = mass->getEntry(i) * length->getEntry(i) * pow(thetaDot->getEntry(i), 2)
            * sin(theta->getEntry(i));
        effForce += threeFourth * mass->getEntry(i) * cos(theta->getEntry(i))
            * ((mup->getEntry(i) * thetaDot->getEntry(i))
                / (mass->getEntry(i) * length->getEntry(i)) + g * sin(theta->getEntry(i)));
        effectiveForce->setEntry(i, effForce);
        effectiveMass->setEntry(i,
            mass->getEntry(i) * (1.0f - threeFourth * pow(cos(theta->getEntry(i)), 2)));
        totalEffectiveForce += effectiveForce->getEntry(i);
        totalEffectiveMass += effectiveMass->getEntry(i);
      }

      float torque = actionRange->bound(a->getEntry(0));
      float xAcc = (torque - muc * Signum::valueOf(xDot) + totalEffectiveForce)
          / (M + totalEffectiveMass);

      // Update the four state variables, using Euler's method.
      x += xDot * stepTime;
      xDot += xAcc * stepTime;

      for (int i = 0; i < nbPoles; i++)
      {
        float thetaDotDot = -threeFourth
            * (xAcc * cos(theta->getEntry(i)) + g * sin(theta->getEntry(i))
                + (mup->getEntry(i) * thetaDot->getEntry(i))
                    / (mass->getEntry(i) * length->getEntry(i))) / length->getEntry(i);

        // Update the four state variables, using Euler's method.
        theta->setEntry(i, theta->getEntry(i) + thetaDot->getEntry(i) * stepTime);
        thetaDot->setEntry(i, thetaDot->getEntry(i) + thetaDotDot * stepTime);
      }

      adjustTheta();
    }

    bool endOfEpisode() const
    {
      bool value = true;
      for (int i = 0; i < nbPoles; i++)
        value *= thetaRange->in(theta->getEntry(i));
      value = value && xRange->in(x);
      return !value;
    }

    T r() const
    {
      float value = 0.0f;
      for (int i = 0; i < nbPoles; i++)
        value += cos(theta->getEntry(i));
      return value;
    }

    T z() const
    {
      return 0.0f;
    }

};

#endif /* NONMARKOVPOLEBALANCING_H_ */
