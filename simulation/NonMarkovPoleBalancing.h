/*
 * NonMarkovPoleBalancing.h
 *
 *  Created on: Oct 19, 2013
 *      Author: sam
 */

#ifndef NONMARKOVPOLEBALANCING_H_
#define NONMARKOVPOLEBALANCING_H_

#include "Environment.h"

/**
 * Incremental Evolution of Complex General Behavior
 * Faustino Gomez and Risto Miikkulainen, 1996
 *
 */
class NonMarkovPoleBalancing: public Environment<float>
{
  protected:
    int nbUnjointedPoles;
    float stepTime; // s
    float x; // m
    float xDot; // ms^{-1}
    float g; // ms^{-2}
    float M; // Kg
    float muc; // coefficient of friction of cart on track

    Range<float>* xRange;
    Range<float>* thetaRange;
    Range<float>* actionRange;

  public:
    DenseVector<float>* theta;
    DenseVector<float>* thetaDot;
    DenseVector<float>* thetaDotDot;
    DenseVector<float>* l; // half length of i^{th} pole
    DenseVector<float>* f; // effective force
    DenseVector<float>* m; // mass of i^{th} pole
    DenseVector<float>* mm; // effective mass
    DenseVector<float>* mup; // coefficient of friction of i^{th} pole's hinge

  public:
    NonMarkovPoleBalancing(const int& nbUnjointedPoles = 1) :
        Environment<float>(1 + nbUnjointedPoles, 3, 1), nbUnjointedPoles(nbUnjointedPoles), stepTime(
            0.01), x(0), xDot(0), g(-9.81), M(1.0), muc(0), xRange(new Range<float>(-2.4f, 2.4f)), actionRange(
            new Range<float>(-10.0f, 10.0f)), theta(new DenseVector<float>(nbUnjointedPoles)), thetaDot(
            new DenseVector<float>(nbUnjointedPoles)), thetaDotDot(
            new DenseVector<float>(nbUnjointedPoles)), l(new DenseVector<float>(nbUnjointedPoles)), f(
            new DenseVector<float>(nbUnjointedPoles)), m(new DenseVector<float>(nbUnjointedPoles)), mm(
            new DenseVector<float>(nbUnjointedPoles)), mup(new DenseVector<float>(nbUnjointedPoles))
    {
      if (nbUnjointedPoles == 2)
      {
        thetaRange = new Range<float>(-15.0f / 180.0f * M_PI, 15.0f / 180.0f * M_PI);
        l->at(0) = 0.5;
        l->at(1) = 0.05;
        m->at(0) = 0.1;
        m->at(1) = 0.01;
        muc = 0.0005f;
        mup->at(0) = mup->at(1) = 0.000002f;
      }
      else
      {
        thetaRange = new Range<float>(-12.0f / 180.0f * M_PI, 12.0f / 180.0f * M_PI);
        l->at(0) = 0.5; // Kg
        m->at(0) = 0.1; // Kg
        muc = 0.0;
      }
      discreteActions->push_back(0, actionRange->min());
      discreteActions->push_back(1, 0.0);
      discreteActions->push_back(2, actionRange->max());

      // subject to change
      continuousActions->push_back(0, 0.0);

      for (int i = 0; i < getVars().dimension(); i++)
        resolutions->at(i) = 6.0;

    }

    ~NonMarkovPoleBalancing()
    {
      delete xRange;
      delete thetaRange;
      delete actionRange;
      delete theta;
      delete thetaDot;
      delete thetaDotDot;
      delete l;
      delete f;
      delete m;
      delete mm;
      delete mup;
    }

  private:
    void adjustTheta()
    {
      for (int i = 0; i < nbUnjointedPoles; i++)
      {
        if (theta->at(i) >= M_PI)
          theta->at(i) -= 2.0 * M_PI;
        if (theta->at(i) < -M_PI)
          theta->at(i) += 2.0 * M_PI;
      }
    }

  public:
    void updateRTStep()
    {
      DenseVector<float>& vars = *output->o_tp1;
      vars[0] = (x - xRange->min()) * resolutions->at(0) / xRange->length();
      observations->at(0) = x;
      for (int i = 0; i < nbUnjointedPoles; i++)
      {
        vars[i + 1] = (theta->at(i) - thetaRange->min()) * resolutions->at(i + 1)
            / thetaRange->length();
        observations->at(1) = theta->at(i);
      }
    }

    void initialize()
    {
      x = 0.0;
      for (int i = 0; i < nbUnjointedPoles; i++)
        theta->at(i) = 0.0f;

      adjustTheta();
      updateRTStep();
    }

    void step(const Action& a)
    {
      float totalEffectiveForce = 0;
      float totalEffectiveMass = 0;
      for (int i = 0; i < nbUnjointedPoles; i++)
      {
        float effectiveForce = 0;
        effectiveForce += m->at(i) * l->at(i) * std::pow(thetaDot->at(i), 2) * ::sin(theta->at(i));
        effectiveForce += 0.75f * m->at(i) * ::cos(theta->at(i))
            * ((mup->at(i) * theta->at(i)) / (m->at(i) * l->at(i)) + g * ::sin(theta->at(i)));
        f->at(i) = effectiveForce;
        mm->at(i) = m->at(i) * (1.0f - 0.75f * std::pow(::cos(theta->at(i)), 2));
        totalEffectiveForce += f->at(i);
        totalEffectiveMass += mm->at(i);
      }

      float torque = actionRange->bound(a.at());
      float xAcc = (torque - muc * Signum::valueOf(xDot) + totalEffectiveForce)
          / (M + totalEffectiveMass);

      // Update the four state variables, using Euler's method.
      x = xRange->bound(x + xDot * stepTime);
      xDot += xAcc * stepTime;

      for (int i = 0; i < nbUnjointedPoles; i++)
      {
        thetaDotDot->at(i) = -0.75f
            * (xAcc * ::cos(theta->at(i)) + g * ::sin(theta->at(i))
                + (mup->at(i) * theta->at(i)) / (m->at(i) * l->at(i))) / l->at(i);

        // Update the four state variables, using Euler's method.
        theta->at(i) = thetaRange->bound(theta->at(i) + thetaDot->at(i) * stepTime);
        thetaDot->at(i) += thetaDotDot->at(i) * stepTime;
      }

      adjustTheta();
      updateRTStep();
    }

    bool endOfEpisode() const
    {
      bool value = true;
      for (int i = 0; i < nbUnjointedPoles; i++)
        value *= thetaRange->in(theta->at(i));
      value = value && xRange->in(x);
      return !value;
    }

    float r() const
    {
      float value = 0;
      for (int i = 0; i < nbUnjointedPoles; i++)
        value += ::cos(theta->at(i));
      return value;
    }

    float z() const
    {
      return 0.0;
    }

};

#endif /* NONMARKOVPOLEBALANCING_H_ */
