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
 * RandlovBike.h
 *
 *  Created on: Sep 19, 2012
 *      Author: sam
 */

#ifndef RANDLOVBIKE_H_
#define RANDLOVBIKE_H_

#include <iostream>
#include "RL.h"
#include "Mathema.h"

/**
 * Reference:
 * @InProceedings{Randlov+Alstrom:1998,
 * author =       "Randl\ov, Jette and Alstr\om, Preben",
 * title =        "Learning to Drive a Bicycle Using Reinforcement Learning and Shaping",
 * booktitle =    "Proceedings of the Fifteenth International Conference on Machine Learning (ICML 1998)",
 * year =         "1998",
 * ISBN =         "1-55860-556-8",
 * editor =    "Shavlik, Jude W.",
 * publisher = "Morgan Kauffman",
 * address =   "San Francisco, CA, USA",
 * pages =     "463--471",
 * url = "http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.52.3038&rep=rep1&type=pdf",
 * bib2html_rescat = "Applications, General RL",
 }
 */
template<typename Type>
class RandlovBike: public RLProblem<Type>
{
    typedef RLProblem<Type> Base;
  protected:
    /*is go-to-target behavior*/
    bool goToTarget;
    /* position of goal */
    float x_goal, y_goal, radius_goal, reinforcement;
    bool isTerminal;

    float omega, omega_dot, omega_d_dot, theta, theta_dot, theta_d_dot, xf, yf, xb,
        yb /* tyre position */, psi_goal /* Angle to the goal */;
    double rCM, rf, rb;
    float T, d, phi, psi /* bike's angle to the y-axis */;

    float R1, R2, R3, R_FACTOR, dt, /* 10 km/t in m/s */
    v, g, dCM, c, h, Mc, Md, Mp, M, R, /* tyre radius */
    sigma_dot, I_bike, I_dc, I_dv, I_dl, l,
    /* distance between the point where the front and back tyre touch the ground */
    pi, Gamma;

    /*ranges*/
    Range<float> *thetaRange, *thetaDotRange, *omegaRange, *omegaDotRange, *omegaDotDotRange,
        *psiRange;

  public:
    RandlovBike(Random<Type>* random, const bool& goToTarget) :
        RLProblem<Type>(random, (goToTarget ? 1 : 0) + 5, 9, 0), goToTarget(goToTarget), //
        x_goal(1000.0f), y_goal(0), radius_goal(10.0f), reinforcement(0), isTerminal(false), //
        omega(0), omega_dot(0), omega_d_dot(0), theta(0), theta_dot(0), theta_d_dot(0), xf(0), //
        yf(0), xb(0), yb(0), psi_goal(0), rCM(0), rf(0), rb(0), T(0), d(0), phi(0), psi(0), //
        R1(-1.0), R2(0.0), R3(+1.0), R_FACTOR(0.00001), dt(0.01), v(10.0 / 3.6), g(9.82), dCM(0.3), //
        c(0.66), h(0.94), Mc(15.0), Md(1.7), Mp(60.0), M(Mc + Mp), R(0.34),
        /* tyre radius */
        sigma_dot(v / R), I_bike((13.0 / 3) * Mc * h * h + Mp * (h + dCM) * (h + dCM)), //
        I_dc(Md * R * R), I_dv((3.0 / 2) * Md * R * R), I_dl((1.0 / 2) * Md * R * R), l(1.11), //
        pi( M_PI), Gamma(0.99), thetaRange(new Range<float>(-M_PI_2, M_PI_2)), //
        thetaDotRange(new Range<float>(-2, 2)), //
        omegaRange(new Range<float>(-M_PI / 15.0f, M_PI / 15.0f)), //
        omegaDotRange(new Range<float>(-0.5, 0.5)), omegaDotDotRange(new Range<float>(-2, 2)), //
        psiRange(new Range<float>(-M_PI, M_PI))
    {

      for (int i = 0; i < Base::discreteActions->dimension(); i++)
        Base::discreteActions->push_back(i, i);
    }

    virtual ~RandlovBike()
    {
      delete thetaRange;
      delete thetaDotRange;
      delete omegaRange;
      delete omegaDotRange;
      delete omegaDotDotRange;
      delete psiRange;
    }

  private:
    enum OperationMode
    {
      start = 0, execute_action,
    };

    float calcDistToGoal(float xf, float xb, float yf, float yb)
    {
      float temp = (x_goal - xf) * (x_goal - xf) + (y_goal - yf) * (y_goal - yf)
          - radius_goal * radius_goal;
      temp = sqrt(std::max(0.0f, temp));
      return (temp);
    }

    float calcAngleToGoal(float xf, float xb, float yf, float yb)
    {
      // Signed angle [-pi, pi]
      float angle = std::atan2(yf - yb, xf - xb) - std::atan2(y_goal - yf, x_goal - xf);
      if (angle > M_PI)
        angle -= 2.0f * M_PI;
      else if (angle < -M_PI)
        angle += 2.0f * M_PI;
      return angle;
    }

  public:
    void updateTRStep()
    {
      Base::output->o_tp1->setEntry(0, omegaRange->toUnit(omega));
      Base::output->o_tp1->setEntry(1, omegaDotRange->toUnit(omega_dot));
      Base::output->o_tp1->setEntry(2, omegaDotDotRange->toUnit(omega_d_dot));
      Base::output->o_tp1->setEntry(3, thetaRange->toUnit(theta));
      Base::output->o_tp1->setEntry(4, thetaDotRange->toUnit(theta_dot));
      if (goToTarget)
        Base::output->o_tp1->setEntry(5, psiRange->toUnit(psi_goal));
      Base::output->observation_tp1->setEntry(0, omega);
      Base::output->observation_tp1->setEntry(1, omega_dot);
      Base::output->observation_tp1->setEntry(2, omega_d_dot);
      Base::output->observation_tp1->setEntry(3, theta);
      Base::output->observation_tp1->setEntry(4, theta_dot);
      if (goToTarget)
        Base::output->observation_tp1->setEntry(5, psi_goal);
    }

    void bike(const int& to_do, const int& action = 0)
    {

      T = 2 * ((action / 3) - 1);
      d = 0.02 * ((action % 3) - 1);
      d = d + 0.04 * (0.5 - Base::random->nextReal()); /* Max noise is 2 cm */

      switch (to_do)
      {
        case start:
        {
          omega = omega_dot = omega_d_dot = 0;
          theta = theta_dot = theta_d_dot = 0;
          xb = 0;
          yb = 0;
          xf = 0;
          yf = l;
          psi = atan((xb - xf) / (yf - yb));
          psi_goal = calcAngleToGoal(xf, xb, yf, yb);
          isTerminal = false;
          break;
        }

        case execute_action:
        {

          if (theta == 0)
          {
            rCM = rf = rb = 9999999; /* just a large number */
          }
          else
          {
            rCM = sqrt(pow(l - c, 2) + l * l / (pow(tan(theta), 2)));
            rf = l / fabs(sin(theta));
            rb = l / fabs(tan(theta));
          } /* rCM, rf and rb are always positiv */

          /* Main physics eq. in the bicycle model coming here: */
          phi = omega + atan(d / h);
          omega_d_dot = (h * M * g * sin(phi)
              - cos(phi)
                  * (I_dc * sigma_dot * theta_dot
                      + Signum::valueOf(theta) * v * v
                          * (Md * R * (1.0 / rf + 1.0 / rb) + M * h / rCM))) / I_bike;
          theta_d_dot = (T - I_dv * omega_dot * sigma_dot) / I_dl;

          /*--- Eulers method ---*/
          omega_dot += omega_d_dot * dt;
          omega += omega_dot * dt;
          theta_dot += theta_d_dot * dt;
          theta += theta_dot * dt;

          if (fabs(theta) > 1.3963)
          { /* handlebars cannot turn more than
           80 degrees */
            theta = Signum::valueOf(theta) * 1.3963;
          }

          /* New position of front tyre */
          float temp = v * dt / (2 * rf);
          if (temp > 1)
            temp = Signum::valueOf(psi + theta) * pi / 2;
          else
            temp = Signum::valueOf(psi + theta) * asin(temp);
          xf += v * dt * (-sin(psi + theta + temp));
          yf += v * dt * cos(psi + theta + temp);

          /* New position of back tyre */
          temp = v * dt / (2 * rb);
          if (temp > 1)
            temp = Signum::valueOf(psi) * pi / 2;
          else
            temp = Signum::valueOf(psi) * asin(temp);
          xb += v * dt * (-sin(psi + temp));
          yb += v * dt * (cos(psi + temp));

          /* Round off errors accumulate so the length of the bike changes over many
           iterations. The following take care of that: */
          temp = sqrt((xf - xb) * (xf - xb) + (yf - yb) * (yf - yb));
          if (fabs(temp - l) > 0.01)
          {
            xb += (xb - xf) * (l - temp) / temp;
            yb += (yb - yf) * (l - temp) / temp;
          }

          temp = yf - yb;
          if ((xf == xb) && (temp < 0))
            psi = pi;
          else
          {
            if (temp > 0)
              psi = atan((xb - xf) / temp);
            else
              psi = Signum::valueOf(xb - xf) * (pi / 2) - atan(temp / (xb - xf));
          }

          psi_goal = calcAngleToGoal(xf, xb, yf, yb);

          break;
        }
      }

      /*-- Calculation of the reinforcement  signal --*/
      if (fabs(omega) > (pi / 15))
      { /* the bike has fallen over */
        reinforcement = R1;
        /* a good place to print some info to a file or the screen */
        isTerminal = true;
      }
      else
      {
        if (calcDistToGoal(xf, xb, yf, yb) < 1e-3)
        {
          reinforcement = R3;
          isTerminal = true;
          std::cout << " #{goal} ";
          std::cout.flush();
        }
        else
        {
          if (goToTarget)
            reinforcement = (4.0f - std::pow(psi_goal, 2)) * R_FACTOR; // << (reward shaping)
          else
            reinforcement = R2; //<< to Balance
          isTerminal = false;
        }
      }
    }

    void initialize()
    {
      bike(start);
    }

    void step(const Action<double>* action)
    {
      bike(execute_action, action->id());
    }

    bool endOfEpisode() const
    {
      return isTerminal;
    }
    Type r() const
    {
      return reinforcement;
    }
    Type z() const
    {
      return reinforcement;
    }

    float getGamma() const
    {
      return Gamma;
    }

};

#endif /* RANDLOVBIKE_H_ */
