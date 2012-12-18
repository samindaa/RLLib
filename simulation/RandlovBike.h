/*
 * RandlovBike.h
 *
 *  Created on: Sep 19, 2012
 *      Author: sam
 */

#ifndef RANDLOVBIKE_H_
#define RANDLOVBIKE_H_

#include "Env.h"
#include "Math.h"
#include <iostream>
#include <cmath>

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
class RandlovBike: public Env<float>
{
  protected:
    /* position of goal */
    float x_goal, y_goal, radius_goal, reinforcement;
    bool isTerminal;

    float omega, omega_dot, omega_d_dot, theta, theta_dot, theta_d_dot, xf, yf,
        xb, yb /* tyre position */, psi_goal /* Angle to the goal */, aux_state;

    float R1, R2, R3, R_FACTOR, NO_STATES2, dt, /* 10 km/t in m/s */
    v, g, dCM, c, h, Mc, Md, Mp, M, R, /* tyre radius */
    sigma_dot, I_bike, I_dc, I_dv, I_dl, l,
    /* distance between the point where the front and back tyre touch the ground */
    pi;

  public:
    RandlovBike() :
        Env<float>(8, 9, 1), x_goal(0), y_goal(0), radius_goal(0), reinforcement(
            0), isTerminal(false), omega(0), omega_dot(0), omega_d_dot(0), theta(
            0), theta_dot(0), theta_d_dot(0), xf(0), yf(0), xb(0), yb(0), psi_goal(
            0), aux_state(0), R1(-1.0), R2(0.0), R3(+1.0),
        // +0.01
        R_FACTOR(0.0001), NO_STATES2(20),

        dt(0.01), v(10.0 / 3.6), g(9.82), dCM(0.3), c(0.66), h(0.94), Mc(15.0), Md(
            1.7), Mp(60.0), M(Mc + Mp), R(0.34),
        /* tyre radius */
        sigma_dot(v / R), I_bike(
            (13.0 / 3) * Mc * h * h + Mp * (h + dCM) * (h + dCM)), I_dc(
            Md * R * R), I_dv((3.0 / 2) * Md * R * R), I_dl(
            (1.0 / 2) * Md * R * R), l(1.11), pi(3.1415927)
    {

      for (unsigned int i = 0; i < discreteActions->dimension(); i++)
        discreteActions->push_back(i, i);
    }

  private:
    enum OperationMode
    {
      start = 0, execute_action,
    };

    float calc_dist_to_goal(float xf, float xb, float yf, float yb)
    {
      float temp = (x_goal - xf) * (x_goal - xf) + (y_goal - yf) * (y_goal - yf)
          - radius_goal * radius_goal;
      temp = sqrt(std::max(0.0f, temp));
      return (temp);
    }
    float calc_angle_to_goal(float xf, float xb, float yf, float yb)
    {
      float temp, scalar, tvaer;

      temp = (xf - xb) * (x_goal - xf) + (yf - yb) * (y_goal - yf);
      scalar = temp / (l * sqrt(pow((x_goal - xf), 2) + pow((y_goal - yf), 2)));
      tvaer = (-yf + yb) * (x_goal - xf) + (xf - xb) * (y_goal - yf);

      if (tvaer <= 0)
        temp = scalar - 1;
      else
        temp = fabs(scalar - 1);

      /* These angles are neither in degrees nor radians, but something
       strange invented in order to save CPU-time. The measure is arranged the
       same way as radians, but with a slightly different negative factor.

       Say, the goal is to the east.
       If the agent rides to the east then  temp = 0
       - " -          - " -   north              = -1
       - " -                  west               = -2 or 2
       - " -                  south              =  1 */

      return (temp);
    }
    int get_box(float theta, float theta_dot, float omega, float omega_dot,
        float omega_d_dot, float psi_goal)
    {
      int box;

      if (theta < -1)
        box = 0;
      else if (theta < -0.2)
        box = 1;
      else if (theta < 0)
        box = 2;
      else if (theta < 0.2)
        box = 3;
      else if (theta < 1)
        box = 4;
      else
        box = 5;
      /* The last restriction is taken care off in the physics part */

      if (theta_dot < -2)
      {
      }
      else if (theta_dot < 0)
        box += 6;
      else if (theta_dot < 2)
        box += 12;
      else
        box += 18;

      if (omega < -0.15)
      {
      }
      else if (omega < -0.06)
        box += 24;
      else if (omega < 0)
        box += 48;
      else if (omega < 0.06)
        box += 72;
      else if (omega < 0.15)
        box += 96;
      else
        box += 120;

      if (omega_dot < -0.45)
      {
      }
      else if (omega_dot < -0.24)
        box += 144;
      else if (omega_dot < 0)
        box += 288;
      else if (omega_dot < 0.24)
        box += 432;
      else if (omega_dot < 0.45)
        box += 576;
      else
        box += 720;

      if (omega_d_dot < -1.8)
      {
      }
      else if (omega_d_dot < 0)
        box += 864;
      else if (omega_d_dot < 1.8)
        box += 1728;
      else
        box += 2592;

      return (box);
    }

  public:

    void update()
    {
      DenseVector<float>& vars = *__vars;
      vars[0] = omega;
      vars[1] = omega_dot;
      vars[2] = omega_d_dot;
      vars[3] = theta;
      vars[4] = theta_dot;
      vars[5] = theta_d_dot;
      vars[6] = psi_goal;
      vars[7] = calc_dist_to_goal(xf, xb, yf, yb);

      // @@>> TODO: this needs a some work
      static float OMEGA = 2.0 * pi;
      static float OMEGA_DOT = 1.;
      static float OMEGA_DOT_DOT = 1.;
      static float THETA = pi / 4.;
      static float THETA_DOT = 1.;
      static float PSI_GOAL = 1.;
      static float AUX_STATE = 1000;

      vars[0] /= OMEGA;
      vars[1] /= OMEGA_DOT;
      vars[2] /= OMEGA_DOT_DOT;
      vars[3] /= THETA;
      vars[4] /= THETA_DOT;
      vars[5] /= PSI_GOAL;
      vars[6] /= AUX_STATE;

      // debug
      //for (int i = 0; i < env->getNumStateVars(); i++)
      //  cout << percepts[i] << " ";
      //cout << endl;
    }

    void bike(int to_do, int action = 0)
    {
      static double rCM, rf, rb;
      static float T, d, phi, psi /* bike's angle to the y-axis */;
      float temp;

      T = 2 * ((action / 3) - 1);
      d = 0.02 * ((action % 3) - 1);
      d = d + 0.04 * (0.5 - Random::nextDouble()); /* Max noise is 2 cm */

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
        psi_goal = calc_angle_to_goal(xf, xb, yf, yb);
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
                        * (Md * R * (1.0 / rf + 1.0 / rb) + M * h / rCM)))
            / I_bike;
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
        temp = v * dt / (2 * rf);
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

        psi_goal = calc_angle_to_goal(xf, xb, yf, yb);

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
        temp = calc_dist_to_goal(xf, xb, yf, yb);

        if (temp < 1e-3)
        {
          reinforcement = R3;
          isTerminal = true;
          std::cout << " #{goal} ";
          std::cout.flush();
        }
        else
        {
          reinforcement = (0.95 - pow(psi_goal, 2)) * R_FACTOR;
          // in order to make the agent to learn
          // reinforcement = R2;
          isTerminal = false;
        }
      }

      /* There are two sorts of state information. The first (*return_state) is
       about the state of the bike, while the second (*return_state2) deals with the
       position relative to the goal */
      /**return_state = get_box(theta, theta_dot, omega, omega_dot, omega_d_dot,
       psi_goal);*/

      int i = 0;
      aux_state = -1;
      while (aux_state < 0)
      {
        temp = -2 + ((float) (4 * (i))) / NO_STATES2;
        if (psi_goal < temp)
          aux_state = i;
        i++;
      }

      update();
    }

    void initialize()
    {
      /* position of goal */
      x_goal = 1000;
      y_goal = 0;
      radius_goal = 10;
      bike(start);
    }

    void step(const Action& action)
    {
      bike(execute_action, action);
    }

    bool endOfEpisode() const
    {
      return isTerminal;
    }
    float r() const
    {
      return reinforcement;
    }
    float z() const
    {
      return reinforcement;
    }

};

#endif /* RANDLOVBIKE_H_ */
