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
 * RK4.h
 *
 *  Created on: Aug 14, 2014
 *      Author: sam
 */

#ifndef RK4_H_
#define RK4_H_

#include <algorithm>

// Runge-Kutta 4th Order ODE Solver
class RK4
{
  private:
    // int M, the spatial dimension.
    int m;
    double dt;
    double* data;

    // Temporary variables
    double *f0;
    double *f1;
    double *f2;
    double *f3;

    double *u1;
    double *u2;
    double *u3;
    double *u0;

    double t0;
    int timeSteps;

  public:
    RK4(const int& m, const double& dt) :
        m(m), dt(dt), data(new double[m * 8])
    {
      f0 = data + 0 * m;
      f1 = data + 1 * m;
      f2 = data + 2 * m;
      f3 = data + 3 * m;

      u1 = data + 4 * m;
      u2 = data + 5 * m;
      u3 = data + 6 * m;
      u0 = data + 7 * m;

      initialize();
    }

    virtual ~RK4()
    {
      delete data;
    }

    void initialize()
    {
      std::fill_n(data, m * 8, 0);
      t0 = 0;
      timeSteps = 0;
    }

    double& operator()(const int& i)
    {
      return *(u0 + i);
    }

    const double& operator()(const int& i) const
    {
      return *(u0 + i);
    }

    double& at(const int& i)
    {
      return *(u0 + i);
    }

    const double& at(const int& i) const
    {
      return *(u0 + i);
    }

    int size() const
    {
      return m;
    }

    double getTime() const
    {
      return t0;
    }

    int getTimeSteps() const
    {
      return timeSteps;
    }

    /**
     * STEP takes one Runge-Kutta step for a vector ODE.
     *
     * Problem is given as
     * x_dot = f(x, t)
     * x(t_0) = x_0;
     *
     * If the user can supply current values of t, x, dt, and a
     * function to evaluate the derivative, this function can compute the
     * fourth-order Runge Kutta estimate to the solution at time t+dt.
     *
     *  Inputs:
     *    double T0, the current time.
     *    double U0[M], the solution estimate at the current time.
     *    double DT, the time step.
     */
    void step()
    {
      //
      //  Get four sample values of the derivative.
      //
      f(t0, m, u0, f0);

      double t1 = t0 + dt / 2.0;

      for (int i = 0; i < m; i++)
        u1[i] = u0[i] + dt * f0[i] / 2.0;

      f(t1, m, u1, f1);

      double t2 = t0 + dt / 2.0;

      for (int i = 0; i < m; i++)
        u2[i] = u0[i] + dt * f1[i] / 2.0;

      f(t2, m, u2, f2);

      double t3 = t0 + dt;

      for (int i = 0; i < m; i++)
        u3[i] = u0[i] + dt * f2[i];

      f(t3, m, u3, f3);

      //
      //  Combine them to estimate the solution.
      //
      for (int i = 0; i < m; i++)
        u0[i] += dt * (f0[i] + 2.0 * f1[i] + 2.0 * f2[i] + f3[i]) / 6.0;

      // update time
      t0 += dt;
      ++timeSteps;
    }

    /**
     * This method evaluates the derivative, or right hand side of the problem.
     *
     * Output is the fourth-order Runge-Kutta solution estimate at time T0+DT.
     */
    virtual void f(const double& t, const int& m, const double* u, double* u_dot) =0;
};

#endif /* RK4_H_ */
