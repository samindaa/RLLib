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
#include "Vector.h"
#include "Action.h"

// Runge-Kutta 4th Order ODE Solver for RL problems
class RK4
{
  private:
    Vector<double>* state;
    VectorPool<double>* pool;

    double timeIncrement;
    double time;
    int timeSteps;

  public:
    RK4(const int& stateVariables, const double& timeIncrement) :
        state(new PVector<double>(stateVariables)), pool(
            new VectorPool<double>(state->dimension())), timeIncrement(timeIncrement)
    {
      initialize();
    }

    virtual ~RK4()
    {
      delete state;
      delete pool;
    }

    void initialize()
    {
      time = 0;
      timeSteps = 0;
    }

    Vector<double>* vec()
    {
      return state;
    }

    double getTime() const
    {
      return time;
    }

    int getTimeSteps() const
    {
      return timeSteps;
    }

    /**
     * STEP takes one Runge-Kutta step for a vector ODE.
     *
     * Problem is given as
     * x_dot = f(x, action, t)
     * x(t_0) = x_0;
     *
     */
    void step(const Action<double>* action = 0)
    {
      //
      //  Get four sample values of the derivative.
      //
      Vector<double>* f0 = pool->newVector(state);
      f(time, action, state, f0);

      Vector<double>* u1 = pool->newVector(state);
      u1->addToSelf(timeIncrement / 2.0f, f0);

      Vector<double>* f1 = pool->newVector(u1);
      f(time + timeIncrement / 2.0f, action, u1, f1);

      Vector<double>* u2 = pool->newVector(state);
      u2->addToSelf(timeIncrement / 2.0f, f1);

      Vector<double>* f2 = pool->newVector(u2);
      f(time + timeIncrement / 2.0f, action, u2, f2);

      Vector<double>* u3 = pool->newVector(state);
      u3->addToSelf(timeIncrement, f2);

      Vector<double>* f3 = pool->newVector(u3);
      f(time + timeIncrement, action, u3, f3);

      //
      //  Combine them to estimate the solution.
      //
      state->addToSelf(timeIncrement / 6.0f, f0)->addToSelf(timeIncrement / 3.0f, f1)->addToSelf(
          timeIncrement / 3.0f, f2)->addToSelf(timeIncrement / 6.0f, f3);

      // Release pool
      pool->releaseAll();

      // update time
      time += timeIncrement;
      ++timeSteps;
    }

    /**
     * This method evaluates the derivative, or right hand side of the problem.
     *
     * Output is the fourth-order Runge-Kutta solution estimate at time TIME + DT.
     */
    virtual void f(const double& time, const Action<double>* action, const Vector<double>* x,
        Vector<double>* x_dot) =0;
};

#endif /* RK4_H_ */
