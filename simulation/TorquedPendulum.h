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
 * TorquedPendulum.h
 *
 *  Created on: Oct 16, 2012
 *      Author: sam
 */

#ifndef TORQUEDPENDULUM_H_
#define TORQUEDPENDULUM_H_

#include <cmath>
#include "util/RK4.h"

class TorquedPendulum: public RK4
{
  protected:
    double m, l, mu, g;

  public:

    TorquedPendulum(const double& m, const double& l, const double& mu, const double& dt) :
        RK4(2, dt), m(m), l(l), mu(mu), g(9.8)
    {
    }

    virtual ~TorquedPendulum()
    {
    }

    void f(const double& time, const Action<double>* action, const Vector<double>* x,
        Vector<double>* x_dot)
    {
      x_dot->setEntry(0, x->getEntry(1));
      x_dot->setEntry(1,
          (-mu * x_dot->getEntry(0) + m * g * l * sin(x->getEntry(0)) + action->getEntry(0))
              / (m * l * l));
    }

};

#endif /* TORQUEDPENDULUM_H_ */
