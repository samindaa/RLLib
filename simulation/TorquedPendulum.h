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
    double m, l, mu, g, force;

  public:

    TorquedPendulum(const double& m, const double& l, const double& mu, const double& dt) :
        RK4(2, dt), m(m), l(l), mu(mu), g(9.8), force(0)
    {
    }

    virtual ~TorquedPendulum()
    {
    }

    void setForce(const double& force)
    {
      this->force = force;
    }

    void f(const double& t, const int& m, const double* u, double* u_dot)
    {
      u_dot[0] = u[1];
      u_dot[1] = ((-mu * u_dot[0] + m * g * l * sin(u[0]) + force) / (m * l * l));
    }

};

#endif /* TORQUEDPENDULUM_H_ */
