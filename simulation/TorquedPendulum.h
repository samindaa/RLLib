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
#include "Dynamics.h"

class TorquedPendulum: public Dynamics
{
  protected:
    double m, l, mu, g, u;

  public:

    TorquedPendulum(const double& m, const double& l, const double& mu, const double& dt) :
        Dynamics(2, dt), m(m), l(l), mu(mu), g(9.8), u(0)
    {
    }

    virtual ~TorquedPendulum()
    {
    }

    void setu(const double& u)
    {
      this->u = u;
    }

    void derivs(const double& a, DenseVector<double>* o, DenseVector<double>* n)
    {
      n->at(0) = o->at(1);
      n->at(1) = ((-mu * o->at(1) + m * g * l * sin(o->at(0)) + u) / (m * l * l));
    }

    void nextcopy()
    {
      for (int i = 0; i < dimensions; i++)
        x->at(i) = xout->at(i);

      // Normalize
      while (x->at(0) > M_PI)
        x->at(0) -= 2.0 * M_PI;
      while (x->at(0) < -M_PI)
        x->at(0) += 2.0 * M_PI;

      t += dt;
    }

};

#endif /* TORQUEDPENDULUM_H_ */
