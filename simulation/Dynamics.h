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
 * Dynamics.h
 *
 *  Created on: Oct 16, 2012
 *      Author: sam
 */

#ifndef DYNAMICS_H_
#define DYNAMICS_H_

#include "Vector.h"

using namespace RLLib;

class Dynamics
{
  protected:
    int dimensions;
    double dt;
    double t;
    DenseVector<double>* dxdt;
    DenseVector<double>* x;
    DenseVector<double>* xout;
    DenseVector<double>* dxm;
    DenseVector<double>* dxt;
    DenseVector<double>* xt;

  public:
    Dynamics(const int& dimensions, const double& dt) :
        dimensions(dimensions), dt(dt), t(0), dxdt(new PVector<double>(dimensions)), x(
            new PVector<double>(dimensions)), xout(new PVector<double>(dimensions)), dxm(
            new PVector<double>(dimensions)), dxt(new PVector<double>(dimensions)), xt(
            new PVector<double>(dimensions))
    {
    }
    virtual ~Dynamics()
    {
      delete dxdt;
      delete x;
      delete xout;
      delete dxm;
      delete dxt;
      delete xt;
    }

    void sett(const double& t)
    {
      this->t = t;
    }
    double gett() const
    {
      return t;
    }
    double getx(const int& index) const
    {
      return x->at(index);
    }
    void setx(const int& index, const double& value)
    {
      x->at(index) = value;
    }

    virtual void derivs(const double& a, DenseVector<double>* b, DenseVector<double>* c) =0;

    void rk4a()
    {
      double d2 = dt * 0.5;
      double d3 = dt / 6.0;
      double d1 = t + d2;

      for (int i = 0; i < dimensions; i++)
        xt->at(i) = (x->at(i) + d2 * dxdt->at(i));

      derivs(d1, xt, dxt);

      for (int i = 0; i < dimensions; i++)
        xt->at(i) = (x->at(i) + d2 * dxt->at(i));
      derivs(d1, xt, dxm);

      for (int i = 0; i < dimensions; i++)
      {
        xt->at(i) = (x->at(i) + dt * dxm->at(i));
        dxm->at(i) += dxt->at(i);
      }

      derivs(t + dt, xt, dxt);

      for (int i = 0; i < dimensions; i++)
        xout->at(i) = (x->at(i) + d3 * (dxdt->at(i) + dxt->at(i) + 2.0 * dxm->at(i)));
    }

    void nextstep()
    {
      derivs(t, x, dxdt);
      rk4a();
    }

    virtual void nextcopy()
    {
      for (int i = 0; i < dimensions; i++)
        x->at(i) = xout->at(i);
      t += dt;
    }
};

#endif /* DYNAMICS_H_ */
