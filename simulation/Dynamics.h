/*
 * Dynamics.h
 *
 *  Created on: Oct 16, 2012
 *      Author: sam
 */

#ifndef DYNAMICS_H_
#define DYNAMICS_H_

#include "Vector.h"

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
        dimensions(dimensions), dt(dt), t(0), dxdt(new DenseVector<double>(dimensions)), x(
            new DenseVector<double>(dimensions)), xout(new DenseVector<double>(dimensions)), dxm(
            new DenseVector<double>(dimensions)), dxt(new DenseVector<double>(dimensions)), xt(
            new DenseVector<double>(dimensions))
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
