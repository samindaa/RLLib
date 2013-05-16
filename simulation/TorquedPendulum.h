/*
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
