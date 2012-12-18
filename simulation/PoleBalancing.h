/*
 * PoleBalancing.h
 *
 *  Created on: Oct 8, 2012
 *      Author: sam
 */

#ifndef POLEBALANCING_H_
#define POLEBALANCING_H_

#include "Env.h"
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"
#include "Math.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::MatrixBase;

// MultivariateNormal random number generator

/*
 * Multivariate normal random number. Based on Matlab mvnrnd function.
 */
template<class Derived1, class Derived2, class Derived3>
void mvnrnd(const MatrixBase<Derived1>& mu, const MatrixBase<Derived2>& Sigma,
    MatrixBase<Derived3>& r)
{
  MatrixXd R = Sigma.llt().matrixL()/*cholcov*/;
  int size = mu.rows();
  for (int i = 0; i < size; i++)
    r(i) = RLLib::Random::nextNormalGaussian();
  r = mu + R * r;
}

class PoleBalancing: public Env<float>
{
  protected:
    double tau, veta, g;
    MatrixXd Sigma0;
    VectorXd mu0;
    VectorXd mu;
    VectorXd x;
    MatrixXd A;
    VectorXd b;
    MatrixXd Q;
    MatrixXd SigmaT;

    VectorXd u;/*only single input is considered*/
    VectorXd R;

  public:
    PoleBalancing() :
        Env<float>(4, 1, 1), tau(1.0 / 60.0), veta(13.2), g(9.81)
    {
      // No discrete actions

      // Only continuous actions
      continuousActions->push_back(0, 0.0);

      // Set up the matrixes
      VectorXd sigma0(4);
      mu0.resize(4);
      mu.resize(4);
      x.resize(4);
      for (int i = 0; i < sigma0.rows(); i++)
      {
        sigma0(i) = 0.1;
        x(i) = mu0(i) = mu(i) = 0;
      }
      Sigma0 = sigma0.asDiagonal();

      A.resize(4, 4);
      A << 1, tau, 0, 0, 0, 1, 0, 0, 0, 0, 1, tau, 0, 0, veta * tau, 1;

      b.resize(4);
      b << 0, tau, 0, veta * tau / g;

      VectorXd q(4);
      q << 1.25, 1, 12, 0.25;
      Q = q.asDiagonal();

      SigmaT = 0.01 * Sigma0;

      u.resize(1);
      u(0) = 0;
      R.resize(1);
      R(0) = 0.01;

    }
    virtual ~PoleBalancing()
    {
    }

    void initialize()
    {
      mvnrnd(mu0, Sigma0, x);
      update();
    }
    void update()
    {
      DenseVector<float>& vars = *__vars;
      for (int i = 0; i < vars.dimension(); i++)
        vars[i] = x[i];
    }
    void step(const Action& action)
    {
      u(0) = action.at(0);
      mu = A * x + b * u;
      mvnrnd(mu, SigmaT, x);
      update();
    }
    bool endOfEpisode() const
    {
      return fabs((double) x(0)) >= 1.5 || fabs((double) x(2)) >= M_PI / 6.0;
    }

    float r() const
    {
      VectorXd r_xt_ut = x.transpose() * Q * x + u.transpose() * R * u;
      return (float) r_xt_ut(0);
    }

    float z() const
    {
      return 0.0;
    }
};

#endif /* POLEBALANCING_H_ */
