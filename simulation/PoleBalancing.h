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
 * PoleBalancing.h
 *
 *  Created on: Oct 8, 2012
 *      Author: sam
 */

#ifndef POLEBALANCING_H_
#define POLEBALANCING_H_

#include "RL.h"
#include "util/Eigen/Dense"
#include "util/Eigen/Eigenvalues"
#include "Mathema.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::MatrixBase;

// MultivariateNormal random number generator

/*
 * Multivariate normal random number. Based on Matlab mvnrnd function.
 */
template<class Derived1, class Derived2, class Derived3>
void mvnrnd(Random<double>* random, const MatrixBase<Derived1>& mu,
    const MatrixBase<Derived2>& Sigma, MatrixBase<Derived3>& r)
{
  MatrixXd R = Sigma.llt().matrixL()/*cholcov*/;
  int size = mu.rows();
  for (int i = 0; i < size; i++)
    r(i) = random->nextNormalGaussian();
  r = mu + R * r;
}

class PoleBalancing: public RLProblem<double>
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
    PoleBalancing(Random<double>* random) :
        RLProblem<double>(random, 4, 1, 1), tau(1.0 / 60.0), veta(13.2), g(9.81)
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
      mvnrnd(random, mu0, Sigma0, x);
    }

    void updateTRStep()
    {
      for (int i = 0; i < output->observation_tp1->dimension(); i++)
      {
        output->observation_tp1->setEntry(i, double(x[i]));
        output->o_tp1->setEntry(i, output->observation_tp1->getEntry(i));
      }
    }

    void step(const Action<double>* action)
    {
      u(0) = action->getEntry(0);
      mu = A * x + b * u;
      mvnrnd(random, mu, SigmaT, x);
    }

    bool endOfEpisode() const
    {
      return fabs((double) x(0)) >= 1.5 || fabs((double) x(2)) >= M_PI / 6.0;
    }

    double r() const
    {
      VectorXd r_xt_ut = x.transpose() * Q * x + u.transpose() * R * u;
      return (float) r_xt_ut(0);
    }

    double z() const
    {
      return 0.0;
    }
};

#endif /* POLEBALANCING_H_ */
