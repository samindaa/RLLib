/*
 * Copyright 2013 Saminda Abeyruwan (saminda@cs.miami.edu)
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

#include "Environment.h"
#include "Math.h"
#include "Matrix.h"

// MultivariateNormal random number generator

/*
 * Multivariate normal random number. Based on Matlab mvnrnd function.
 */
void mvnrnd(const Matrix& mu, const Matrix& Sigma, Matrix& r)
{
  Matrix U, D;
  Sigma.eig(U, D);
  for (unsigned int i = 0; i < mu.rows(); i++)
    r(i) = RLLib::Probabilistic::nextNormalGaussian() * ::sqrt(D(i));
  r = mu + U * r;
}

class PoleBalancing: public Environment<float>
{
  protected:
    double tau, veta, g;
    Matrix Sigma0;
    Matrix mu0;
    Matrix mu;
    Matrix x;
    Matrix A;
    Matrix b;
    Matrix Q;
    Matrix SigmaT;

    Matrix u;/*only single input is considered*/
    Matrix R;

  public:
    PoleBalancing() :
        Environment<float>(4, 1, 1), tau(1.0 / 60.0), veta(13.2), g(9.81)
    {
      // No discrete actions

      // Only continuous actions
      continuousActions->push_back(0, 0.0);

      // Set up the matrixes
      Matrix sigma0(4);
      mu0.resize(4);
      mu.resize(4);
      x.resize(4);
      for (unsigned int i = 0; i < sigma0.rows(); i++)
      {
        sigma0(i) = 0.1;
        x(i) = mu0(i) = mu(i) = 0;
      }
      Sigma0 = sigma0.diag();

      A.resize(4, 4);
      A.insert(1.f, tau, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, tau, 0.f, 0.f, veta * tau,
          1.f);

      b.resize(4);
      b.insert(0.f, tau, 0.f, veta * tau / g);

      Matrix q(4);
      q.insert(1.25f, 1.f, 12.f, 0.25f);
      Q = q.diag();

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
      updateRTStep();
    }
    void updateRTStep()
    {
      DenseVector<float>& vars = *output->o_tp1;
      for (int i = 0; i < vars.dimension(); i++)
        vars[i] = x[i];
      output->updateRTStep(r(), z(), endOfEpisode());
    }
    void step(const Action& action)
    {
      u(0) = action.at(0);
      mu = A * x + b * u;
      mvnrnd(mu, SigmaT, x);
      updateRTStep();
    }
    bool endOfEpisode() const
    {
      return fabs((double) x(0)) >= 1.5 || fabs((double) x(2)) >= M_PI / 6.0;
    }

    float r() const
    {
      Matrix r_xt_ut = x.T() * Q * x + u.T() * R * u;
      return (float) r_xt_ut(0);
    }

    float z() const
    {
      return 0.0;
    }
};

#endif /* POLEBALANCING_H_ */
