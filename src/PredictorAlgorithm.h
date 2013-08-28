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
 * PredictorAlgorithm.h
 *
 *  Created on: Aug 25, 2012
 *      Author: sam
 */

#ifndef PREDICTORALGORITHM_H_
#define PREDICTORALGORITHM_H_

#include "Predictor.h"
#include "Vector.h"
#include "Trace.h"

namespace RLLib
{

// Simple predictor algorithms
template<class T>
class TD: public OnPolicyTD<T>
{
  protected:
    double delta_t;
    bool initialized;
    double alpha_v, gamma;
    SparseVector<T>* v;
  public:
    TD(const double& alpha_v, const double& gamma, const int& nbFeatures) :
        delta_t(0), initialized(false), alpha_v(alpha_v), gamma(gamma), v(
            new SparseVector<T>(nbFeatures))
    {
    }
    virtual ~TD()
    {
      delete v;
    }

  public:

    double initialize()
    {
      initialized = true;
      delta_t = 0;
      return delta_t;
    }

    virtual double update(const SparseVector<T>& x_t, const SparseVector<T>& x_tp1,
        const double& r_tp1, const double& gamma_tp1)
    {
      assert(initialized);
      delta_t = r_tp1 + gamma_tp1 * v->dot(x_tp1) - v->dot(x_t);
      v->addToSelf(alpha_v * delta_t, x_t);
      return delta_t;
    }

    double update(const SparseVector<T>& x_t, const SparseVector<T>& x_tp1, double r_tp1)
    {
      assert(initialized);
      return update(x_t, x_tp1, r_tp1, gamma);
    }

    void reset()
    {
      v->clear();
    }

    int dimension() const
    {
      return v->dimension();
    }

    double predict(const SparseVector<T>& phi) const
    {
      return v->dot(phi);
    }

    void persist(const std::string& f) const
    {
      v->persist(f);
    }

    void resurrect(const std::string& f)
    {
      v->resurrect(f);
    }

};

template<class T>
class TDLambda: public TD<T>
{
  protected:
    typedef TD<T> super;
    double lambda;
    Trace<T>* e;
  public:
    TDLambda(const double& alpha, const double& gamma, const double& lambda, Trace<T>* e) :
        TD<T>(alpha, gamma, e->vect().dimension()), lambda(lambda), e(e)
    {
    }
    virtual ~TDLambda()
    {
    }

  public:

    double initialize()
    {
      super::initialize();
      e->clear();
      return super::delta_t;
    }

    double update(const SparseVector<T>& x_t, const SparseVector<T>& x_tp1, const double& r_tp1,
        const double& gamma_tp1)
    {
      assert(super::initialized);

      super::delta_t = r_tp1 + gamma_tp1 * super::v->dot(x_tp1) - super::v->dot(x_t);
      e->update(lambda * gamma_tp1, x_t);
      super::v->addToSelf(super::alpha_v * super::delta_t, e->vect());
      return super::delta_t;
    }

    void reset()
    {
      super::reset();
      e->clear();
    }
};

template<class T>
class Sarsa: public Predictor<T>
{
  protected:
    double v_t, v_tp1, delta; // temporary variables
    bool initialized;
    double alpha, gamma, lambda;
    Trace<T>* e;
    SparseVector<T>* v;
  public:
    Sarsa(const double& alpha, const double& gamma, const double& lambda, Trace<T>* e) :
        v_t(0), v_tp1(0), delta(0), initialized(false), alpha(alpha), gamma(gamma), lambda(lambda), e(
            e), v(new SparseVector<T>(e->vect().dimension()))
    {
    }
    virtual ~Sarsa()
    {
      delete v;
    }

  public:

    double initialize()
    {
      e->clear();
      initialized = true;
      return 0.0;
    }

    virtual void updateV()
    {

    }

    virtual void updateAlpha(const SparseVector<T>& phi_t, const SparseVector<T>& phi_tp1,
        double r_tp1)
    {/*Default is for fixed step-size*/
    }

    double update(const SparseVector<T>& phi_t, const SparseVector<T>& phi_tp1, double r_tp1)
    {
      assert(initialized);

      updateAlpha(phi_t, phi_tp1, r_tp1);

      v_t = v->dot(phi_t);
      v_tp1 = v->dot(phi_tp1);

      e->update(gamma * lambda, phi_t);
      delta = r_tp1 + gamma * v_tp1 - v_t;

      v->addToSelf(alpha * delta, e->vect());
      return delta;
    }

    void reset()
    {
      e->clear();
      v->clear();
      initialized = false;
    }

    int dimension() const
    {
      return v->dimension();
    }

    double predict(const SparseVector<T>& phi_sa) const
    {
      return v->dot(phi_sa);
    }

    void persist(const std::string& f) const
    {
      v->persist(f);
    }

    void resurrect(const std::string& f)
    {
      v->resurrect(f);
    }
};

template<class T>
class SarsaAdaptive: public Sarsa<T>
{
  private:
    typedef Sarsa<T> super;
    double minUpdate;
    SparseVector<T>* gammaPhi_tp1MinusPhi_t;
  public:
    SarsaAdaptive(const double& gamma, const double& lambda, Trace<T>* e) :
        Sarsa<T>(1.0f/*According to the paper*/, gamma, lambda, e), minUpdate(1e-10), gammaPhi_tp1MinusPhi_t(
            new SparseVector<T>(e->vect().dimension()))
    {
    }
    virtual ~SarsaAdaptive()
    {
      delete gammaPhi_tp1MinusPhi_t;
    }

  public:

    void updateAlpha(const SparseVector<T>& phi_t, const SparseVector<T>& phi_tp1, double r_tp1)
    {
      // Update the adaptive step-size
      double secondValue = std::abs(
          super::e->vect().dot(
              gammaPhi_tp1MinusPhi_t->set(phi_tp1).multiplyToSelf(super::gamma).substractToSelf(
                  phi_t)));
      secondValue = std::max(minUpdate, secondValue); // To get numerically stable value
      super::alpha = std::min(super::alpha, 1.0f / secondValue);
    }

};

// Gradient decent
template<class T>
class GQ: public Predictor<T>
{
  private:
    double delta_t;
    bool initialized;

  protected:
    double alpha_v, alpha_w, beta_tp1, lambda_t;
    Trace<T>* e;
    SparseVector<T>* v;
    SparseVector<T>* w;

  public:
    GQ(const double& alpha_v, const double& alpha_w, const double& beta_tp1, const double& lambda_t,
        Trace<T>* e) :
        delta_t(0), initialized(false), alpha_v(alpha_v), alpha_w(alpha_w), beta_tp1(beta_tp1), lambda_t(
            lambda_t), e(e), v(new SparseVector<T>(e->vect().dimension())), w(
            new SparseVector<T>(e->vect().dimension()))
    {
    }

    virtual ~GQ()
    {
      delete v;
      delete w;
    }

    double initialize()
    {
      e->clear();
      initialized = true;
      return 0.0;
    }

    double update(const SparseVector<T>& phi_t, const SparseVector<T>& phi_bar_tp1,
        const double& rho_t, double r_tp1, double z_tp1)
    {
      assert(initialized);
      delta_t = r_tp1 + beta_tp1 * z_tp1 + (1.0 - beta_tp1) * v->dot(phi_bar_tp1) - v->dot(phi_t);
      e->update((1.0 - beta_tp1) * lambda_t * rho_t, phi_t); // paper says beta_t ?
      // v
      // part 1
      v->addToSelf(alpha_v * delta_t, e->vect());
      // part 2
      v->addToSelf(-alpha_v * (1.0 - beta_tp1) * (1.0 - lambda_t) * w->dot(e->vect()), phi_bar_tp1); // paper says beta_t ?
      // w
      // part 1
      w->addToSelf(alpha_w * delta_t, e->vect());
      // part 2
      w->addToSelf(-alpha_w * w->dot(phi_t), phi_t);
      return delta_t;
    }

    void reset()
    {
      e->clear();
      v->clear();
      w->clear();
      initialized = false;
    }

    int dimension() const
    {
      return v->dimension();
    }

    double predict(const SparseVector<T>& phi_sa) const
    {
      return v->dot(phi_sa);
    }

    void persist(const std::string& f) const
    {
      v->persist(f);
    }
    void resurrect(const std::string& f)
    {
      v->resurrect(f);
    }
};

// Prediction problems
template<class T>
class GTDLambda: public GVF<T>
{
  private:
    double delta_t;
    bool initialized;

  protected:
    double alpha_v, alpha_w, gamma_t, lambda_t;
    Trace<T>* e;
    SparseVector<T>* v;
    SparseVector<T>* w;

  public:
    GTDLambda(const double& alpha_v, const double& alpha_w, const double& gamma_t,
        const double& lambda_t, Trace<T>* e) :
        delta_t(0), initialized(false), alpha_v(alpha_v), alpha_w(alpha_w), gamma_t(gamma_t), lambda_t(
            lambda_t), e(e), v(new SparseVector<T>(e->vect().dimension())), w(
            new SparseVector<T>(e->vect().dimension()))
    {
    }

    virtual ~GTDLambda()
    {
      delete v;
      delete w;
    }

    double initialize()
    {
      e->clear();
      initialized = true;
      return 0.0;
    }

    double update(const SparseVector<T>& phi_t, const SparseVector<T>& phi_tp1,
        const double& gamma_tp1, const double& lambda_tp1, const double& rho_t, const double& r_tp1,
        const double& z_tp1)
    {
      delta_t = r_tp1 + (1.0 - gamma_tp1) * z_tp1 + gamma_tp1 * v->dot(phi_tp1) - v->dot(phi_t);
      e->update(gamma_t * lambda_t, phi_t);
      e->multiplyToSelf(rho_t);

      // v
      // part 1
      v->addToSelf(alpha_v * delta_t, e->vect());
      // part2
      v->addToSelf(-alpha_v * gamma_tp1 * (1.0 - lambda_tp1) * w->dot(e->vect()), phi_tp1);

      // w
      // part 1
      w->addToSelf(alpha_w * delta_t, e->vect());
      // part 2
      w->addToSelf(-alpha_w * w->dot(phi_t), phi_t);

      gamma_t = gamma_tp1;
      lambda_t = lambda_tp1;
      return delta_t;
    }

    double update(const SparseVector<T>& phi_t, const SparseVector<T>& phi_tp1, const double& rho_t,
        const double& gamma_t, double r_tp1, double z_tp1)
    {
      return update(phi_t, phi_tp1, gamma_t, lambda_t, rho_t, r_tp1, z_tp1);
    }

    void reset()
    {
      e->clear();
      v->clear();
      w->clear();
      initialized = false;
    }

    int dimension() const
    {
      return v->dimension();
    }

    double predict(const SparseVector<T>& phi) const
    {
      return v->dot(phi);
    }

    void persist(const std::string& f) const
    {
      v->persist(f);
    }
    void resurrect(const std::string& f)
    {
      v->resurrect(f);
    }

};

} // namespace RLLib

#endif /* PREDICTORALGORITHM_H_ */
