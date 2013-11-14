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
    double alpha_v;
    double gamma;
    Vector<T>* v;
    bool initialized;
  public:
    TD(const double& alpha_v, const double& gamma, const int& nbFeatures) :
        delta_t(0), alpha_v(alpha_v), gamma(gamma), v(new PVector<T>(nbFeatures)), initialized(
            false)
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

    virtual double update(const Vector<T>* x_t, const Vector<T>* x_tp1, const double& r_tp1,
        const double& gamma_tp1)
    {
      assert(initialized);
      delta_t = r_tp1 + gamma_tp1 * v->dot(x_tp1) - v->dot(x_t);
      v->addToSelf(alpha_v * delta_t, x_t);
      return delta_t;
    }

    double update(const Vector<T>* x_t, const Vector<T>* x_tp1, double r_tp1)
    {
      assert(initialized);
      return update(x_t, x_tp1, r_tp1, gamma);
    }

    void reset()
    {
      v->clear();
    }

    double predict(const Vector<T>* phi) const
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

    const Vector<T>* weights() const
    {
      return v;
    }

};

template<class T>
class TDLambda: public TD<T>
{
  protected:
    typedef TD<T> Base;
    double lambda, gamma_t;
    Trace<T>* e;
  public:
    TDLambda(const double& alpha, const double& gamma, const double& lambda, Trace<T>* e) :
        TD<T>(alpha, gamma, e->vect()->dimension()), lambda(lambda), gamma_t(gamma), e(e)
    {
    }
    virtual ~TDLambda()
    {
    }

  public:

    double initialize()
    {
      Base::initialize();
      e->clear();
      gamma_t = Base::gamma;
      return Base::delta_t;
    }

    virtual void updateAlpha(const Vector<T>* x_t, const Vector<T>* x_tp1, const double& r_tp1,
        const double& gamma_tp1)
    {/*Default is for fixed step-size*/
    }

    virtual double update(const Vector<T>* x_t, const Vector<T>* x_tp1, const double& r_tp1,
        const double& gamma_tp1)
    {
      assert(Base::initialized);

      Base::delta_t = r_tp1 + gamma_tp1 * Base::v->dot(x_tp1) - Base::v->dot(x_t);
      e->update(lambda * gamma_t, x_t);
      updateAlpha(x_t, x_tp1, r_tp1, gamma_tp1);
      Base::v->addToSelf(Base::alpha_v * Base::delta_t, e->vect());
      gamma_t = gamma_tp1;
      return Base::delta_t;
    }

    void reset()
    {
      Base::reset();
      gamma_t = Base::gamma;
      e->clear();
    }
};

template<class T>
class TDLambdaAlphaBound: public TDLambda<T>
{
  private:
    typedef TDLambda<T> Base;
    SparseVector<T>* gammaX_tp1MinusX_t;
  public:
    TDLambdaAlphaBound(const double& gamma, const double& lambda, Trace<T>* e) :
        TDLambda<T>(1.0f, gamma, lambda, e), gammaX_tp1MinusX_t(
            new SparseVector<T>(e->vect().dimension()))
    {
    }

    virtual ~TDLambdaAlphaBound()
    {
      delete gammaX_tp1MinusX_t;
    }

    void reset()
    {
      Base::reset();
      TD<T>::alpha_v = 1.0f;
    }

    virtual void updateAlpha(const Vector<T>* x_t, const Vector<T>* x_tp1, const double& r_tp1,
        const double& gamma_tp1)
    {
      // Update the adaptive step-size
      double b = std::abs(
          Base::e->vect().dot(
              gammaX_tp1MinusX_t->set(x_tp1).multiplyToSelf(gamma_tp1).substractToSelf(x_t)));
      if (b > 0.0f)
        TD<T>::alpha_v = std::min(TD<T>::alpha_v, 1.0f / b);
    }

};

template<class T>
class Sarsa: public Predictor<T>, public LinearLearner<T>
{
  protected:
    double v_t, v_tp1, delta; // temporary variables
    bool initialized;
    double alpha, gamma, lambda;
    Trace<T>* e;
    Vector<T>* q;
  public:
    Sarsa(const double& alpha, const double& gamma, const double& lambda, Trace<T>* e) :
        v_t(0), v_tp1(0), delta(0), initialized(false), alpha(alpha), gamma(gamma), lambda(lambda), e(
            e), q(new PVector<T>(e->vect()->dimension()))
    {
    }
    virtual ~Sarsa()
    {
      delete q;
    }

  public:

    double initialize()
    {
      e->clear();
      initialized = true;
      return 0.0;
    }

    virtual void updateAlpha(const Vector<T>* phi_t, const Vector<T>* phi_tp1, double r_tp1)
    {/*Default is for fixed step-size*/
    }

    double update(const Vector<T>* phi_t, const Vector<T>* phi_tp1, double r_tp1)
    {
      assert(initialized);

      v_t = q->dot(phi_t);
      v_tp1 = q->dot(phi_tp1);
      e->update(gamma * lambda, phi_t);
      updateAlpha(phi_t, phi_tp1, r_tp1);
      delta = r_tp1 + gamma * v_tp1 - v_t;
      q->addToSelf(alpha * delta, e->vect());
      return delta;
    }

    void reset()
    {
      e->clear();
      q->clear();
      initialized = false;
    }

    double predict(const Vector<T>* phi_sa) const
    {
      return q->dot(phi_sa);
    }

    void persist(const std::string& f) const
    {
      q->persist(f);
    }

    void resurrect(const std::string& f)
    {
      q->resurrect(f);
    }

    const Vector<T>* weights() const
    {
      return q;
    }
};

template<class T>
class SarsaAlphaBound: public Sarsa<T>
{
  private:
    typedef Sarsa<T> Base;
    SparseVector<T>* gammaXtp1MinusX;
  public:
    SarsaAlphaBound(const double& gamma, const double& lambda, Trace<T>* e) :
        Sarsa<T>(1.0f/*According to the paper*/, gamma, lambda, e), gammaXtp1MinusX(
            new SVector<T>(e->vect()->dimension()))
    {
    }
    virtual ~SarsaAlphaBound()
    {
      delete gammaXtp1MinusX;
    }

    void reset()
    {
      Base::reset();
      Base::alpha = 1.0f;
    }

    void updateAlpha(const Vector<T>* phi_t, const Vector<T>* phi_tp1, double r_tp1)
    {
      // Update the adaptive step-size
      double b = std::fabs(
          Base::e->vect()->dot(
              gammaXtp1MinusX->set(phi_tp1)->mapMultiplyToSelf(Base::gamma)->subtractToSelf(
                  phi_t)));
      if (b > 0.0f)
        Base::alpha = std::min(Base::alpha, 1.0f / b);
    }

};

// Gradient decent
template<class T>
class GQ: public Predictor<T>, public LinearLearner<T>
{
  private:
    double delta_t;
    bool initialized;

  protected:
    double alpha_v, alpha_w, beta_tp1, lambda_t;
    Trace<T>* e;
    Vector<T>* v;
    Vector<T>* w;

  public:
    GQ(const double& alpha_v, const double& alpha_w, const double& beta_tp1, const double& lambda_t,
        Trace<T>* e) :
        delta_t(0), initialized(false), alpha_v(alpha_v), alpha_w(alpha_w), beta_tp1(beta_tp1), lambda_t(
            lambda_t), e(e), v(new PVector<T>(e->vect()->dimension())), w(
            new PVector<T>(e->vect()->dimension()))
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

    double update(const Vector<T>* phi_t, const Vector<T>* phi_bar_tp1, const double& rho_t,
        double r_tp1, double z_tp1)
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

    double predict(const Vector<T>* phi_sa) const
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

    const Vector<T>* weights() const
    {
      return v;
    }
};

// Prediction problems
template<class T>
class GTDLambda: public OnPolicyTD<T>, public GVF<T>
{
  private:
    double delta_t;
    bool initialized;

  protected:
    double alpha_v, alpha_w, gamma_t, lambda_t;
    Trace<T>* e;
    Vector<T>* v;
    Vector<T>* w;

  public:
    GTDLambda(const double& alpha_v, const double& alpha_w, const double& gamma_t,
        const double& lambda_t, Trace<T>* e) :
        delta_t(0), initialized(false), alpha_v(alpha_v), alpha_w(alpha_w), gamma_t(gamma_t), lambda_t(
            lambda_t), e(e), v(new PVector<T>(e->vect()->dimension())), w(
            new PVector<T>(e->vect()->dimension()))
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

    double update(const Vector<T>* phi_t, const Vector<T>* phi_tp1, const double& gamma_tp1,
        const double& lambda_tp1, const double& rho_t, const double& r_tp1, const double& z_tp1)
    {
      delta_t = r_tp1 + (1.0 - gamma_tp1) * z_tp1 + gamma_tp1 * v->dot(phi_tp1) - v->dot(phi_t);
      e->update(gamma_t * lambda_t, phi_t);
      e->vect()->mapMultiplyToSelf(rho_t);

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

    double update(const Vector<T>* phi_t, const Vector<T>* phi_tp1, const double& rho_t,
        double r_tp1, double z_tp1)
    {
      return update(phi_t, phi_tp1, gamma_t, lambda_t, rho_t, r_tp1, z_tp1);
    }

    double update(const Vector<T>* x_t, const Vector<T>* x_tp1, double r_tp1)
    {
      return update(x_t, x_tp1, gamma_t, lambda_t, 1.0, r_tp1, 0.0);
    }

    void reset()
    {
      e->clear();
      v->clear();
      w->clear();
      initialized = false;
    }

    double predict(const Vector<T>* phi) const
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

    const Vector<T>* weights() const
    {
      return v;
    }
};

} // namespace RLLib

#endif /* PREDICTORALGORITHM_H_ */
