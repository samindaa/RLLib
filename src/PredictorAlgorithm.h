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
    T delta_t;
    T alpha_v;
    T gamma;
    Vector<T>* v;
    bool initialized;
  public:
    TD(const T& alpha_v, const T& gamma, const int& nbFeatures) :
        delta_t(0), alpha_v(alpha_v), gamma(gamma), v(new PVector<T>(nbFeatures)), initialized(
            false)
    {
    }
    virtual ~TD()
    {
      delete v;
    }

  public:

    T initialize()
    {
      initialized = true;
      delta_t = 0;
      return delta_t;
    }

    virtual T update(const Vector<T>* x_t, const Vector<T>* x_tp1, const T& r_tp1,
        const T& gamma_tp1)
    {
      ASSERT(initialized);
      delta_t = r_tp1 + gamma_tp1 * v->dot(x_tp1) - v->dot(x_t);
      v->addToSelf(alpha_v * delta_t, x_t);
      return delta_t;
    }

    T update(const Vector<T>* x_t, const Vector<T>* x_tp1, T r_tp1)
    {
      ASSERT(initialized);
      return update(x_t, x_tp1, r_tp1, gamma);
    }

    void reset()
    {
      v->clear();
    }

    T predict(const Vector<T>* x) const
    {
      return v->dot(x);
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
class TDLambdaAbstract: public TD<T>
{
  protected:
    typedef TD<T> Base;
    T lambda, gamma_t;
    Trace<T>* e;

  public:
    TDLambdaAbstract(const T& alpha, const T& gamma, const T& lambda, Trace<T>* e) :
        TD<T>(alpha, gamma, e->vect()->dimension()), lambda(lambda), gamma_t(gamma), e(e)
    {
    }
    virtual ~TDLambdaAbstract()
    {
    }

  public:

    T initialize()
    {
      Base::initialize();
      e->clear();
      gamma_t = Base::gamma;
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
class TDLambda: public TDLambdaAbstract<T>
{
  protected:
    typedef TDLambdaAbstract<T> Base;

  public:
    TDLambda(const T& alpha, const T& gamma, const T& lambda, Trace<T>* e) :
        TDLambdaAbstract<T>(alpha, gamma, lambda, e)
    {
    }

    virtual ~TDLambda()
    {
    }

  public:
    T update(const Vector<T>* x_t, const Vector<T>* x_tp1, const T& r_tp1,
        const T& gamma_tp1)
    {
      ASSERT(TD<T>::initialized);

      TD<T>::delta_t = r_tp1 + gamma_tp1 * TD<T>::v->dot(x_tp1) - TD<T>::v->dot(x_t);
      Base::e->update(Base::lambda * Base::gamma_t, x_t, TD<T>::alpha_v);
      TD<T>::v->addToSelf(TD<T>::delta_t, Base::e->vect());
      Base::gamma_t = gamma_tp1;
      return TD<T>::delta_t;
    }
};

template<class T>
class TDLambdaTrue: public TDLambdaAbstract<T>
{
  protected:
    typedef TDLambdaAbstract<T> Base;
    T v_t;
    T v_tp1;
    bool initializedVt;
  public:
    TDLambdaTrue(const T& alpha, const T& gamma, const T& lambda, Trace<T>* e) :
        TDLambdaAbstract<T>(alpha, gamma, lambda, e), v_t(0), v_tp1(0), initializedVt(false)
    {
    }

    virtual ~TDLambdaTrue()
    {
    }

    T initialize()
    {
      Base::initialize();
      initializedVt = false;
      return Base::delta_t;
    }

    void initialize(const Vector<T>* x_t)
    {
      v_t = TD<T>::v->dot(x_t);
      initializedVt = true;
    }

    T update(const Vector<T>* x_t, const Vector<T>* x_tp1, const T& r_tp1,
        const T& gamma_tp1)
    {
      ASSERT(TD<T>::initialized);
      if (!initializedVt)
        initialize(x_t);
      v_tp1 = TD<T>::v->dot(x_tp1);
      TD<T>::delta_t = r_tp1 + gamma_tp1 * v_tp1 - v_t;
      Base::e->update(Base::lambda * Base::gamma_t, x_t,
          TD<T>::alpha_v * (T(1) - Base::gamma_t * Base::lambda * Base::e->vect()->dot(x_t)));
      TD<T>::v->addToSelf(TD<T>::alpha_v * (v_t - TD<T>::v->dot(x_t)), x_t)->addToSelf(
          TD<T>::delta_t, Base::e->vect());
      v_t = v_tp1;
      Base::gamma_t = gamma_tp1;
      return TD<T>::delta_t;
    }
};

template<class T>
class TDLambdaAlphaBound: public TDLambdaAbstract<T>
{
  private:
    typedef TDLambdaAbstract<T> Base;
    Vector<T>* gammaXtp1MinusX;

  public:
    TDLambdaAlphaBound(const T& alpha, const T& gamma, const T& lambda, Trace<T>* e) :
        TDLambdaAbstract<T>(alpha, gamma, lambda, e), gammaXtp1MinusX(
            new SVector<T>(e->vect()->dimension()))
    {
    }

    virtual ~TDLambdaAlphaBound()
    {
      delete gammaXtp1MinusX;
    }

    void reset()
    {
      Base::reset();
      TD<T>::alpha_v = T(1);
    }

  private:
    void updateAlpha(const Vector<T>* x_t, const Vector<T>* x_tp1, const T& gamma_tp1)
    {
      // Update the adaptive step-size
      T b = std::fabs(
          Base::e->vect()->dot(
              gammaXtp1MinusX->set(x_tp1)->mapMultiplyToSelf(gamma_tp1)->subtractToSelf(x_t)));
      if (b > 0.0f)
        TD<T>::alpha_v = std::min(TD<T>::alpha_v, 1.0f / b);
    }

  public:
    T update(const Vector<T>* x_t, const Vector<T>* x_tp1, const T& r_tp1,
        const T& gamma_tp1)
    {
      ASSERT(TD<T>::initialized);

      TD<T>::delta_t = r_tp1 + gamma_tp1 * TD<T>::v->dot(x_tp1) - TD<T>::v->dot(x_t);
      Base::e->update(Base::lambda * Base::gamma_t, x_t);
      updateAlpha(x_t, x_tp1, gamma_tp1);
      TD<T>::v->addToSelf(TD<T>::alpha_v * TD<T>::delta_t, Base::e->vect());
      Base::gamma_t = gamma_tp1;
      return TD<T>::delta_t;
    }

};

template<class T>
class Sarsa: public Predictor<T>, public LinearLearner<T>
{
  protected:
    T v_t, v_tp1, delta; // temporary variables
    bool initialized;
    T alpha, gamma, lambda;
    Trace<T>* e;
    Vector<T>* q;

  public:
    Sarsa(const T& alpha, const T& gamma, const T& lambda, Trace<T>* e) :
        v_t(0), v_tp1(0), delta(0), initialized(false), alpha(alpha), gamma(gamma), lambda(lambda), e(
            e), q(new PVector<T>(e->vect()->dimension()))
    {
    }
    virtual ~Sarsa()
    {
      delete q;
    }

  public:

    T initialize()
    {
      e->clear();
      initialized = true;
      return T(0);
    }

    virtual T update(const Vector<T>* phi_t, const Vector<T>* phi_tp1, T r_tp1)
    {
      ASSERT(initialized);

      v_t = q->dot(phi_t);
      v_tp1 = q->dot(phi_tp1);
      e->update(gamma * lambda, phi_t, alpha);
      delta = r_tp1 + gamma * v_tp1 - v_t;
      q->addToSelf(delta, e->vect());
      return delta;
    }

    void reset()
    {
      e->clear();
      q->clear();
      initialized = false;
    }

    T predict(const Vector<T>* phi_sa) const
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
class SarsaTrue: public Sarsa<T>
{
  private:
    typedef Sarsa<T> Base;
    bool initializedVt;
  public:
    SarsaTrue(const T& alpha, const T& gamma, const T& lambda, Trace<T>* e) :
        Sarsa<T>(alpha, gamma, lambda, e), initializedVt(false)
    {
    }

    virtual ~SarsaTrue()
    {
    }

  public:
    T initialize()
    {
      Base::initialize();
      initializedVt = false;
      return T(0);
    }

    void initialize(const Vector<T>* phi_t)
    {
      Base::v_t = Base::q->dot(phi_t);
      initializedVt = true;
    }

    T update(const Vector<T>* phi_t, const Vector<T>* phi_tp1, T r_tp1)
    {
      ASSERT(Base::initialized);
      if (!initializedVt)
        initialize(phi_t);
      Base::v_tp1 = Base::q->dot(phi_tp1);
      Base::delta = r_tp1 + Base::gamma * Base::v_tp1 - Base::v_t;
      Base::e->update(Base::lambda * Base::gamma, phi_t,
          Base::alpha * (T(1) - Base::gamma * Base::lambda * Base::e->vect()->dot(phi_t)));
      Base::q->addToSelf(Base::alpha * (Base::v_t - Base::q->dot(phi_t)), phi_t)->addToSelf(
          Base::delta, Base::e->vect());
      Base::v_t = Base::v_tp1;
      return Base::delta;
    }
};

template<class T>
class SarsaAlphaBound: public Sarsa<T>
{
  private:
    typedef Sarsa<T> Base;
    Vector<T>* gammaXtp1MinusX;
    T alpha_0;
  public:
    SarsaAlphaBound(const T& alpha, const T& gamma, const T& lambda, Trace<T>* e) :
        Sarsa<T>(alpha, gamma, lambda, e), gammaXtp1MinusX(new SVector<T>(e->vect()->dimension())), alpha_0(
            alpha)
    {
    }

    virtual ~SarsaAlphaBound()
    {
      delete gammaXtp1MinusX;
    }

    void reset()
    {
      Base::reset();
      Base::alpha = alpha_0;
    }

  private:
    void updateAlpha(const Vector<T>* phi_t, const Vector<T>* phi_tp1)
    {
      // Update the adaptive step-size
      T b = std::fabs(
          Base::e->vect()->dot(
              gammaXtp1MinusX->set(phi_tp1)->mapMultiplyToSelf(Base::gamma)->subtractToSelf(
                  phi_t)));
      if (b > 0.0f)
        Base::alpha = std::min(Base::alpha, 1.0f / b);
    }

  public:
    T update(const Vector<T>* phi_t, const Vector<T>* phi_tp1, T r_tp1)
    {
      ASSERT(Base::initialized);

      Base::v_t = Base::q->dot(phi_t);
      Base::v_tp1 = Base::q->dot(phi_tp1);
      Base::e->update(Base::gamma * Base::lambda, phi_t);
      Base::delta = r_tp1 + Base::gamma * Base::v_tp1 - Base::v_t;
      updateAlpha(phi_t, phi_tp1);
      Base::q->addToSelf(Base::alpha * Base::delta, Base::e->vect());
      return Base::delta;
    }
};

// Gradient decent
template<class T>
class GQ: public Predictor<T>, public LinearLearner<T>
{
  private:
    T delta_t;
    bool initialized;

  protected:
    T alpha_v, alpha_w, beta_tp1, lambda_t;
    Trace<T>* e;
    Vector<T>* v;
    Vector<T>* w;

  public:
    GQ(const T& alpha_v, const T& alpha_w, const T& beta_tp1, const T& lambda_t,
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

    T initialize()
    {
      e->clear();
      initialized = true;
      return T(0);
    }

    T update(const Vector<T>* phi_t, const Vector<T>* phi_bar_tp1, const T& rho_t,
        T r_tp1, T z_tp1)
    {
      ASSERT(initialized);
      delta_t = r_tp1 + beta_tp1 * z_tp1 + (T(1) - beta_tp1) * v->dot(phi_bar_tp1) - v->dot(phi_t);
      e->update((T(1) - beta_tp1) * lambda_t * rho_t, phi_t); // paper says beta_t ?
      // v
      // part 1
      v->addToSelf(alpha_v * delta_t, e->vect());
      // part 2
      v->addToSelf(-alpha_v * (T(1) - beta_tp1) * (T(1) - lambda_t) * w->dot(e->vect()),
          phi_bar_tp1); // paper says beta_t ?

      // w
      // part 2
      w->addToSelf(-alpha_w * w->dot(phi_t), phi_t);
      // part 1
      w->addToSelf(alpha_w * delta_t, e->vect());
      return delta_t;
    }

    void reset()
    {
      e->clear();
      v->clear();
      w->clear();
      initialized = false;
    }

    T predict(const Vector<T>* phi_sa) const
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
    T delta_t;
    bool initialized;

  protected:
    T alpha_v, alpha_w, gamma_t, lambda_t;
    Trace<T>* e;
    Vector<T>* v;
    Vector<T>* w;

  public:
    GTDLambda(const T& alpha_v, const T& alpha_w, const T& gamma_t,
        const T& lambda_t, Trace<T>* e) :
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

    T initialize()
    {
      e->clear();
      initialized = true;
      return T(0);
    }

    T update(const Vector<T>* phi_t, const Vector<T>* phi_tp1, const T& gamma_tp1,
        const T& lambda_tp1, const T& rho_t, const T& r_tp1, const T& z_tp1)
    {
      delta_t = r_tp1 + (T(1) - gamma_tp1) * z_tp1 + gamma_tp1 * v->dot(phi_tp1) - v->dot(phi_t);
      e->update(gamma_t * lambda_t, phi_t);
      e->vect()->mapMultiplyToSelf(rho_t);

      // v
      // part 1
      v->addToSelf(alpha_v * delta_t, e->vect());
      // part2
      v->addToSelf(-alpha_v * gamma_tp1 * (T(1) - lambda_tp1) * w->dot(e->vect()), phi_tp1);

      // w
      // part 2
      w->addToSelf(-alpha_w * w->dot(phi_t), phi_t);
      // part 1
      w->addToSelf(alpha_w * delta_t, e->vect());

      gamma_t = gamma_tp1;
      lambda_t = lambda_tp1;
      return delta_t;
    }

    T update(const Vector<T>* phi_t, const Vector<T>* phi_tp1, const T& rho_t,
        T r_tp1, T z_tp1)
    {
      return update(phi_t, phi_tp1, gamma_t, lambda_t, rho_t, r_tp1, z_tp1);
    }

    T update(const Vector<T>* x_t, const Vector<T>* x_tp1, T r_tp1)
    {
      return update(x_t, x_tp1, gamma_t, lambda_t, T(1), r_tp1, T(0));
    }

    void reset()
    {
      e->clear();
      v->clear();
      w->clear();
      initialized = false;
    }

    T predict(const Vector<T>* phi) const
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
