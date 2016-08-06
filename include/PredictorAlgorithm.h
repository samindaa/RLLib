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
  template<typename T>
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
          delta_t(0), alpha_v(alpha_v), gamma(gamma), v(new PVector<T>(nbFeatures)), //
          initialized(false)
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

      T update(const Vector<T>* x_t, const Vector<T>* x_tp1, const T& r_tp1)
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

      void persist(const char* f) const
      {
        v->persist(f);
      }

      void resurrect(const char* f)
      {
        v->resurrect(f);
      }

      Vector<T>* weights() const
      {
        return v;
      }
  };

  template<typename T>
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

  template<typename T>
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
      T update(const Vector<T>* x_t, const Vector<T>* x_tp1, const T& r_tp1, const T& gamma_tp1)
      {
        ASSERT(TD<T>::initialized);
        TD<T>::delta_t = r_tp1 + gamma_tp1 * TD<T>::v->dot(x_tp1) - TD<T>::v->dot(x_t);
        Base::e->update(Base::lambda * Base::gamma_t, x_t, TD<T>::alpha_v);
        TD<T>::v->addToSelf(TD<T>::delta_t, Base::e->vect());
        Base::gamma_t = gamma_tp1;
        return TD<T>::delta_t;
      }
  };

  template<typename T>
  class TDLambdaTrue: public TDLambdaAbstract<T>
  {
    protected:
      typedef TDLambdaAbstract<T> Base;
      T v_t;
      T v_tp1;
      T v_old;

    public:
      TDLambdaTrue(const T& alpha, const T& gamma, const T& lambda, Trace<T>* e) :
          TDLambdaAbstract<T>(alpha, gamma, lambda, e), v_t(0), v_tp1(0), v_old(0)
      {
      }

      virtual ~TDLambdaTrue()
      {
      }

      T initialize()
      {
        Base::initialize();
        v_old = 0;
        return Base::delta_t;
      }

      T update(const Vector<T>* x_t, const Vector<T>* x_tp1, const T& r_tp1, const T& gamma_tp1)
      {
        ASSERT(TD<T>::initialized);
        v_t = TD<T>::v->dot(x_t);
        v_tp1 = TD<T>::v->dot(x_tp1);
        TD<T>::delta_t = r_tp1 + gamma_tp1 * v_tp1 - v_t;

        Base::e->update(Base::gamma_t * Base::lambda, x_t,
            (T(1) - TD<T>::alpha_v * Base::gamma_t * Base::lambda * Base::e->vect()->dot(x_t)));
        TD<T>::v->addToSelf(-TD<T>::alpha_v * (v_t - v_old), x_t)->addToSelf(
            TD<T>::alpha_v * (TD<T>::delta_t + v_t - v_old), Base::e->vect());

        v_old = v_tp1;
        Base::gamma_t = gamma_tp1;
        return TD<T>::delta_t;
      }

  };

  template<typename T>
  class TDLambdaAlphaBound: public TDLambdaAbstract<T>
  {
    private:
      typedef TDLambdaAbstract<T> Base;
      Vector<T>* gammaXtp1MinusX;

    public:
      TDLambdaAlphaBound(const T& alpha, const T& gamma, const T& lambda, Trace<T>* e) :
          TDLambdaAbstract<T>(alpha, gamma, lambda, e), //
          gammaXtp1MinusX(new SVector<T>(e->vect()->dimension()))
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
        T b = std::abs(
            Base::e->vect()->dot(
                gammaXtp1MinusX->set(x_tp1)->mapMultiplyToSelf(gamma_tp1)->subtractToSelf(x_t)));
        if (b > 0.0f)
          TD<T>::alpha_v = std::min(TD<T>::alpha_v, 1.0f / b);
      }

    public:
      T update(const Vector<T>* x_t, const Vector<T>* x_tp1, const T& r_tp1, const T& gamma_tp1)
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

  template<typename T>
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
          v_t(0), v_tp1(0), delta(0), initialized(false), alpha(alpha), gamma(gamma), //
          lambda(lambda), e(e), q(new PVector<T>(e->vect()->dimension()))
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

      virtual T update(const Vector<T>* phi_t, const Vector<T>* phi_tp1, const T& r_tp1)
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

      void persist(const char* f) const
      {
        q->persist(f);
      }

      void resurrect(const char* f)
      {
        q->resurrect(f);
      }

      Vector<T>* weights() const
      {
        return q;
      }
  };

  template<typename T>
  class SarsaTrue: public Sarsa<T>
  {
    private:
      typedef Sarsa<T> Base;
      T v_old;

    public:
      SarsaTrue(const T& alpha, const T& gamma, const T& lambda, Trace<T>* e) :
          Sarsa<T>(alpha, gamma, lambda, e), v_old(0)
      {
      }

      virtual ~SarsaTrue()
      {
      }

    public:
      T initialize()
      {
        Base::initialize();
        v_old = 0;
        return T(0);
      }

      T update(const Vector<T>* phi_t, const Vector<T>* phi_tp1, const T& r_tp1)
      {
        ASSERT(Base::initialized);

        Base::v_t = Base::q->dot(phi_t);
        Base::v_tp1 = Base::q->dot(phi_tp1);
        Base::delta = r_tp1 + Base::gamma * Base::v_tp1 - Base::v_t;

        Base::e->update(Base::gamma * Base::lambda, phi_t,
            (T(1) - Base::alpha * Base::gamma * Base::lambda * Base::e->vect()->dot(phi_t)));
        Base::q->addToSelf(-Base::alpha * (Base::v_t - v_old), phi_t)->addToSelf(
            Base::alpha * (Base::delta + Base::v_t - v_old), Base::e->vect());

        v_old = Base::v_tp1;
        return Base::delta;
      }
  };

  template<typename T>
  class SarsaAlphaBound: public Sarsa<T>
  {
    private:
      typedef Sarsa<T> Base;
      Vector<T>* gammaXtp1MinusX;
      T alpha_0;
    public:
      SarsaAlphaBound(const T& alpha, const T& gamma, const T& lambda, Trace<T>* e) :
          Sarsa<T>(alpha, gamma, lambda, e), gammaXtp1MinusX(
              new SVector<T>(e->vect()->dimension())), //
          alpha_0(alpha)
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
        T b = std::abs(
            Base::e->vect()->dot(
                gammaXtp1MinusX->set(phi_tp1)->mapMultiplyToSelf(Base::gamma)->subtractToSelf(
                    phi_t)));
        if (b > 0.0f)
          Base::alpha = std::min(Base::alpha, 1.0f / b);
      }

    public:
      T update(const Vector<T>* phi_t, const Vector<T>* phi_tp1, const T& r_tp1)
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
  template<typename T>
  class GQ: public OnPolicyTD<T>, public GVF<T>
  {
    private:
      T delta_t;
      bool initialized;

    protected:
      T alpha_v, alpha_w, gamma_t, gamma_tp1, lambda_t, lambda_tp1;
      Trace<T>* e;
      Vector<T>* v;
      Vector<T>* w;

    public:
      GQ(const T& alpha_v, const T& alpha_w, const T& gamma_tp1, const T& lambda_t, Trace<T>* e) :
          delta_t(0), initialized(false), alpha_v(alpha_v), alpha_w(alpha_w), gamma_t(gamma_tp1), //
          gamma_tp1(gamma_tp1), lambda_t(lambda_t), lambda_tp1(lambda_t), e(e), //
          v(new PVector<T>(e->vect()->dimension())), w(new PVector<T>(e->vect()->dimension()))
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

      T update(const Vector<T>* phi_t, const Vector<T>* phi_bar_tp1, const T& gamma_tp1,
          const T& lambda_tp1, const T& rho_t, const T& r_tp1, const T& z_tp1)
      {
        ASSERT(initialized);
        delta_t = //
            r_tp1 + (T(1) - gamma_tp1) * z_tp1 + gamma_tp1 * v->dot(phi_bar_tp1) - v->dot(phi_t);
        e->update(gamma_t * lambda_t * rho_t, phi_t);
        // v
        // part 1
        v->addToSelf(alpha_v * delta_t, e->vect());
        // part 2
        v->addToSelf(-alpha_v * gamma_tp1 * (T(1) - lambda_tp1) * w->dot(e->vect()), phi_bar_tp1);

        // w
        // part 2
        w->addToSelf(-alpha_w * w->dot(phi_t), phi_t);
        // part 1
        w->addToSelf(alpha_w * delta_t, e->vect());

        gamma_t = gamma_tp1;
        lambda_t = lambda_tp1;
        return delta_t;

      }

      T update(const Vector<T>* phi_t, const Vector<T>* phi_bar_tp1, const T& rho_t, const T& r_tp1,
          const T& z_tp1)
      {
        return update(phi_t, phi_bar_tp1, gamma_tp1, lambda_tp1, rho_t, r_tp1, z_tp1);
      }

      T update(const Vector<T>* phi_t, const Vector<T>* phi_bar_tp1, const T& r_tp1)
      {
        return update(phi_t, phi_bar_tp1, gamma_t, lambda_t, T(1), r_tp1, T(0));
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

      void persist(const char* f) const
      {
        v->persist(f);
      }
      void resurrect(const char* f)
      {
        v->resurrect(f);
      }

      Vector<T>* weights() const
      {
        return v;
      }

      void set_gamma_tp1(const T& gamma_tp1)
      {
        this->gamma_tp1 = gamma_tp1;
      }

      void set_lambda_tp1(const T& lambda_tp1)
      {
        this->lambda_tp1 = lambda_tp1;
      }
  };

  template<typename T>
  class GTDLambdaAbstract: public OnPolicyTD<T>, public GVF<T>
  {
    protected:
      T delta_t;
      bool initialized;

      T alpha_v, alpha_w, gamma_t, lambda_t;
      Trace<T>* e;
      Vector<T>* v;
      Vector<T>* w;

      GTDLambdaAbstract(const T& alpha_v, const T& alpha_w, const T& gamma_t, const T& lambda_t,
          Trace<T>* e) :
          delta_t(0), initialized(false), alpha_v(alpha_v), alpha_w(alpha_w), gamma_t(gamma_t), //
          lambda_t(lambda_t), e(e), v(new PVector<T>(e->vect()->dimension())), //
          w(new PVector<T>(e->vect()->dimension()))
      {
      }

      virtual ~GTDLambdaAbstract()
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

      virtual T update(const Vector<T>* x_t, const Vector<T>* x_tp1, const T& gamma_tp1,
          const T& lambda_tp1, const T& rho_t, const T& r_tp1, const T& z_tp1) =0;

      T update(const Vector<T>* phi_t, const Vector<T>* phi_tp1, const T& rho_t, const T& r_tp1,
          const T& z_tp1)
      {
        return update(phi_t, phi_tp1, gamma_t, lambda_t, rho_t, r_tp1, z_tp1);
      }

      T update(const Vector<T>* x_t, const Vector<T>* x_tp1, const T& r_tp1)
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

      void persist(const char* f) const
      {
        v->persist(f);
      }

      void resurrect(const char* f)
      {
        v->resurrect(f);
      }

      Vector<T>* weights() const
      {
        return v;
      }

  };

// Prediction problems
  template<typename T>
  class GTDLambda: public GTDLambdaAbstract<T>
  {
    protected:
      typedef GTDLambdaAbstract<T> Base;

    public:
      GTDLambda(const T& alpha_v, const T& alpha_w, const T& gamma_t, const T& lambda_t,
          Trace<T>* e) :
          GTDLambdaAbstract<T>(alpha_v, alpha_w, gamma_t, lambda_t, e)
      {
      }

      T update(const Vector<T>* phi_t, const Vector<T>* phi_tp1, const T& gamma_tp1,
          const T& lambda_tp1, const T& rho_t, const T& r_tp1, const T& z_tp1)
      {
        Base::delta_t = r_tp1 + (T(1) - gamma_tp1) * z_tp1 + gamma_tp1 * Base::v->dot(phi_tp1)
            - Base::v->dot(phi_t);
        Base::e->update(Base::gamma_t * Base::lambda_t, phi_t);
        Base::e->vect()->mapMultiplyToSelf(rho_t);

        // v
        // part 1
        Base::v->addToSelf(Base::alpha_v * Base::delta_t, Base::e->vect());
        // part2
        Base::v->addToSelf(
            -Base::alpha_v * gamma_tp1 * (T(1) - lambda_tp1) * Base::w->dot(Base::e->vect()),
            phi_tp1);

        // w
        // part 2
        Base::w->addToSelf(-Base::alpha_w * Base::w->dot(phi_t), phi_t);
        // part 1
        Base::w->addToSelf(Base::alpha_w * Base::delta_t, Base::e->vect());

        Base::gamma_t = gamma_tp1;
        Base::lambda_t = lambda_tp1;
        return Base::delta_t;
      }

  };

  // Off-policy TD() with a true online equivalence
  template<typename T>
  class GTDLambdaTrue: public GTDLambdaAbstract<T>
  {
    protected:
      typedef GTDLambdaAbstract<T> Base;

      T v_t, v_tp1, v_old, gamma_tp1, lambda_tp1, rho_tm1;
      Trace<T>* e_d;
      Trace<T>* e_w;

    public:
      GTDLambdaTrue(const T& alpha_v, const T& alpha_w, const T& gamma_t, const T& lambda_t,
          Trace<T>* e, Trace<T>* e_d, Trace<T>* e_w) :
          GTDLambdaAbstract<T>(alpha_v, alpha_w, gamma_t, lambda_t, e), v_t(0), v_tp1(0), v_old(0), //
          gamma_tp1(0), lambda_tp1(0), rho_tm1(0), e_d(e_d), e_w(e_w)
      {
      }

      T initialize()
      {
        Base::initialize();
        e_d->clear();
        e_w->clear();
        v_old = 0;
        rho_tm1 = 0;
        return T(0);
      }

      T update(const Vector<T>* phi_t, const Vector<T>* phi_tp1, const T& gamma_tp1,
          const T& lambda_tp1, const T& rho_t, const T& r_tp1, const T& z_tp1)
      {
        v_t = Base::v->dot(phi_t);
        v_tp1 = Base::v->dot(phi_tp1);
        Base::delta_t = r_tp1 + (T(1) - gamma_tp1) * z_tp1 + gamma_tp1 * v_tp1 - v_t;

        // e
        Base::e->update(Base::gamma_t * Base::lambda_t, phi_t, Base::alpha_v * //
            (T(1) - rho_t * Base::gamma_t * Base::lambda_t * Base::e->vect()->dot(phi_t)));
        Base::e->vect()->mapMultiplyToSelf(rho_t);

        // e^{\Delta}
        e_d->update(Base::gamma_t * Base::lambda_t, phi_t);
        e_d->vect()->mapMultiplyToSelf(rho_t);

        // e^w
        e_w->update(rho_tm1 * Base::gamma_t * Base::lambda_t, phi_t, Base::alpha_w * //
            (T(1) - rho_tm1 * Base::gamma_t * Base::lambda_t * e_w->vect()->dot(phi_t)));

        // v
        // part 1
        Base::v->addToSelf((Base::delta_t + v_t - v_old), Base::e->vect());
        // part 2
        Base::v->addToSelf(-Base::alpha_v * rho_t * (v_t - v_old), phi_t);
        // part3
        Base::v->addToSelf(
            -Base::alpha_v * gamma_tp1 * (T(1) - lambda_tp1) * Base::w->dot(e_d->vect()), phi_tp1);

        // w
        // part 2
        Base::w->addToSelf(-Base::alpha_w * Base::w->dot(phi_t), phi_t);
        // part 1
        Base::w->addToSelf(rho_t * Base::delta_t, e_w->vect());

        Base::gamma_t = gamma_tp1;
        Base::lambda_t = lambda_tp1;
        rho_tm1 = rho_t;
        v_old = v_tp1;
        return Base::delta_t;
      }

      void reset()
      {
        Base::reset();
        e_d->clear();
        e_w->clear();
        v_old = 0;
        rho_tm1 = 0;
      }

  };

} // namespace RLLib

#endif /* PREDICTORALGORITHM_H_ */
