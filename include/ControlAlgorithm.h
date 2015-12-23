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
 * ControlAlgorithm.h
 *
 *  Created on: Aug 25, 2012
 *      Author: sam
 */

#ifndef CONTROLALGORITHM_H_
#define CONTROLALGORITHM_H_

#include "Control.h"
#include "Action.h"
#include "Policy.h"
#include "PredictorAlgorithm.h"
#include "StateToStateAction.h"

namespace RLLib
{

  // Simple control algorithm
  template<typename T>
  class SarsaControl: public OnPolicyControlLearner<T>
  {
    protected:
      Policy<T>* acting;
      StateToStateAction<T>* toStateAction;
      Sarsa<T>* sarsa;
      Vector<T>* xa_t;

    public:
      SarsaControl(Policy<T>* acting, StateToStateAction<T>* toStateAction, Sarsa<T>* sarsa) :
          acting(acting), toStateAction(toStateAction), sarsa(sarsa), xa_t(0)
      {
      }

      virtual ~SarsaControl()
      {
        if (xa_t)
          delete xa_t;
      }

      const Action<T>* initialize(const Vector<T>* x)
      {
        sarsa->initialize();
        const Representations<T>* phi = toStateAction->stateActions(x);
        const Action<T>* a = Policies::sampleAction(acting, phi);
        Vectors<T>::bufferedCopy(phi->at(a), xa_t);
        return a;
      }

      const Action<T>* step(const Vector<T>* x_t, const Action<T>* a_t, const Vector<T>* x_tp1,
          const T& r_tp1, const T& z_tp1)
      {
        (void) x_t;
        (void) a_t;
        (void) z_tp1;
        const Representations<T>* phi_tp1 = toStateAction->stateActions(x_tp1);
        const Action<T>* a_tp1 = Policies::sampleAction(acting, phi_tp1);
        const Vector<T>* xa_tp1 = phi_tp1->at(a_tp1);
        sarsa->update(xa_t, xa_tp1, r_tp1);
        Vectors<T>::bufferedCopy(xa_tp1, xa_t);
        return a_tp1;
      }

      void reset()
      {
        sarsa->reset();
      }

      const Action<T>* proposeAction(const Vector<T>* x)
      {
        return Policies::sampleBestAction(acting, toStateAction->stateActions(x));
      }

      T computeValueFunction(const Vector<T>* x) const
      {
        const Representations<T>* phis = toStateAction->stateActions(x);
        acting->update(phis);
        T v_s = T(0);
        // V(s) = \sum_{a \in A} \pi(s,a) * Q(s,a)
        for (typename Actions<T>::const_iterator a = toStateAction->getActions()->begin();
            a != toStateAction->getActions()->end(); ++a)
          v_s += acting->pi(*a) * sarsa->predict(phis->at(*a));
        return v_s;
      }

      const Predictor<T>* predictor() const
      {
        return sarsa;
      }

      void persist(const char* f) const
      {
        sarsa->persist(f);
      }
      void resurrect(const char* f)
      {
        sarsa->resurrect(f);
      }

  };

  template<typename T>
  class ExpectedSarsaControl: public SarsaControl<T>
  {
    protected:
      Actions<T>* actions;
      VectorPool<T>* pool;
      typedef SarsaControl<T> Base;
    public:

      ExpectedSarsaControl(Policy<T>* acting, StateToStateAction<T>* toStateAction, Sarsa<T>* sarsa,
          Actions<T>* actions) :
          SarsaControl<T>(acting, toStateAction, sarsa), actions(actions), //
          pool(new VectorPool<T>(toStateAction->dimension()))
      {
      }
      virtual ~ExpectedSarsaControl()
      {
        delete pool;
      }

      const Action<T>* step(const Vector<T>* x_t, const Action<T>* a_t, const Vector<T>* x_tp1,
          const T& r_tp1, const T& z_tp1)
      {
        const Representations<T>* phi_tp1 = Base::toStateAction->stateActions(x_tp1);
        const Action<T>* a_tp1 = Policies::sampleAction(Base::acting, phi_tp1);
        Vector<T>* phi_bar_tp1 = pool->newVector(phi_tp1->at(a_tp1));
        phi_bar_tp1->clear();
        for (typename Actions<T>::const_iterator a = actions->begin(); a != actions->end(); ++a)
        {
          T pi = Base::acting->pi(*a);
          if (pi == 0)
          {
            ASSERT((*a)->id() != a_tp1->id());
            continue;
          }
          phi_bar_tp1->addToSelf(pi, phi_tp1->at(*a));
        }

        const Vector<T>* xa_tp1 = phi_tp1->at(a_tp1);
        Base::sarsa->update(SarsaControl<T>::xa_t, phi_bar_tp1, r_tp1);
        Vectors<T>::bufferedCopy(xa_tp1, Base::xa_t);
        pool->releaseAll();
        return a_tp1;
      }

  };

  /**
   * This is an implementation of
   * a linear, gradient-descent version of Watkins's Q($\lambda $).
   * http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node89.html
   * This algorithms does not have convergence guarantees. Use with care.
   */
  template<typename T>
  class Q: public Predictor<T>, public LinearLearner<T>
  {
    protected:
      T alpha, gamma, lambda;
      bool initialized;
      T delta;
      Trace<T>* e;
      StateToStateAction<T>* toStateAction;
      Vector<T>* q;
      Greedy<T>* target;
      Vector<T>* phi_sa_t;

    public:
      Q(const T& alpha, const T& gamma, const T& lambda, Trace<T>* e, Actions<T>* actions,
          StateToStateAction<T>* toStateAction) :
          alpha(alpha), gamma(gamma), lambda(lambda), initialized(false), delta(0.0f), e(e), //
          toStateAction(toStateAction), q(new PVector<T>(e->vect()->dimension())), //
          target(new Greedy<T>(actions, this)), phi_sa_t(0)
      {
      }

      virtual ~Q()
      {
        delete q;
        delete target;
        if (phi_sa_t)
          delete phi_sa_t;
      }

      T update(const Vector<T>* x_t, const Action<T>* a_t, const Vector<T>* x_tp1, const T& r_tp1)
      {
        ASSERT(initialized);
        const Representations<T>* phi_t = toStateAction->stateActions(x_t);
        Vectors<T>::bufferedCopy(phi_t->at(a_t), phi_sa_t);
        target->update(phi_t);
        const Action<T>* at_star = target->sampleBestAction();
        const Representations<T>* phi_tp1 = toStateAction->stateActions(x_tp1);
        target->update(phi_tp1);
        delta = r_tp1 + gamma * target->sampleBestActionValue() - q->dot(phi_sa_t);
        if (a_t->id() == at_star->id())
          e->update(gamma * lambda, phi_sa_t);
        else
        {
          e->clear();
          e->update(T(0), phi_sa_t);
        }
        q->addToSelf(alpha * delta, e->vect());
        return delta;
      }

      T predict(const Vector<T>* x) const
      {
        return q->dot(x);
      }

      T initialize()
      {
        e->clear();
        initialized = true;
        return T(0);
      }

      void reset()
      {
        e->clear();
        q->clear();
        initialized = false;
      }

      Vector<T>* weights() const
      {
        return q;
      }

      void persist(const char* f) const
      {
        q->persist(f);
      }

      void resurrect(const char* f)
      {
        q->resurrect(f);
      }

  };

// Gradient decent control
  template<typename T>
  class QControl: public OffPolicyControlLearner<T>
  {
    protected:
      Policy<T>* behavior;
      StateToStateAction<T>* toStateAction;
      Q<T>* q;

    public:
      QControl(Policy<T>* behavior, StateToStateAction<T>* toStateAction, Q<T>* q) :
          behavior(behavior), toStateAction(toStateAction), q(q)
      {
      }

      virtual ~QControl()
      {
      }

      const Action<T>* initialize(const Vector<T>* x)
      {
        q->initialize();
        const Representations<T>* phi = toStateAction->stateActions(x);
        behavior->update(phi);
        return Policies::sampleAction(behavior, phi);
      }

      void learn(const Vector<T>* x_t, const Action<T>* a_t, const Vector<T>* x_tp1, const T& r_tp1,
          const T& z_tp1)
      {
        q->update(x_t, a_t, x_tp1, r_tp1);
      }

      const Action<T>* step(const Vector<T>* x_t, const Action<T>* a_t, const Vector<T>* x_tp1,
          const T& r_tp1, const T& z_tp1)
      {
        learn(x_t, a_t, x_tp1, r_tp1, z_tp1);
        return Policies::sampleAction(behavior, toStateAction->stateActions(x_tp1));
      }

      void reset()
      {
        q->reset();
      }

      const Action<T>* proposeAction(const Vector<T>* x)
      {
        return Policies::sampleBestAction(behavior, toStateAction->stateActions(x));
      }

      T computeValueFunction(const Vector<T>* x) const
      {
        const Representations<T>* phis = toStateAction->stateActions(x);
        behavior->update(phis); // ?
        T v_s = T(0);
        // V(s) = \sum_{a \in A} \pi(s,a) * Q(s,a)
        for (typename Actions<T>::const_iterator a = toStateAction->getActions()->begin();
            a != toStateAction->getActions()->end(); ++a)
          v_s += behavior->pi(*a) * q->predict(phis->at(*a));
        return v_s;
      }

      const Predictor<T>* predictor() const
      {
        return q;
      }

      void persist(const char* f) const
      {
        q->persist(f);
      }

      void resurrect(const char* f)
      {
        q->resurrect(f);
      }
  };

// Gradient decent control
  template<typename T>
  class GreedyGQ: public OffPolicyControlLearner<T>
  {
    private:
      T rho_t;
    protected:
      Policy<T>* target;
      Policy<T>* behavior;
      Actions<T>* actions;

      StateToStateAction<T>* toStateAction;
      GQ<T>* gq;
      Vector<T>* phi_t;
      Vector<T>* phi_bar_tp1;

    public:
      GreedyGQ(Policy<T>* target, Policy<T>* behavior, Actions<T>* actions,
          StateToStateAction<T>* toStateAction, GQ<T>* gq) :
          rho_t(0), target(target), behavior(behavior), actions(actions), //
          toStateAction(toStateAction), gq(gq), phi_t(0), phi_bar_tp1(0)
      {
      }

      virtual ~GreedyGQ()
      {
        if (phi_t)
          delete phi_t;
        if (phi_t)
          delete phi_bar_tp1;
      }

      const Action<T>* initialize(const Vector<T>* x)
      {
        gq->initialize();
        const Representations<T>* phi = toStateAction->stateActions(x);
        target->update(phi);
        const Action<T>* a = Policies::sampleAction(behavior, phi);
        Vectors<T>::bufferedCopy(phi->at(a), phi_t);
        Vectors<T>::bufferedCopy(phi_t, phi_bar_tp1);
        return a;
      }

      virtual T computeRho(const Action<T>* a_t)
      {
        return target->pi(a_t) / behavior->pi(a_t);
      }

      void learn(const Vector<T>* x_t, const Action<T>* a_t, const Vector<T>* x_tp1, const T& r_tp1,
          const T& z_tp1)
      {
        const Representations<T>* xas_t = toStateAction->stateActions(x_t);
        target->update(xas_t);
        behavior->update(xas_t);
        rho_t = computeRho(a_t);
        Vectors<T>::bufferedCopy(xas_t->at(a_t), phi_t);

        const Representations<T>* xas_tp1 = toStateAction->stateActions(x_tp1);
        target->update(xas_tp1);
        phi_bar_tp1->clear();
        for (typename Actions<T>::const_iterator a = actions->begin(); a != actions->end(); ++a)
        {
          T pi = target->pi(*a);
          if (pi == 0)
            continue;
          phi_bar_tp1->addToSelf(pi, xas_tp1->at(*a));
        }

        gq->update(phi_t, phi_bar_tp1, rho_t, r_tp1, z_tp1);
      }

      const Action<T>* step(const Vector<T>* x_t, const Action<T>* a_t, const Vector<T>* x_tp1,
          const T& r_tp1, const T& z_tp1)
      {
        learn(x_t, a_t, x_tp1, r_tp1, z_tp1);
        return Policies::sampleAction(behavior, toStateAction->stateActions(x_tp1));
      }

      void reset()
      {
        gq->reset();
      }

      const Action<T>* proposeAction(const Vector<T>* x)
      {
        return Policies::sampleBestAction(target, toStateAction->stateActions(x));
      }

      T computeValueFunction(const Vector<T>* x) const
      {
        const Representations<T>* phis = toStateAction->stateActions(x);
        target->update(phis);
        T v_s = T(0);
        // V(s) = \sum_{a \in A} \pi(s,a) * Q(s,a)
        for (typename Actions<T>::const_iterator a = actions->begin(); a != actions->end(); ++a)
          v_s += target->pi(*a) * gq->predict(phis->at(*a));
        return v_s;
      }

      const Predictor<T>* predictor() const
      {
        return gq;
      }

      void persist(const char* f) const
      {
        gq->persist(f);
      }

      void resurrect(const char* f)
      {
        gq->resurrect(f);
      }
  };

  template<typename T>
  class GQOnPolicyControl: public GreedyGQ<T>
  {
    public:
      GQOnPolicyControl(Policy<T>* acting, Actions<T>* actions,
          StateToStateAction<T>* toStateAction, GQ<T>* gq) :
          GreedyGQ<T>(acting, acting, actions, toStateAction, gq)
      {
      }
      virtual ~GQOnPolicyControl()
      {
      }
      virtual T computeRho(const Action<T>* a_t)
      {
        return T(1);
      }
  };

  template<typename T>
  class AbstractActorOffPolicy: public ActorOffPolicy<T>
  {
    protected:
      bool initialized;
      PolicyDistribution<T>* targetPolicy;
      Vectors<T>* u;
    public:
      AbstractActorOffPolicy(PolicyDistribution<T>* targetPolicy) :
          initialized(false), targetPolicy(targetPolicy), u(targetPolicy->parameters())
      {
      }

      virtual ~AbstractActorOffPolicy()
      {
      }

    public:
      void initialize()
      {
        initialized = true;
      }

      PolicyDistribution<T>* policy() const
      {
        return targetPolicy;
      }

      const Action<T>* proposeAction(const Representations<T>* phi)
      {
        return Policies::sampleBestAction(targetPolicy, phi);
      }

      void reset()
      {
        u->clear();
        initialized = false;
      }

      T pi(const Action<T>* a) const
      {
        return targetPolicy->pi(a);
      }

      void persist(const char* f) const
      {
        u->persist(f);
      }
      void resurrect(const char* f)
      {
        u->resurrect(f);
      }

  };

  template<typename T>
  class ActorLambdaOffPolicy: public AbstractActorOffPolicy<T>
  {
    protected:
      T alpha_u, gamma, lambda;
      Traces<T>* e_u;
      typedef AbstractActorOffPolicy<T> Base;
    public:
      ActorLambdaOffPolicy(const T& alpha_u, const T& gamma/*not used*/, const T& lambda,
          PolicyDistribution<T>* targetPolicy, Traces<T>* e) :
          AbstractActorOffPolicy<T>(targetPolicy), alpha_u(alpha_u), gamma(gamma), lambda(lambda), //
          e_u(e)
      {
      }

      virtual ~ActorLambdaOffPolicy()
      {
      }

    public:
      void initialize()
      {
        Base::initialize();
        e_u->clear();
      }

      void update(const Representations<T>* phi_t, const Action<T>* a_t, T const& rho_t,
          const T& delta_t)
      {
        ASSERT(Base::initialized);
        const Vectors<T>* gradLog = Base::targetPolicy->computeGradLog(phi_t, a_t);
        for (int i = 0; i < e_u->dimension(); i++)
        {
          e_u->getEntry(i)->update(lambda, gradLog->getEntry(i));
          e_u->getEntry(i)->vect()->mapMultiplyToSelf(rho_t);
          Base::u->getEntry(i)->addToSelf(alpha_u * delta_t, e_u->getEntry(i)->vect());
        }
      }

      void reset()
      {
        Base::reset();
        e_u->clear();
      }
  };

  template<typename T>
  class OffPAC: public OffPolicyControlLearner<T>
  {
    private:
      T rho_t, delta_t;
    protected:
      Policy<T>* behavior;
      OffPolicyTD<T>* critic;
      ActorOffPolicy<T>* actor;
      StateToStateAction<T>* toStateAction;
      Projector<T>* projector;
      Vector<T>* phi_t;

    public:
      OffPAC(Policy<T>* behavior, OffPolicyTD<T>* critic, ActorOffPolicy<T>* actor,
          StateToStateAction<T>* toStateAction, Projector<T>* projector) :
          rho_t(0), delta_t(0), behavior(behavior), critic(critic), actor(actor), //
          toStateAction(toStateAction), projector(projector), phi_t(0)
      {
      }

      virtual ~OffPAC()
      {
        if (phi_t)
          delete phi_t;
      }

      const Action<T>* initialize(const Vector<T>* x)
      {
        critic->initialize();
        actor->initialize();
        const Representations<T>* phi = toStateAction->stateActions(x);
        const Action<T>* a = Policies::sampleAction(behavior, phi);
        Vectors<T>::bufferedCopy(projector->project(x), phi_t);
        return a;
      }

      void learn(const Vector<T>* x_t, const Action<T>* a_t, const Vector<T>* x_tp1, const T& r_tp1,
          const T& z_tp1)
      {
        const Representations<T>* xas_t = toStateAction->stateActions(x_t);
        actor->policy()->update(xas_t);
        behavior->update(xas_t);
        rho_t = actor->pi(a_t) / behavior->pi(a_t);
        ASSERT(Boundedness::checkValue(rho_t));

        const Vector<T>* phi_tp1 = projector->project(x_tp1);
        delta_t = critic->update(phi_t, phi_tp1, rho_t, r_tp1, z_tp1);
        Vectors<T>::bufferedCopy(phi_tp1, phi_t);
        ASSERT(Boundedness::checkValue(delta_t));
        actor->update(xas_t, a_t, rho_t, delta_t);
      }

      const Action<T>* step(const Vector<T>* x_t, const Action<T>* a_t, const Vector<T>* x_tp1,
          const T& r_tp1, const T& z_tp1)
      {
        learn(x_t, a_t, x_tp1, r_tp1, z_tp1);
        return Policies::sampleAction(behavior, toStateAction->stateActions(x_tp1));
      }

      void reset()
      {
        critic->reset();
        actor->reset();
      }

      const Action<T>* proposeAction(const Vector<T>* x)
      {
        return actor->proposeAction(toStateAction->stateActions(x));
      }

      T computeValueFunction(const Vector<T>* x) const
      {
        return critic->predict(projector->project(x));
      }

      const Predictor<T>* predictor() const
      {
        return critic;
      }

      void persist(const char* f) const
      {
#if !defined(EMBEDDED_MODE)
        std::string fcritic(f);
        fcritic.append(".critic");
        critic->persist(fcritic.c_str());
        std::string factor(f);
        factor.append(".actor");
        actor->persist(factor.c_str());
#endif
      }

      void resurrect(const char* f)
      {
#if !defined(EMBEDDED_MODE)
        std::string fcritic(f);
        fcritic.append(".critic");
        critic->resurrect(fcritic.c_str());
        std::string factor(f);
        factor.append(".actor");
        actor->resurrect(factor.c_str());
#endif
      }
  };

  template<typename T>
  class Actor: public ActorOnPolicy<T>
  {
    protected:
      bool initialized;
      T alpha_u;
      PolicyDistribution<T>* policyDistribution;
      Vectors<T>* u;

    public:
      Actor(const T& alpha_u, PolicyDistribution<T>* policyDistribution) :
          initialized(false), alpha_u(alpha_u), policyDistribution(policyDistribution), //
          u(policyDistribution->parameters())
      {
      }

      void initialize()
      {
        initialized = true;
      }

      void reset()
      {
        u->clear();
        initialized = false;
      }

      void update(const Representations<T>* phi_t, const Action<T>* a_t, const T& delta_t)
      {
        ASSERT(initialized);
        const Vectors<T>* gradLog = policyDistribution->computeGradLog(phi_t, a_t);
        for (int i = 0; i < gradLog->dimension(); i++)
          u->getEntry(i)->addToSelf(alpha_u * delta_t, gradLog->getEntry(i));
      }

      PolicyDistribution<T>* policy() const
      {
        return policyDistribution;
      }

      const Action<T>* proposeAction(const Representations<T>* phi)
      {
        policy()->update(phi);
        return policyDistribution->sampleBestAction();
      }

      void persist(const char* f) const
      {
        u->persist(f);
      }
      void resurrect(const char* f)
      {
        u->resurrect(f);
      }

  };

  template<typename T>
  class ActorLambda: public Actor<T>
  {
    protected:
      typedef Actor<T> Base;
      T gamma, lambda;
      Traces<T>* e;

    public:
      ActorLambda(const T& alpha_u, const T& gamma, const T& lambda,
          PolicyDistribution<T>* policyDistribution, Traces<T>* e) :
          Actor<T>(alpha_u, policyDistribution), gamma(gamma), lambda(lambda), e(e)
      {
        ASSERT(e->dimension() == Base::u->dimension());
      }

      void initialize(const Vector<T>* x)
      {
        Base::initialize(x);
        e->clear();
      }

      void reset()
      {
        Base::reset();
        e->clear();
      }

      void update(const Representations<T>* phi_t, const Action<T>* a_t, T delta)
      {
        ASSERT(Base::initialized);
        const Vectors<T>* gradLog = Base::policy()->computeGradLog(phi_t, a_t);
        for (int i = 0; i < Base::u->dimension(); i++)
        {
          e->getEntry(i)->update(gamma * lambda, gradLog->getEntry(i));
          Base::u->getEntry(i)->addToSelf(Base::alpha_u * delta, e->getEntry(i)->vect());
        }
      }
  };

  template<typename T>
  class ActorNatural: public Actor<T>
  {
    protected:
      typedef Actor<T> Base;
      Vectors<T>* w;
      T alpha_v;
    public:
      ActorNatural(const T& alpha_u, const T& alpha_v, PolicyDistribution<T>* policyDistribution) :
          Actor<T>(alpha_u, policyDistribution), w(new Vectors<T>()), alpha_v(alpha_v)
      {
        for (int i = 0; i < Base::u->dimension(); i++)
          w->push_back(new SVector<T>(Base::u->getEntry(i)->dimension()));
      }

      virtual ~ActorNatural()
      {
        for (typename Vectors<T>::iterator iter = w->begin(); iter != w->end(); ++iter)
          delete *iter;
        delete w;
      }

      void update(const Representations<T>* phi_t, const Action<T>* a_t, T delta)
      {
        ASSERT(Base::initialized);
        const Vectors<T>* gradLog = Base::policy()->computeGradLog(phi_t, a_t);
        T advantageValue = T(0);
        // Calculate the advantage function
        for (int i = 0; i < w->dimension(); i++)
          advantageValue += gradLog->getEntry(i)->dot(w->getEntry(i));
        for (int i = 0; i < w->dimension(); i++)
        {
          // Update the weights of the advantage function
          w->getEntry(i)->addToSelf(alpha_v * (delta - advantageValue), gradLog->getEntry(i));
          // Update the policy parameters
          Base::u->getEntry(i)->addToSelf(Base::alpha_u, w->getEntry(i));
        }
      }

      void reset()
      {
        Base::reset();
        w->clear();
      }

  };

  template<typename T>
  class AbstractActorCritic: public OnPolicyControlLearner<T>
  {
    protected:
      OnPolicyTD<T>* critic;
      ActorOnPolicy<T>* actor;
      Projector<T>* projector;
      StateToStateAction<T>* toStateAction;
      Vector<T>* phi_t;

    public:
      AbstractActorCritic(OnPolicyTD<T>* critic, ActorOnPolicy<T>* actor, Projector<T>* projector,
          StateToStateAction<T>* toStateAction) :
          critic(critic), actor(actor), projector(projector), toStateAction(toStateAction), phi_t(0)
      {
      }

      virtual ~AbstractActorCritic()
      {
        if (phi_t)
          delete phi_t;
      }

    protected:
      virtual T updateCritic(const Vector<T>* x_t, const Action<T>* a_t, const Vector<T>* x_tp1,
          const T& r_tp1, const T& z_tp1) =0;

      void updateActor(const Vector<T>* x_t, const Action<T>* a_t, const T& actorDelta)
      {
        const Representations<T>* phi_t = toStateAction->stateActions(x_t);
        policy()->update(phi_t);
        actor->update(phi_t, a_t, actorDelta);
      }

    public:
      PolicyDistribution<T>* policy() const
      {
        return actor->policy();
      }

      void reset()
      {
        critic->reset();
        actor->reset();
      }
      const Action<T>* initialize(const Vector<T>* x)
      {
        critic->initialize();
        actor->initialize();
        policy()->update(toStateAction->stateActions(x));
        Vectors<T>::bufferedCopy(projector->project(x), phi_t);
        return policy()->sampleAction();
      }

      const Action<T>* proposeAction(const Vector<T>* x)
      {
        return actor->proposeAction(toStateAction->stateActions(x));
      }

      const Action<T>* step(const Vector<T>* x_t, const Action<T>* a_t, const Vector<T>* x_tp1,
          const T& r_tp1, const T& z_tp1)
      {
        // Update critic
        T delta_t = updateCritic(x_t, a_t, x_tp1, r_tp1, z_tp1);
        // Update actor
        updateActor(x_t, a_t, delta_t);
        policy()->update(toStateAction->stateActions(x_tp1));
        return policy()->sampleAction();
      }

      T computeValueFunction(const Vector<T>* x) const
      {
        return critic->predict(projector->project(x));
      }

      const Predictor<T>* predictor() const
      {
        return critic;
      }

      void persist(const char* f) const
      {
#if !defined(EMBEDDED_MODE)
        std::string fcritic(f);
        fcritic.append(".critic");
        critic->persist(fcritic.c_str());
        std::string factor(f);
        factor.append(".actor");
        actor->persist(factor.c_str());
#endif
      }

      void resurrect(const char* f)
      {
#if !defined(EMBEDDED_MODE)
        std::string fcritic(f);
        fcritic.append(".critic");
        critic->resurrect(fcritic.c_str());
        std::string factor(f);
        factor.append(".actor");
        actor->resurrect(factor.c_str());
#endif
      }

  };

  template<typename T>
  class ActorCritic: public AbstractActorCritic<T>
  {
    protected:
      typedef AbstractActorCritic<T> Base;
    public:
      ActorCritic(OnPolicyTD<T>* critic, ActorOnPolicy<T>* actor, Projector<T>* projector,
          StateToStateAction<T>* toStateAction) :
          AbstractActorCritic<T>(critic, actor, projector, toStateAction)
      {
      }

      virtual ~ActorCritic()
      {
      }

      T updateCritic(const Vector<T>* x_t, const Action<T>* a_t, const Vector<T>* x_tp1,
          const T& r_tp1, const T& z_tp1)
      {
        const Vector<T>* phi_tp1 = Base::projector->project(x_tp1);
        // Update critic
        T delta_t = Base::critic->update(Base::phi_t, phi_tp1, r_tp1);
        Vectors<T>::bufferedCopy(phi_tp1, Base::phi_t);
        return delta_t;
      }
  };

  template<typename T>
  class AverageRewardActorCritic: public AbstractActorCritic<T>
  {
    protected:
      typedef AbstractActorCritic<T> Base;
      T alpha_r, averageReward;

    public:
      AverageRewardActorCritic(OnPolicyTD<T>* critic, ActorOnPolicy<T>* actor,
          Projector<T>* projector, StateToStateAction<T>* toStateAction, T alpha_r) :
          AbstractActorCritic<T>(critic, actor, projector, toStateAction), alpha_r(alpha_r), //
          averageReward(0)
      {
      }

      virtual ~AverageRewardActorCritic()
      {
      }

      T updateCritic(const Vector<T>* x_t, const Action<T>* a_t, const Vector<T>* x_tp1,
          const T& r_tp1, const T& z_tp1)
      {
        (void) x_t;
        (void) a_t;
        (void) z_tp1;
        const Vector<T>* phi_tp1 = Base::projector->project(x_tp1);
        // Update critic
        T delta_t = Base::critic->update(Base::phi_t, phi_tp1, r_tp1 - averageReward);
        averageReward += alpha_r * delta_t;
        Vectors<T>::bufferedCopy(phi_tp1, Base::phi_t);
        return delta_t;
      }
  };

} // namespace RLLib

#endif /* CONTROLALGORITHM_H_ */
