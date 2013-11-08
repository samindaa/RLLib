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
 * ControlAlgorithm.h
 *
 *  Created on: Aug 25, 2012
 *      Author: sam
 */

#ifndef CONTROLALGORITHM_H_
#define CONTROLALGORITHM_H_

#include <vector>

#include "Control.h"
#include "Action.h"
#include "Policy.h"
#include "PredictorAlgorithm.h"
#include "StateToStateAction.h"

namespace RLLib
{

// Simple control algorithm
template<class T>
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
        const double& r_tp1, const double& z_tp1)
    {
      const Representations<T>* phi_tp1 = toStateAction->stateActions(x_tp1);
      const Action<T>* a_tp1 = Policies::sampleAction(acting, phi_tp1);
      const Vector<T>* xa_tp1 = phi_tp1->at(a_tp1);
      sarsa->update(xa_t, xa_tp1, r_tp1);
      xa_t->set(xa_tp1);
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

    const double computeValueFunction(const Vector<T>* x) const
    {
      const Representations<T>* phis = toStateAction->stateActions(x);
      acting->update(phis);
      double v_s = 0;
      // V(s) = \sum_{a \in A} \pi(s,a) * Q(s,a)
      for (typename ActionList<T>::const_iterator a = toStateAction->getActionList()->begin();
          a != toStateAction->getActionList()->end(); ++a)
        v_s += acting->pi(*a) * sarsa->predict(phis->at(*a));
      return v_s;
    }

    const Predictor<T>* predictor() const
    {
      return sarsa;
    }

    void persist(const std::string& f) const
    {
      sarsa->persist(f);
    }
    void resurrect(const std::string& f)
    {
      sarsa->resurrect(f);
    }

};

template<class T>
class ExpectedSarsaControl: public SarsaControl<T>
{
  protected:
    Vector<T>* phi_bar_tp1;
    ActionList<T>* actions;
    typedef SarsaControl<T> super;
  public:

    ExpectedSarsaControl(Policy<T>* acting, StateToStateAction<T>* toStateAction, Sarsa<T>* sarsa,
        ActionList<T>* actions) :
        SarsaControl<T>(acting, toStateAction, sarsa), phi_bar_tp1(
            new SVector<T>(toStateAction->dimension())), actions(actions)
    {
    }
    virtual ~ExpectedSarsaControl()
    {
      delete phi_bar_tp1;
    }

    const Action<T>* step(const Vector<T>* x_t, const Action<T>* a_t, const Vector<T>* x_tp1,
        const double& r_tp1, const double& z_tp1)
    {
      phi_bar_tp1->clear();
      const Representations<T>* phi_tp1 = super::toStateAction->stateActions(x_tp1);
      const Action<T>* a_tp1 = Policies::sampleAction(super::acting, phi_tp1);
      for (typename ActionList<T>::const_iterator a = actions->begin(); a != actions->end(); ++a)
      {
        double pi = super::acting->pi(*a);
        if (pi == 0)
        {
          assert((*a)->id() != a_tp1->id());
          continue;
        }
        phi_bar_tp1->addToSelf(pi, phi_tp1->at(*a));
      }

      const Vector<T>* xa_tp1 = phi_tp1->at(a_tp1);
      super::sarsa->update(SarsaControl<T>::xa_t, phi_bar_tp1, r_tp1);
      super::xa_t->set(xa_tp1);
      return a_tp1;
    }

};

// Gradient decent control
template<class T>
class GreedyGQ: public OffPolicyControlLearner<T>
{
  private:
    double rho_t;
  protected:
    Policy<T>* target;
    Policy<T>* behavior;
    ActionList<T>* actions;

    StateToStateAction<T>* toStateAction;
    GQ<T>* gq;
    Vector<T>* phi_t;
    Vector<T>* phi_bar_tp1;

  public:
    GreedyGQ(Policy<T>* target, Policy<T>* behavior, ActionList<T>* actions,
        StateToStateAction<T>* toStateAction, GQ<T>* gq) :
        rho_t(0), target(target), behavior(behavior), actions(actions), toStateAction(
            toStateAction), gq(gq), phi_t(0), phi_bar_tp1(0)
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

    virtual double computeRho(const Action<T>* a_t)
    {
      return target->pi(a_t) / behavior->pi(a_t);
    }

    const Action<T>* step(const Vector<T>* x_t, const Action<T>* a_t, const Vector<T>* x_tp1,
        const double& r_tp1, const double& z_tp1)
    {
      rho_t = computeRho(a_t);

      const Representations<T>* xas_tp1 = toStateAction->stateActions(x_tp1);
      target->update(xas_tp1);
      phi_bar_tp1->clear();
      for (typename ActionList<T>::const_iterator a = actions->begin(); a != actions->end(); ++a)
      {
        double pi = target->pi(*a);
        if (pi == 0)
          continue;
        phi_bar_tp1->addToSelf(pi, xas_tp1->at(*a));
      }

      gq->update(phi_t, phi_bar_tp1, rho_t, r_tp1, z_tp1);
      // Next cycle update the target policy
      target->update(xas_tp1);
      const Action<T>* a_tp1 = Policies::sampleAction(behavior, xas_tp1);
      phi_t->set(xas_tp1->at(a_tp1));
      return a_tp1;
    }

    void reset()
    {
      gq->reset();
    }

    const Action<T>* proposeAction(const Vector<T>* x)
    {
      return Policies::sampleBestAction(target, toStateAction->stateActions(x));
    }

    const double computeValueFunction(const Vector<T>* x) const
    {
      const Representations<T>* phis = toStateAction->stateActions(x);
      target->update(phis);
      double v_s = 0;
      // V(s) = \sum_{a \in A} \pi(s,a) * Q(s,a)
      for (typename ActionList<T>::const_iterator a = actions->begin(); a != actions->end(); ++a)
        v_s += target->pi(*a) * gq->predict(phis->at(*a));
      return v_s;
    }

    const Predictor<T>* predictor() const
    {
      return gq;
    }

    void persist(const std::string& f) const
    {
      gq->persist(f);
    }

    void resurrect(const std::string& f)
    {
      gq->resurrect(f);
    }
};

template<class T>
class GQOnPolicyControl: public GreedyGQ<T>
{
  public:
    GQOnPolicyControl(Policy<T>* acting, ActionList<T>* actions,
        StateToStateAction<T>* toStateAction, GQ<T>* gq) :
        GreedyGQ<T>(acting, acting, actions, toStateAction, gq)
    {
    }
    virtual ~GQOnPolicyControl()
    {
    }
    virtual double computeRho(const Action<T>* a_t)
    {
      return 1.0;
    }
};

template<class T>
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

    double pi(const Action<T>* a) const
    {
      return targetPolicy->pi(a);
    }

    void persist(const std::string& f) const
    {
      u->persist(f);
    }
    void resurrect(const std::string& f)
    {
      u->resurrect(f);
    }

};

template<class T>
class ActorLambdaOffPolicy: public AbstractActorOffPolicy<T>
{
  protected:
    double alpha_u, gamma, lambda;
    Traces<T>* e_u;
    typedef AbstractActorOffPolicy<T> super;
  public:
    ActorLambdaOffPolicy(const double& alpha_u, const double& gamma/*not used*/,
        const double& lambda, PolicyDistribution<T>* targetPolicy, Traces<T>* e) :
        AbstractActorOffPolicy<T>(targetPolicy), alpha_u(alpha_u), gamma(gamma), lambda(lambda), e_u(
            e)
    {
    }

    virtual ~ActorLambdaOffPolicy()
    {
    }

  public:
    void initialize()
    {
      super::initialize();
      e_u->clear();
    }

    void update(const Representations<T>* phi_t, const Action<T>* a_t, double const& rho_t,
        const double& delta_t)
    {
      assert(super::initialized);
      const Vectors<T>& gradLog = super::targetPolicy->computeGradLog(phi_t, a_t);
      for (int i = 0; i < e_u->dimension(); i++)
      {
        e_u->at(i)->update(lambda, gradLog[i]);
        e_u->at(i)->vect()->mapMultiplyToSelf(rho_t);
        super::u->at(i)->addToSelf(alpha_u * delta_t, e_u->at(i)->vect());
      }
    }

    void reset()
    {
      super::reset();
      e_u->clear();
    }
};

template<class T>
class OffPAC: public OffPolicyControlLearner<T>
{
  private:
    double rho_t, delta_t;
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
        rho_t(0), delta_t(0), behavior(behavior), critic(critic), actor(actor), toStateAction(
            toStateAction), projector(projector), phi_t(0)
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
      Vectors<T>::bufferedCopy(phi->at(a), phi_t);
      return a;
    }

    const Action<T>* step(const Vector<T>* x_t, const Action<T>* a_t, const Vector<T>* x_tp1,
        const double& r_tp1, const double& z_tp1)
    {
      const Representations<T>* xas_t = toStateAction->stateActions(x_t);
      actor->policy()->update(xas_t);
      behavior->update(xas_t);
      rho_t = actor->pi(a_t) / behavior->pi(a_t);
      Boundedness::checkValue(rho_t);

      Vectors<T>::bufferedCopy(projector->project(x_t), phi_t);
      const Vector<T>* phi_tp1 = projector->project(x_tp1);
      delta_t = critic->update(phi_t, phi_tp1, rho_t, r_tp1, z_tp1);
      Boundedness::checkValue(delta_t);
      actor->update(xas_t, a_t, rho_t, delta_t);

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

    const double computeValueFunction(const Vector<T>* x) const
    {
      return critic->predict(projector->project(x));
    }

    const Predictor<T>* predictor() const
    {
      return critic;
    }

    void persist(const std::string& f) const
    {
      std::string fcritic(f);
      fcritic.append(".critic");
      critic->persist(fcritic);
      std::string factor(f);
      factor.append(".actor");
      actor->persist(factor);
    }

    void resurrect(const std::string& f)
    {
      std::string fcritic(f);
      fcritic.append(".critic");
      critic->resurrect(fcritic);
      std::string factor(f);
      factor.append(".actor");
      actor->resurrect(factor);
    }
};

template<class T>
class Actor: public ActorOnPolicy<T>
{
  protected:
    bool initialized;
    double alpha_u;
    PolicyDistribution<T>* policyDistribution;
    Vectors<T>* u;

  public:
    Actor(const double& alpha_u, PolicyDistribution<T>* policyDistribution) :
        initialized(false), alpha_u(alpha_u), policyDistribution(policyDistribution), u(
            policyDistribution->parameters())
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

    void update(const Representations<T>* phi_t, const Action<T>* a_t, double delta)
    {
      assert(initialized);
      const Vectors<T>& gradLog = policyDistribution->computeGradLog(phi_t, a_t);
      for (int i = 0; i < gradLog.dimension(); i++)
        u->at(i)->addToSelf(alpha_u * delta, gradLog[i]);
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

    void persist(const std::string& f) const
    {
      u->persist(f);
    }
    void resurrect(const std::string& f)
    {
      u->resurrect(f);
    }

};

template<class T>
class ActorLambda: public Actor<T>
{
  protected:
    typedef Actor<T> super;
    double gamma, lambda;
    Traces<T>* e;

  public:
    ActorLambda(const double& alpha_u, const double& gamma, const double& lambda,
        PolicyDistribution<T>* policyDistribution, Traces<T>* e) :
        Actor<T>(alpha_u, policyDistribution), gamma(gamma), lambda(lambda), e(e)
    {
      assert(e->dimension() == super::u->dimension());
    }

    void initialize()
    {
      super::initialize();
      e->clear();
    }

    void reset()
    {
      super::reset();
      e->clear();
    }

    void update(const Representations<T>* phi_t, const Action<T>* a_t, double delta)
    {
      assert(super::initialized);
      const Vectors<T>& gradLog = super::policy()->computeGradLog(phi_t, a_t);
      for (int i = 0; i < super::u->dimension(); i++)
      {
        e->at(i)->update(gamma * lambda, gradLog[i]);
        super::u->at(i)->addToSelf(super::alpha_u * delta, e->at(i)->vect());
      }
    }
};

template<class T>
class ActorNatural: public Actor<T>
{
  protected:
    typedef Actor<T> super;
    Vectors<T>* w;
    double alpha_v;
  public:
    ActorNatural(const double& alpha_u, const double& alpha_v,
        PolicyDistribution<T>* policyDistribution) :
        Actor<T>(alpha_u, policyDistribution), w(new Vectors<T>()), alpha_v(alpha_v)
    {
      for (int i = 0; i < super::u->dimension(); i++)
        w->push_back(new SVector<T>(super::u->at(i)->dimension()));
    }

    virtual ~ActorNatural()
    {
      for (typename Vectors<T>::iterator iter = w->begin(); iter != w->end(); ++iter)
        delete *iter;
      delete w;
    }

    void update(const Representations<T>* phi_t, const Action<T>* a_t, double delta)
    {
      assert(super::initialized);
      const Vectors<T>& gradLog = super::policy()->computeGradLog(phi_t, a_t);
      double advantageValue = 0;
      // Calculate the advantage function
      for (int i = 0; i < w->dimension(); i++)
        advantageValue += gradLog[i]->dot(w->at(i));
      for (int i = 0; i < w->dimension(); i++)
      {
        // Update the weights of the advantage function
        w->at(i)->addToSelf(alpha_v * (delta - advantageValue), gradLog[i]);
        // Update the policy parameters
        super::u->at(i)->addToSelf(super::alpha_u, w->at(i));
      }
    }

    void reset()
    {
      super::reset();
      w->clear();
    }

};

template<class T>
class AbstractActorCritic: public OnPolicyControlLearner<T>
{
  protected:
    OnPolicyTD<T>* critic;
    ActorOnPolicy<T>* actor;
    Projector<T>* projector;
    StateToStateAction<T>* toStateAction;

  public:
    AbstractActorCritic(OnPolicyTD<T>* critic, ActorOnPolicy<T>* actor, Projector<T>* projector,
        StateToStateAction<T>* toStateAction) :
        critic(critic), actor(actor), projector(projector), toStateAction(toStateAction)
    {
    }

    virtual ~AbstractActorCritic()
    {

    }

  protected:
    virtual double updateCritic(const Vector<T>* x_t, const Action<T>* a_t, const Vector<T>* x_tp1,
        const double& r_tp1, const double& z_tp1) =0;

    void updateActor(const Vector<T>* x_t, const Action<T>* a_t, const double& actorDelta)
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
      return policy()->sampleAction();
    }

    const Action<T>* proposeAction(const Vector<T>* x)
    {
      return actor->proposeAction(toStateAction->stateActions(x));
    }

    const Action<T>* step(const Vector<T>* x_t, const Action<T>* a_t, const Vector<T>* x_tp1,
        const double& r_tp1, const double& z_tp1)
    {
      // Update critic
      double delta_t = updateCritic(x_t, a_t, x_tp1, r_tp1, z_tp1);
      // Update actor
      updateActor(x_t, a_t, delta_t);
      policy()->update(toStateAction->stateActions(x_tp1));
      return policy()->sampleAction();
    }

    const double computeValueFunction(const Vector<T>* x) const
    {
      return critic->predict(projector->project(x));
    }

    const Predictor<T>* predictor() const
    {
      return critic;
    }

    void persist(const std::string& f) const
    {
      std::string fcritic(f);
      fcritic.append(".critic");
      critic->persist(fcritic);
      std::string factor(f);
      factor.append(".actor");
      actor->persist(factor);
    }

    void resurrect(const std::string& f)
    {
      std::string fcritic(f);
      fcritic.append(".critic");
      critic->resurrect(fcritic);
      std::string factor(f);
      factor.append(".actor");
      actor->resurrect(factor);
    }

};

template<class T>
class ActorCritic: public AbstractActorCritic<T>
{
  protected:
    typedef AbstractActorCritic<T> super;
    Vector<T>* phi_t;
  public:
    ActorCritic(OnPolicyTD<T>* critic, ActorOnPolicy<T>* actor, Projector<T>* projector,
        StateToStateAction<T>* toStateAction) :
        AbstractActorCritic<T>(critic, actor, projector, toStateAction), phi_t(0)
    {
    }

    virtual ~ActorCritic()
    {
      if (phi_t)
        delete phi_t;
    }

    double updateCritic(const Vector<T>* x_t, const Action<T>* a_t, const Vector<T>* x_tp1,
        const double& r_tp1, const double& z_tp1)
    {
      Vectors<T>::bufferedCopy(super::projector->project(x_t), phi_t);
      const Vector<T>* phi_tp1 = super::projector->project(x_tp1);
      // Update critic
      double delta_t = super::critic->update(phi_t, phi_tp1, r_tp1);
      Vectors<T>::bufferedCopy(phi_tp1, phi_t);
      return delta_t;
    }
};

template<class T>
class AverageRewardActorCritic: public AbstractActorCritic<T>
{
  protected:
    typedef AbstractActorCritic<T> super;
    double alpha_r, averageReward;
    Vector<T>* phi_t;

  public:
    AverageRewardActorCritic(OnPolicyTD<T>* critic, ActorOnPolicy<T>* actor,
        Projector<T>* projector, StateToStateAction<T>* toStateAction, double alpha_r) :
        AbstractActorCritic<T>(critic, actor, projector, toStateAction), alpha_r(alpha_r), averageReward(
            0), phi_t(0)
    {
    }

    virtual ~AverageRewardActorCritic()
    {
      if (phi_t)
        delete phi_t;
    }

    double updateCritic(const Vector<T>* x_t, const Action<T>* a_t, const Vector<T>* x_tp1,
        const double& r_tp1, const double& z_tp1)
    {
      Vectors<T>::bufferedCopy(super::projector->project(x_t), phi_t);
      const Vector<T>* phi_tp1 = super::projector->project(x_tp1);
      // Update critic
      double delta_t = super::critic->update(phi_t, phi_tp1, r_tp1 - averageReward);
      averageReward += alpha_r * delta_t;
      Vectors<T>::bufferedCopy(phi_tp1, phi_t);
      return delta_t;
    }
};

} // namespace RLLib

#endif /* CONTROLALGORITHM_H_ */
