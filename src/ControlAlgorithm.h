/*
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
#include "Representation.h"

namespace RLLib
{

// Simple control algorithm
template<class T, class O>
class SarsaControl: public OnPolicyControlLearner<T, O>
{
  protected:
    Policy<T>* acting;
    StateToStateAction<T, O>* toStateAction;
    Sarsa<T>* sarsa;
    SparseVector<T>* xa_t;

  public:
    SarsaControl(Policy<T>* acting, StateToStateAction<T, O>* toStateAction, Sarsa<T>* sarsa) :
        acting(acting), toStateAction(toStateAction), sarsa(sarsa), xa_t(
            new SparseVector<T>(sarsa->dimension()))
    {
    }

    virtual ~SarsaControl()
    {
      delete xa_t;
    }

    const Action& initialize(const DenseVector<O>& x_0)
    {
      sarsa->initialize();
      const Representations<T>& xas_0 = toStateAction->stateActions(x_0);
      const Action& a_0 = Policies::sampleAction(acting, xas_0);
      xa_t->set(xas_0.at(a_0));
      return a_0;
    }

    const Action& step(const DenseVector<O>& x_t, const Action& a_t, const DenseVector<O>& x_tp1,
        const double& r_tp1, const double& z_tp1)
    {
      const Representations<T>& xas_tp1 = toStateAction->stateActions(x_tp1);
      const Action& a_tp1 = Policies::sampleAction(acting, xas_tp1);
      const SparseVector<T>& xa_tp1 = xas_tp1.at(a_tp1);
      sarsa->update(*xa_t, xa_tp1, r_tp1);
      xa_t->set(xa_tp1);
      return a_tp1;
    }

    void reset()
    {
      sarsa->reset();
    }

    const Action& proposeAction(const DenseVector<O>& x)
    {
      return Policies::sampleBestAction(acting, toStateAction->stateActions(x));
    }

    const double computeValueFunction(const DenseVector<O>& x) const
    {
      const Representations<T>& phis = toStateAction->stateActions(x);
      acting->update(phis);
      double v_s = 0;
      // V(s) = \sum_{a \in A} \pi(s,a) * Q(s,a)
      for (ActionList::const_iterator a = toStateAction->getActionList().begin();
          a != toStateAction->getActionList().end(); ++a)
        v_s += acting->pi(**a) * sarsa->predict(phis.at(**a));
      return v_s;
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

template<class T, class O>
class ExpectedSarsaControl: public SarsaControl<T, O>
{
  protected:
    SparseVector<T>* phi_bar_tp1;
    ActionList* actions;
    typedef SarsaControl<T, O> super;
  public:

    ExpectedSarsaControl(Policy<T>* acting, StateToStateAction<T, O>* toStateAction,
        Sarsa<T>* sarsa, ActionList* actions) :
        SarsaControl<T, O>(acting, toStateAction, sarsa), phi_bar_tp1(
            new SparseVector<T>(sarsa->dimension())), actions(actions)
    {
    }
    virtual ~ExpectedSarsaControl()
    {
      delete phi_bar_tp1;
    }

    const Action& step(const DenseVector<O>& x_t, const Action& a_t, const DenseVector<O>& x_tp1,
        const double& r_tp1, const double& z_tp1)
    {
      phi_bar_tp1->clear();
      const Representations<T>& xas_tp1 = super::toStateAction->stateActions(x_tp1);
      const Action& a_tp1 = Policies::sampleAction(super::acting, xas_tp1);
      for (ActionList::const_iterator a = actions->begin(); a != actions->end(); ++a)
      {
        double pi = super::acting->pi(**a);
        if (pi == 0)
        {
          assert((*a)->id() != a_tp1.id());
          continue;
        }
        phi_bar_tp1->addToSelf(pi, xas_tp1.at(**a));
      }

      const SparseVector<T>& xa_tp1 = xas_tp1.at(a_tp1);
      super::sarsa->update(*SarsaControl<T, O>::xa_t, *phi_bar_tp1, r_tp1);
      super::xa_t->set(xa_tp1);
      return a_tp1;
    }

};

// Gradient decent control
template<class T, class O>
class GreedyGQ: public OffPolicyControlLearner<T, O>
{
  private:
    double rho_t;
  protected:
    Policy<T>* target;
    Policy<T>* behavior;
    ActionList* actions;

    StateToStateAction<T, O>* toStateAction;
    GQ<T>* gq;
    SparseVector<T>* phi_t;
    SparseVector<T>* phi_bar_tp1;

  public:
    GreedyGQ(Policy<T>* target, Policy<T>* behavior, ActionList* actions,
        StateToStateAction<T, O>* toStateAction, GQ<T>* gq) :
        rho_t(0), target(target), behavior(behavior), actions(actions), toStateAction(
            toStateAction), gq(gq), phi_t(new SparseVector<T>(gq->dimension())), phi_bar_tp1(
            new SparseVector<T>(gq->dimension()))
    {
    }

    virtual ~GreedyGQ()
    {
      delete phi_t;
      delete phi_bar_tp1;
    }

    const Action& initialize(const DenseVector<O>& x_0)
    {
      gq->initialize();
      const Representations<T>& xas_0 = toStateAction->stateActions(x_0);
      target->update(xas_0);
      const Action& a_0 = Policies::sampleAction(behavior, xas_0);
      phi_t->set(xas_0.at(a_0));
      return a_0;
    }

    virtual double computeRho(const Action& a_t)
    {
      return target->pi(a_t) / behavior->pi(a_t);
    }

    const Action& step(const DenseVector<O>& x_t, const Action& a_t, const DenseVector<O>& x_tp1,
        const double& r_tp1, const double& z_tp1)
    {

      rho_t = computeRho(a_t);

      const Representations<T>& xas_tp1 = toStateAction->stateActions(x_tp1);
      target->update(xas_tp1);
      phi_bar_tp1->clear();
      for (ActionList::const_iterator a = actions->begin(); a != actions->end(); ++a)
      {
        double pi = target->pi(**a);
        if (pi == 0)
          continue;
        phi_bar_tp1->addToSelf(pi, xas_tp1.at(**a));
      }

      gq->update(*phi_t, *phi_bar_tp1, rho_t, r_tp1, z_tp1);
      // Next cycle update the target policy
      target->update(xas_tp1);
      const Action& a_tp1 = Policies::sampleAction(behavior, xas_tp1);
      phi_t->set(xas_tp1.at(a_tp1));
      return a_tp1;
    }

    void reset()
    {
      gq->reset();
    }

    const Action& proposeAction(const DenseVector<O>& x)
    {
      return Policies::sampleBestAction(target, toStateAction->stateActions(x));
    }

    const double computeValueFunction(const DenseVector<O>& x) const
    {
      const Representations<T>& phis = toStateAction->stateActions(x);
      target->update(phis);
      double v_s = 0;
      // V(s) = \sum_{a \in A} \pi(s,a) * Q(s,a)
      for (ActionList::const_iterator a = actions->begin(); a != actions->end(); ++a)
        v_s += target->pi(**a) * gq->predict(phis.at(**a));
      return v_s;
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

template<class T, class O>
class GQOnPolicyControl: public GreedyGQ<T, O>
{
  public:
    GQOnPolicyControl(Policy<T>* acting, ActionList* actions,
        StateToStateAction<T, O>* toStateAction, GQ<T>* gq) :
        GreedyGQ<T, O>(acting, acting, actions, toStateAction, gq)
    {
    }
    virtual ~GQOnPolicyControl()
    {
    }
    virtual double computeRho(const Action& a_t)
    {
      return 1.0;
    }
};

template<class T, class O>
class ActorLambdaOffPolicy: public ActorOffPolicy<T, O>
{
  protected:
    bool initialized;
    double alpha_u, gamma_t, lambda;
    PolicyDistribution<T>* targetPolicy;
    Traces<T>* e;
    SparseVectors<T>* u;

  public:
    ActorLambdaOffPolicy(const double& alpha_u, const double& gamma_t, const double& lambda,
        PolicyDistribution<T>* targetPolicy, Traces<T>* e) :
        initialized(false), alpha_u(alpha_u), gamma_t(gamma_t), lambda(lambda), targetPolicy(
            targetPolicy), e(e), u(targetPolicy->parameters())
    {
    }

    virtual ~ActorLambdaOffPolicy()
    {
    }

  public:
    void initialize()
    {
      e->clear();
      initialized = true;
    }

    void update(const Representations<T>& xas_t, const Action& a_t, double const& rho_t,
        double const& gamma_t, double delta_t)
    {
      assert(initialized);
      const SparseVectors<T>& gradLog = targetPolicy->computeGradLog(xas_t, a_t);
      for (unsigned int i = 0; i < e->dimension(); i++)
      {
        e->at(i)->update(gamma_t * lambda, gradLog[i]);
        e->at(i)->multiplyToSelf(rho_t);
        u->at(i)->addToSelf(alpha_u * delta_t, e->at(i)->vect());
      }

    }

    PolicyDistribution<T>& policy() const
    {
      return *targetPolicy;
    }

    const Action& proposeAction(const Representations<T>& xas)
    {
      return Policies::sampleBestAction(targetPolicy, xas);
    }

    void reset()
    {
      u->clear();
      e->clear();
      initialized = false;
    }

    double pi(const Action& a) const
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

template<class T, class O>
class OffPAC: public OffPolicyControlLearner<T, O>
{
  private:
    double rho_t, delta_t;
  protected:
    Policy<T>* behavior;
    GTDLambda<T>* critic;
    ActorOffPolicy<T, O>* actor;
    StateToStateAction<T, O>* toStateAction;
    Projector<T, O>* projector;
    double gamma_t;
    SparseVector<T>* phi_t;
    SparseVector<T>* phi_tp1;

  public:
    OffPAC(Policy<T>* behavior, GTDLambda<T>* critic, ActorOffPolicy<T, O>* actor,
        StateToStateAction<T, O>* toStateAction, Projector<T, O>* projector, const double& gamma_t) :
        rho_t(0), delta_t(0), behavior(behavior), critic(critic), actor(actor), toStateAction(
            toStateAction), projector(projector), gamma_t(gamma_t), phi_t(
            new SparseVector<T>(projector->dimension())), phi_tp1(
            new SparseVector<T>(projector->dimension()))
    {
    }

    virtual ~OffPAC()
    {
      delete phi_t;
      delete phi_tp1;
    }

    const Action& initialize(const DenseVector<O>& x_0)
    {
      critic->initialize();
      actor->initialize();
      return Policies::sampleAction(behavior, toStateAction->stateActions(x_0));
    }

    const Action& step(const DenseVector<O>& x_t, const Action& a_t, const DenseVector<O>& x_tp1,
        const double& r_tp1, const double& z_tp1)
    {
      phi_t->set(projector->project(x_t));
      phi_tp1->set(projector->project(x_tp1));

      const Representations<T>& xas_t = toStateAction->stateActions(x_t);
      actor->policy().update(xas_t);
      behavior->update(xas_t);
      rho_t = actor->pi(a_t) / behavior->pi(a_t);

      Boundedness::checkValue(rho_t);
      delta_t = critic->update(*phi_t, *phi_tp1, rho_t, gamma_t, r_tp1, z_tp1);
      Boundedness::checkValue(delta_t);
      actor->update(xas_t, a_t, rho_t, gamma_t, delta_t);

      const Representations<T>& xas_tp1 = toStateAction->stateActions(x_tp1);
      return Policies::sampleAction(behavior, xas_tp1);
    }

    void reset()
    {
      critic->reset();
      actor->reset();
    }

    const Action& proposeAction(const DenseVector<O>& x)
    {
      return actor->proposeAction(toStateAction->stateActions(x));
    }

    const double computeValueFunction(const DenseVector<O>& x) const
    {
      return critic->predict(projector->project(x));
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

template<class T, class O>
class Actor: public ActorOnPolicy<T, O>
{
  protected:
    bool initialized;
    double alpha_u;
    PolicyDistribution<T>* policyDistribution;
    SparseVectors<T>* u;

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

    void update(const Representations<T>& xas_t, const Action& a_t, double delta)
    {
      assert(initialized);
      const SparseVectors<T>& gradLog = policyDistribution->computeGradLog(xas_t, a_t);
      for (unsigned int i = 0; i < gradLog.dimension(); i++)
        u->at(i)->addToSelf(alpha_u * delta, gradLog[i]);
    }

    PolicyDistribution<T>& policy() const
    {
      return *policyDistribution;
    }

    const Action& proposeAction(const Representations<T>& xas)
    {
      policy().update(xas);
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

template<class T, class O>
class ActorLambda: public Actor<T, O>
{
  protected:
    typedef Actor<T, O> super;
    double gamma, lambda;
    Traces<T>* e;

  public:
    ActorLambda(const double& alpha_u, const double& gamma, const double& lambda,
        PolicyDistribution<T>* policyDistribution, Traces<T>* e) :
        Actor<T, O>(alpha_u, policyDistribution), gamma(gamma), lambda(lambda), e(e)
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

    void update(const Representations<T>& xas_t, const Action& a_t, double delta)
    {
      assert(super::initialized);
      const SparseVectors<T>& gradLog = super::policy().computeGradLog(xas_t, a_t);
      for (unsigned int i = 0; i < super::u->dimension(); i++)
      {
        e->at(i)->update(gamma * lambda, gradLog[i]);
        super::u->at(i)->addToSelf(super::alpha_u * delta, e->at(i)->vect());
      }
    }

};

template<class T, class O>
class AbstractActorCritic: public OnPolicyControlLearner<T, O>
{
  protected:
    OnPolicyTD<T>* critic;
    ActorOnPolicy<T, O>* actor;
    Projector<T, O>* projector;
    StateToStateAction<T, O>* toStateAction;

  public:
    AbstractActorCritic(OnPolicyTD<T>* critic, ActorOnPolicy<T, O>* actor,
        Projector<T, O>* projector, StateToStateAction<T, O>* toStateAction) :
        critic(critic), actor(actor), projector(projector), toStateAction(toStateAction)
    {
    }

    virtual ~AbstractActorCritic()
    {

    }

  protected:
    virtual double updateCritic(const DenseVector<O>& x_t, const Action& a_t,
        const DenseVector<O>& x_tp1, const double& r_tp1, const double& z_tp1) =0;

    void updateActor(const DenseVector<O>& x_t, const Action& a_t, const double& actorDelta)
    {
      const Representations<T>& xas_t = toStateAction->stateActions(x_t);
      policy().update(xas_t);
      actor->update(xas_t, a_t, actorDelta);
    }

  public:
    PolicyDistribution<T>& policy() const
    {
      return actor->policy();
    }

    void reset()
    {
      critic->reset();
      actor->reset();
    }
    const Action& initialize(const DenseVector<O>& x_0)
    {
      critic->initialize();
      actor->initialize();
      policy().update(toStateAction->stateActions(x_0));
      return policy().sampleAction();
    }

    const Action& proposeAction(const DenseVector<O>& x)
    {
      return actor->proposeAction(toStateAction->stateActions(x));
    }

    const Action& step(const DenseVector<O>& x_t, const Action& a_t, const DenseVector<O>& x_tp1,
        const double& r_tp1, const double& z_tp1)
    {
      // Update critic
      double delta_t = updateCritic(x_t, a_t, x_tp1, r_tp1, z_tp1);
      // Update actor
      updateActor(x_t, a_t, delta_t);
      policy().update(toStateAction->stateActions(x_tp1));
      return policy().sampleAction();
    }

    const double computeValueFunction(const DenseVector<O>& x) const
    {
      return critic->predict(projector->project(x));
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

template<class T, class O>
class ActorCritic: public AbstractActorCritic<T, O>
{
  protected:
    typedef AbstractActorCritic<T, O> super;
    SparseVector<T>* phi_t;
    SparseVector<T>* phi_tp1;
  public:
    ActorCritic(OnPolicyTD<T>* critic, ActorOnPolicy<T, O>* actor, Projector<T, O>* projector,
        StateToStateAction<T, O>* toStateAction) :
        AbstractActorCritic<T, O>(critic, actor, projector, toStateAction), phi_t(
            new SparseVector<T>(projector->dimension())), phi_tp1(
            new SparseVector<T>(projector->dimension()))
    {
    }

    virtual ~ActorCritic()
    {
      delete phi_t;
      delete phi_tp1;
    }

    double updateCritic(const DenseVector<O>& x_t, const Action& a_t, const DenseVector<O>& x_tp1,
        const double& r_tp1, const double& z_tp1)
    {
      phi_t->set(super::projector->project(x_t));
      phi_tp1->set(super::projector->project(x_tp1));
      // Update critic
      return super::critic->update(*phi_t, *phi_tp1, r_tp1);
    }

};

template<class T, class O>
class AverageRewardActorCritic: public AbstractActorCritic<T, O>
{
  protected:
    typedef AbstractActorCritic<T, O> super;
    double alpha_r, averageReward;
    SparseVector<T>* phi_t;
    SparseVector<T>* phi_tp1;

  public:
    AverageRewardActorCritic(OnPolicyTD<T>* critic, ActorOnPolicy<T, O>* actor,
        Projector<T, O>* projector, StateToStateAction<T, O>* toStateAction, double alpha_r) :
        AbstractActorCritic<T, O>(critic, actor, projector, toStateAction), alpha_r(alpha_r), averageReward(
            0), phi_t(new SparseVector<T>(projector->dimension())), phi_tp1(
            new SparseVector<T>(projector->dimension()))
    {
    }

    virtual ~AverageRewardActorCritic()
    {
      delete phi_t;
      delete phi_tp1;
    }

    double updateCritic(const DenseVector<O>& x_t, const Action& a_t, const DenseVector<O>& x_tp1,
        const double& r_tp1, const double& z_tp1)
    {
      phi_t->set(super::projector->project(x_t));
      phi_tp1->set(super::projector->project(x_tp1));
      // Update critic
      double delta_t = super::critic->update(*phi_t, *phi_tp1, r_tp1 - averageReward);
      averageReward += alpha_r * delta_t;
      return delta_t;
    }
};

} // namespace RLLib

#endif /* CONTROLALGORITHM_H_ */
