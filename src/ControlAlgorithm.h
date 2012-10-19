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
    SarsaControl(Policy<T>* acting, StateToStateAction<T, O>* toStateAction,
        Sarsa<T>* sarsa) :
        acting(acting), toStateAction(toStateAction), sarsa(sarsa),
            xa_t(new SparseVector<T>(sarsa->dimension()))
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
      const Action& a_0 = acting->decide(xas_0);
      xa_t->set(xas_0.at(a_0));
      return a_0;
    }

    const Action& step(const DenseVector<O>& x_t, const Action& a_t,
        const DenseVector<O>& x_tp1, const double& r_tp1, const double& z_tp1)
    {
      const Representations<T>& xas_tp1 = toStateAction->stateActions(x_tp1);
      const Action& a_tp1 = acting->decide(xas_tp1);
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
      acting->decide(toStateAction->stateActions(x));
      return acting->sampleBestAction();
    }

    const double computeValueFunction(const DenseVector<O>& x) const
    {
      const Representations<T>& phis = toStateAction->stateActions(x);
      acting->update(phis);
      double v_s = 0;
      // V(s) = \sum_{a \in A} \pi(s,a) * Q(s,a)
      for (ActionList::const_iterator a =
          toStateAction->getActionList().begin();
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
  public:

    ExpectedSarsaControl(Policy<T>* acting,
        StateToStateAction<T, O>* toStateAction, Sarsa<T>* sarsa,
        ActionList* actions) :
        SarsaControl<T, O>(acting, toStateAction, sarsa),
            phi_bar_tp1(new SparseVector<T>(sarsa->dimension())),
            actions(actions)
    {
    }
    virtual ~ExpectedSarsaControl()
    {
      delete phi_bar_tp1;
    }

    const Action& step(const DenseVector<O>& x_t, const Action& a_t,
        const DenseVector<O>& x_tp1, const double& r_tp1, const double& z_tp1)
    {
      phi_bar_tp1->clear();
      const Representations<T>& xas_tp1 =
          SarsaControl<T, O>::toStateAction->stateActions(x_tp1);
      const Action& a_tp1 = SarsaControl<T, O>::acting->decide(xas_tp1);
      for (ActionList::const_iterator a = actions->begin(); a != actions->end();
          ++a)
      {
        double pi = SarsaControl<T, O>::acting->pi(**a);
        if (pi == 0)
        {
          assert(**a != a_tp1);
          continue;
        }
        phi_bar_tp1->addToSelf(pi, xas_tp1.at(**a));
      }

      const SparseVector<T>& xa_tp1 = xas_tp1.at(a_tp1);
      SarsaControl<T, O>::sarsa->update(*SarsaControl<T, O>::xa_t, *phi_bar_tp1,
          r_tp1);
      SarsaControl<T, O>::xa_t->set(xa_tp1);
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
        rho_t(0), target(target), behavior(behavior), actions(actions),
            toStateAction(toStateAction), gq(gq),
            phi_t(new SparseVector<T>(gq->dimension())),
            phi_bar_tp1(new SparseVector<T>(gq->dimension()))
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
      const Action& a_0 = behavior->decide(xas_0);
      phi_t->set(xas_0.at(a_0));
      return a_0;
    }

    const Action& step(const DenseVector<O>& x_t, const Action& a_t,
        const DenseVector<O>& x_tp1, const double& r_tp1, const double& z_tp1)
    {

      rho_t = target->pi(a_t) / behavior->pi(a_t);

      const Representations<T>& xas_tp1 = toStateAction->stateActions(x_tp1);
      target->update(xas_tp1);
      phi_bar_tp1->clear();
      for (ActionList::const_iterator a = actions->begin(); a != actions->end();
          ++a)
      {
        double pi = target->pi(**a);
        if (pi == 0)
          continue;
        phi_bar_tp1->addToSelf(pi, xas_tp1.at(**a));
      }

      gq->update(*phi_t, *phi_bar_tp1, rho_t, r_tp1, z_tp1);
      // Next cycle update the target policy
      target->update(xas_tp1);
      const Action& a_tp1 = behavior->decide(xas_tp1);
      phi_t->set(xas_tp1.at(a_tp1));
      return a_tp1;
    }

    void reset()
    {
      gq->reset();
    }

    const Action& proposeAction(const DenseVector<O>& x)
    {
      target->decide(toStateAction->stateActions(x));
      return target->sampleBestAction();
    }

    const double computeValueFunction(const DenseVector<O>& x) const
    {
      const Representations<T>& phis = toStateAction->stateActions(x);
      target->update(phis);
      double v_s = 0;
      // V(s) = \sum_{a \in A} \pi(s,a) * Q(s,a)
      for (ActionList::const_iterator a = actions->begin(); a != actions->end();
          ++a)
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
class ActorLambdaOffPolicy: public ActorOffPolicy<T, O>
{
  protected:
    bool initialized;
    double alpha_u, gamma_t, lambda;
    PolicyDistribution<T>* policy;
    Trace<T>* e;
    SparseVector<T>* u;

  public:
    ActorLambdaOffPolicy(const double& alpha_u, const double& gamma_t,
        const double& lambda, PolicyDistribution<T>* policy, Trace<T>* e) :
        initialized(false), alpha_u(alpha_u), gamma_t(gamma_t), lambda(lambda),
            policy(policy), e(e), u(policy->parameters())
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

    void update(const Representations<T>& xas_t, const Action& a_t,
        double const& rho_t, double const& gamma_t, double delta_t)
    {
      assert(initialized);
      e->update(gamma_t * lambda, policy->computeGradLog(xas_t, a_t));
      e->multiplyToSelf(rho_t);
      u->addToSelf(alpha_u * delta_t, e->vect());
    }

    void updatePolicy(const Representations<T>& xas)
    {
      policy->update(xas);
    }

    const Action& proposeAction(const Representations<T>& xas)
    {
      policy->update(xas);
      return policy->sampleBestAction();
    }

    void reset()
    {
      u->clear();
      e->clear();
      initialized = false;
    }

    double pi(const Action& a) const
    {
      return policy->pi(a);
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
    OffPAC(Policy<T>* behavior, GTDLambda<T>* critic,
        ActorOffPolicy<T, O>* actor, StateToStateAction<T, O>* toStateAction,
        Projector<T, O>* projector, const double& gamma_t) :
        rho_t(0), delta_t(0), behavior(behavior), critic(critic), actor(actor),
            toStateAction(toStateAction), projector(projector),
            gamma_t(gamma_t),
            phi_t(new SparseVector<T>(projector->dimension())),
            phi_tp1(new SparseVector<T>(projector->dimension()))
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
      return behavior->decide(toStateAction->stateActions(x_0));
    }

    const Action& step(const DenseVector<O>& x_t, const Action& a_t,
        const DenseVector<O>& x_tp1, const double& r_tp1, const double& z_tp1)
    {
      phi_t->set(projector->project(x_t));
      phi_tp1->set(projector->project(x_tp1));

      const Representations<T>& xas_t = toStateAction->stateActions(x_t);
      actor->updatePolicy(xas_t);
      behavior->update(xas_t);
      rho_t = actor->pi(a_t) / behavior->pi(a_t);

      Boundedness::checkValue(rho_t);
      delta_t = critic->update(*phi_t, *phi_tp1, rho_t, gamma_t, r_tp1, z_tp1);
      Boundedness::checkValue(delta_t);
      actor->update(xas_t, a_t, rho_t, gamma_t, delta_t);

      const Representations<T>& xas_tp1 = toStateAction->stateActions(x_tp1);
      return behavior->decide(xas_tp1);
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
    double alpha_u, gamma, lambda;
    PolicyDistribution<T>* policy;
    Trace<T>* e;
    SparseVector<T>* u;

  public:
    Actor(const double& alpha_u, const double& gamma, const double& lambda,
        PolicyDistribution<T>* policy, Trace<T>* e) :
        initialized(false), alpha_u(alpha_u), gamma(gamma), lambda(lambda),
            policy(policy), e(e), u(policy->parameters())
    {
      assert(e->vect().dimension() == u->dimension());
    }

    void initialize()
    {
      e->clear();
      initialized = true;
    }

    void reset()
    {
      u->clear();
      e->clear();
      initialized = false;
    }

    void update(const Representations<T>& xas_t, const Action& a_t,
        double delta_t)
    {
      assert(initialized);
      e->update(gamma * lambda, policy->computeGradLog(xas_t, a_t));
      u->addToSelf(alpha_u * delta_t, e->vect());
    }

    void updatePolicy(const Representations<T>& xas)
    {
      policy->update(xas);
    }

    const Action& proposeAction(const Representations<T>& xas)
    {
      updatePolicy(xas);
      return policy->sampleBestAction();
    }

    const Action& decide(const Representations<T>& xas)
    {
      updatePolicy(xas);
      return policy->decide(xas);
    }
};

template<class T, class O>
class AverageRewardActorCritic: public OnPolicyControlLearner<T, O>
{
  private:
    double delta_t;
  protected:
    TDLambda<T>* critic;
    ActorOnPolicy<T, O>* actor;
    double alpha_r, averageReward;
    StateToStateAction<T, O>* toStateAction;
    SparseVector<T>* phi_t;
    SparseVector<T>* phi_tp1;

  public:
    AverageRewardActorCritic(TDLambda<T>* critic, ActorOnPolicy<T, O>* actor,
        StateToStateAction<T, O>* toStateAction, double alpha_r) :
        delta_t(0), critic(critic), actor(actor), alpha_r(alpha_r),
            averageReward(0), toStateAction(toStateAction),
            phi_t(new SparseVector<T>(toStateAction->getProjector().dimension())),
            phi_tp1(new SparseVector<T>(toStateAction->getProjector().dimension()))
    {
    }

    virtual ~AverageRewardActorCritic()
    {
      delete phi_t;
      delete phi_tp1;
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
      return actor->decide(toStateAction->stateActions(x_0));
    }

    const Action& proposeAction(const DenseVector<O>& x)
    {
      return actor->proposeAction(toStateAction->stateActions(x));
    }

    const Action& step(const DenseVector<O>& x_t, const Action& a_t,
        const DenseVector<O>& x_tp1, const double& r_tp1, const double& z_tp1)
    {
      phi_t->set(toStateAction->stateAction(x_t));
      phi_tp1->set(toStateAction->stateAction(x_tp1));
      // update critic
      delta_t = critic->update(*phi_t, *phi_tp1, r_tp1 - averageReward);
      averageReward += alpha_r * delta_t;
      // update actor
      const Representations<T>& xas_t = toStateAction->stateActions(x_t);
      actor->updatePolicy(xas_t);
      actor->update(xas_t, a_t, delta_t);
      return actor->decide(toStateAction->stateActions(x_tp1));
    }

    const double computeValueFunction(const DenseVector<O>& x) const
    {
      return critic->predict(
          toStateAction->stateActions(x).at(
              toStateAction->getActionList().at(0)));
    }

    void persist(const std::string& f) const
    {
      //@@>> TODO:
    }
    void resurrect(const std::string& f)
    {
      //@@>> TODO:
    }

};

} // namespace RLLib

#endif /* CONTROLALGORITHM_H_ */
