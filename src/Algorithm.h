/*
 * Algorithm.h
 *
 *  Created on: Aug 19, 2012
 *      Author: sam
 */

#ifndef ALGORITHM_H_
#define ALGORITHM_H_

#include <vector>

#include "Vector.h"
#include "Action.h"
#include "Trace.h"
#include "Policy.h"
#include "Predictor.h"
#include "Control.h"

// Simple predictor algorithms

template<class T>
class TDLambda: Predictor<T>
{
  private:
    double v_t, v_tp1, delta;
    bool initialized;
  protected:
    double alpha, gamma, lambda;
    Trace<T>* e;
    SparseVector<T>* v;
  public:
    TDLambda(const double& alpha, const double& gamma, const double& lambda,
        Trace<T>* e) :
        v_t(0), v_tp1(0), delta(0), initialized(false), alpha(alpha),
            gamma(gamma), lambda(lambda), e(e),
            v(new SparseVector<T>(e->vect().dimension()))
    {
    }
    virtual ~TDLambda()
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

    double update(const SparseVector<T>& phi_t, const SparseVector<T>& phi_tp1,
        double r_tp1)
    {
      assert(initialized);

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
    }

    int dimension() const
    {
      return v->dimension();
    }

    double predict(const SparseVector<T>& phi) const
    {
      return v->dot(phi);
    }
};

template<class T>
class Sarsa: public Predictor<T>
{
  private:
    double v_t, v_tp1, delta; // temporary variables
    bool initialized;
  protected:
    double alpha, gamma, lambda;
    Trace<T>* e;
    SparseVector<T>* v;
  public:
    Sarsa(const double& alpha, const double& gamma, const double& lambda,
        Trace<T>* e) :
        v_t(0), v_tp1(0), delta(0), initialized(false), alpha(alpha),
            gamma(gamma), lambda(lambda), e(e),
            v(new SparseVector<T>(e->vect().dimension()))
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

    double update(const SparseVector<T>& phi_t, const SparseVector<T>& phi_tp1,
        double r_tp1)
    {
      assert(initialized);

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
    }

    int dimension() const
    {
      return v->dimension();
    }

    double predict(const SparseVector<T>& phi_sa) const
    {
      return v->dot(phi_sa);
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
    GQ(const double& alpha_v, const double& alpha_w, const double& beta_tp1,
        const double& lambda_t, Trace<T>* e) :
        delta_t(0), initialized(false), alpha_v(alpha_v), alpha_w(alpha_w),
            beta_tp1(beta_tp1), lambda_t(lambda_t), e(e),
            v(new SparseVector<T>(e->vect().dimension())),
            w(new SparseVector<T>(e->vect().dimension()))
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

    double update(const SparseVector<T>& phi_t,
        const SparseVector<T>& phi_bar_tp1, const double& rho_t, double r_tp1,
        double z_tp1)
    {
      assert(initialized);
      delta_t = r_tp1 + beta_tp1 * z_tp1
          + (1.0 - beta_tp1) * v->dot(phi_bar_tp1) - v->dot(phi_t);
      e->update((1.0 - beta_tp1) * lambda_t * rho_t, phi_t); // paper says beta_t ?
      // v
      // part 1
      v->addToSelf(alpha_v * delta_t, e->vect());
      // part 2
      v->addToSelf(
          -alpha_v * (1.0 - beta_tp1) * (1.0 - lambda_t) * w->dot(e->vect()),
          phi_bar_tp1); // paper says beta_t ?
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
    }

    int dimension() const
    {
      return v->dimension();
    }

    double predict(const SparseVector<T>& phi_sa) const
    {
      return v->dot(phi_sa);
    }

};

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
      const std::vector<SparseVector<T>*>& xas_0 = toStateAction->stateActions(
          x_0);
      const Action& a_0 = acting->decide(xas_0);
      xa_t->set(toStateAction->stateAction(xas_0, a_0));
      return a_0;
    }

    const Action& step(const DenseVector<O>& x_t, const Action& a_t,
        const DenseVector<O>& x_tp1, const double& r_tp1, const double& z_tp1)
    {

      const std::vector<SparseVector<T>*>& xas_tp1 =
          toStateAction->stateActions(x_tp1);
      const Action& a_tp1 = acting->decide(xas_tp1);
      const SparseVector<T>& xa_tp1 = toStateAction->stateAction(xas_tp1,
          a_tp1);
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
      const std::vector<SparseVector<T>*>& xas_tp1 =
          SarsaControl<T, O>::toStateAction->stateActions(x_tp1);

      for (unsigned int action = 0; action < actions->getNumActions(); action++)
      {
        double pi = SarsaControl<T, O>::acting->pi(actions->at(action));
        if (pi == 0) continue;
        phi_bar_tp1->addToSelf(pi,
            SarsaControl<T, O>::toStateAction->stateAction(xas_tp1,
                actions->at(action)));
      }

      const Action& a_tp1 = SarsaControl<T, O>::acting->decide(xas_tp1);
      const SparseVector<T>& xa_tp1 =
          SarsaControl<T, O>::toStateAction->stateAction(xas_tp1, a_tp1);
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
      const std::vector<SparseVector<T>*>& xas_0 = toStateAction->stateActions(
          x_0);
      target->decide(xas_0);
      const Action& a_0 = behavior->decide(xas_0);
      phi_t->set(toStateAction->stateAction(xas_0, a_0));
      return a_0;
    }

    const Action& step(const DenseVector<O>& x_t, const Action& a_t,
        const DenseVector<O>& x_tp1, const double& r_tp1, const double& z_tp1)
    {

      rho_t = target->pi(a_t) / behavior->pi(a_t);

      const std::vector<SparseVector<T>*>& xas_tp1 =
          toStateAction->stateActions(x_tp1);
      target->decide(xas_tp1);
      phi_bar_tp1->clear();
      for (unsigned int action = 0; action < actions->getNumActions(); action++)
      {
        double pi = target->pi(actions->at(action));
        if (pi == 0) continue;
        phi_bar_tp1->addToSelf(pi,
            toStateAction->stateAction(xas_tp1, actions->at(action)));
      }

      gq->update(*phi_t, *phi_bar_tp1, rho_t, r_tp1, z_tp1);
      target->decide(xas_tp1);
      const Action& a_tp1 = behavior->decide(xas_tp1);
      phi_t->set(toStateAction->stateAction(xas_tp1, a_tp1));
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
};

// Prediction problems
template<class T>
class GTDLambda: public Predictor<T>
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
    GTDLambda(const double& alpha_v, const double& alpha_w,
        const double& gamma_t, const double& lambda_t, Trace<T>* e) :
        delta_t(0), initialized(false), alpha_v(alpha_v), alpha_w(alpha_w),
            gamma_t(gamma_t), lambda_t(lambda_t), e(e),
            v(new SparseVector<T>(e->vect().dimension())),
            w(new SparseVector<T>(e->vect().dimension()))
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
        const double& gamma_tp1, const double& lambda_tp1, const double& rho_t,
        double r_tp1, double z_tp1)
    {
      delta_t = r_tp1 + (1.0 - gamma_tp1) * z_tp1 + gamma_tp1 * v->dot(phi_tp1)
          - v->dot(phi_t);
      e->update(gamma_t * lambda_t, phi_t);
      e->multiplyToSelf(rho_t);

      // v
      // part 1
      v->addToSelf(alpha_v * delta_t, e->vect());
      // part2
      v->addToSelf(
          -alpha_v * gamma_tp1 * (1.0 - lambda_tp1) * w->dot(e->vect()),
          phi_tp1);

      // w
      // part 1
      w->addToSelf(alpha_w * delta_t, e->vect());
      // part 2
      w->addToSelf(-alpha_w * w->dot(phi_t), phi_t);

      gamma_t = gamma_tp1;
      lambda_t = lambda_tp1;
      return delta_t;
    }

    double update(const SparseVector<T>& phi_t, const SparseVector<T>& phi_tp1,
        const double& rho_t, const double& gamma_t, double r_tp1, double z_tp1)
    {
      return update(phi_t, phi_tp1, gamma_t, lambda_t, rho_t, r_tp1, z_tp1);
    }

    void reset()
    {
      e->clear();
      v->clear();
      w->clear();
    }

    int dimension() const
    {
      return v->dimension();
    }

    double predict(const SparseVector<T>& phi) const
    {
      return v->dot(phi);
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
            policy(policy), e(e), u(new SparseVector<T>(e->vect().dimension()))
    {
    }

    virtual ~ActorLambdaOffPolicy()
    {
      delete u;
    }

    void initialize()
    {
      e->clear();
      initialized = true;
    }

    void updatePolicy(const std::vector<SparseVector<T>*>& xas)
    {
      policy->update(*u, xas);
    }

    void update(const std::vector<SparseVector<T>*>& xas_t, const Action& a_t,
        double const& rho_t, double const& gamma_t, double delta_t)
    {
      assert(initialized);
      e->update(gamma_t * lambda, policy->computeGradLog(xas_t, a_t));
      e->multiplyToSelf(rho_t);
      u->addToSelf(alpha_u * delta_t, e->vect());

    }
    const Action& proposeAction(const std::vector<SparseVector<T>*>& xas)
    {
      updatePolicy(xas);
      return policy->sampleBestAction();
    }

    void reset()
    {
      u->clear();
      e->clear();
    }

    const PolicyDistribution<T>& getPolicy() const
    {
      return *policy;
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
      const std::vector<SparseVector<T>*>& xas_0 = toStateAction->stateActions(
          x_0);
      actor->updatePolicy(xas_0);
      return behavior->decide(xas_0);
    }

    const Action& step(const DenseVector<O>& x_t, const Action& a_t,
        const DenseVector<O>& x_tp1, const double& r_tp1, const double& z_tp1)
    {
      phi_t->set(projector->project(x_t));
      phi_tp1->set(projector->project(x_tp1));

      const std::vector<SparseVector<T>*>& xas_t = toStateAction->stateActions(
          x_t);
      actor->updatePolicy(xas_t);
      rho_t = actor->getPolicy().pi(a_t) / behavior->pi(a_t);

      delta_t = critic->update(*phi_t, *phi_tp1, rho_t, gamma_t, r_tp1, z_tp1);
      actor->update(xas_t, a_t, rho_t, gamma_t, delta_t);

      const std::vector<SparseVector<T>*>& xas_tp1 =
          toStateAction->stateActions(x_tp1);
      actor->updatePolicy(xas_tp1);
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
};

#endif /* ALGORITHM_H_ */
