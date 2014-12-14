/*
 * ActorCriticOnPolicyOnStateTest.h
 *
 *  Created on: Oct 28, 2013
 *      Author: sam
 */

#ifndef ACTORCRITICONPOLICYONSTATETEST_H_
#define ACTORCRITICONPOLICYONSTATETEST_H_

#include "Test.h"

RLLIB_TEST(ActorCriticOnPolicyOnStateTest)

class ActorCriticOnPolicyOnStateTest: public ActorCriticOnPolicyOnStateTestBase
{
  protected:
    double gamma;
    double rewardRequired;
    double mu;
    double sigma;

    Random<double>* random;
    RLProblem<double>* problem;
    Projector<double>* projector;
    StateToStateAction<double>* toStateAction;

    double alpha_v;
    double alpha_u;
    double alpha_r;
    double lambda;

    Trace<double>* criticE;
    OnPolicyTD<double>* critic;

    PolicyDistribution<double>* policyDistribution;

    Trace<double>* actorMuE;
    Trace<double>* actorSigmaE;
    Traces<double>* actorTraces;
    ActorOnPolicy<double>* actor;

    OnPolicyControlLearner<double>* control;
    RLAgent<double>* agent;
    RLRunner<double>* sim;

  public:
    ActorCriticOnPolicyOnStateTest();
    ~ActorCriticOnPolicyOnStateTest();

    void run();

  protected:
    void checkDistribution(PolicyDistribution<double>* policyDistribution);
    void testNormalDistribution();
    void testNormalDistributionMeanAdjusted();
    void testNormalDistributionWithEligibility();

    void deleteObjects();
};

class NoStateProblemProjector: public Projector<double>
{
  protected:
    Vector<double>* vec;
  public:
    NoStateProblemProjector() :
        vec(new SVector<double>(1, 1))
    {
    }

    ~NoStateProblemProjector()
    {
      delete vec;
    }

    const Vector<double>* project(const Vector<double>* x)
    {
      vec->clear();
      if (x->empty())
        return vec;
      vec->setEntry(0, 1.0);
      return vec;
    }

    const Vector<double>* project(const Vector<double>* x, const int& h1)
    {
      return project(x);
    }

    double vectorNorm() const
    {
      return 1.0;
    }

    int dimension() const
    {
      return 1;
    }
};

class NoStateProblemStateToStateAction: public StateToStateAction<double>
{
  protected:
    Projector<double>* projector;
    Actions<double>* actions;
    Representations<double>* phis;

  public:
    NoStateProblemStateToStateAction(Projector<double>* projector, Actions<double>* actions) :
        projector(projector), actions(actions), phis(
            new Representations<double>(projector->dimension(), actions))
    {
    }

    virtual ~NoStateProblemStateToStateAction()
    {
      delete phis;
    }

  public:
    const Vector<double>* stateAction(const Vector<double>* x, const Action<double>* a)
    {
      if (actions->dimension() == 1)
        return projector->project(x); // projection from whole space
      else
        return projector->project(x, a->id());
    }

    const Representations<double>* stateActions(const Vector<double>* x)
    {
      ASSERT(actions->dimension() == phis->dimension());
      if (x->empty())
      {
        phis->clear();
        return phis;
      }
      for (Actions<double>::const_iterator a = actions->begin(); a != actions->end(); ++a)
        phis->set(stateAction(x, *a), *a);
      return phis;
    }

    const Actions<double>* getActions() const
    {
      return actions;
    }

    double vectorNorm() const
    {
      return projector->vectorNorm();
    }

    int dimension() const
    {
      return projector->dimension();
    }
};

#endif /* ACTORCRITICONPOLICYONSTATETEST_H_ */
