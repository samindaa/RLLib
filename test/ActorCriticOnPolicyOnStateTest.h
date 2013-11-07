/*
 * ActorCriticOnPolicyOnStateTest.h
 *
 *  Created on: Oct 28, 2013
 *      Author: sam
 */

#ifndef ACTORCRITICONPOLICYONSTATETEST_H_
#define ACTORCRITICONPOLICYONSTATETEST_H_

#include "HeaderTest.h"

RLLIB_TEST(ActorCriticOnPolicyOnStateTest)

class ActorCriticOnPolicyOnStateTest: public ActorCriticOnPolicyOnStateTestBase
{
  protected:
    double gamma;
    double rewardRequired;
    double mu;
    double sigma;

    Environment<float>* problem;
    Projector<double, float>* projector;
    StateToStateAction<double, float>* toStateAction;

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
    ActorOnPolicy<double, float>* actor;

    OnPolicyControlLearner<double, float>* control;
    Simulator<double, float>* sim;

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

class NoStateProblemProjector: public Projector<double, float>
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

    const Vector<double>* project(const Vector<float>* x)
    {
      vec->clear();
      if (x->empty())
        return vec;
      vec->setEntry(0, 1.0);
      return vec;
    }

    const Vector<double>* project(const Vector<float>* x, int h1)
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

class NoStateProblemStateToStateAction: public StateToStateAction<double, float>
{
  protected:
    Projector<double, float>* projector;
    ActionList* actions;
    Representations<double>* phis;

  public:
    NoStateProblemStateToStateAction(Projector<double, float>* projector, ActionList* actions) :
        projector(projector), actions(actions), phis(
            new Representations<double>(projector->dimension(), actions))
    {
    }

    virtual ~NoStateProblemStateToStateAction()
    {
      delete phis;
    }

  public:
    const Representations<double>* stateActions(const Vector<float>* x)
    {
      assert(actions->dimension() == phis->dimension());
      if (x->empty())
      {
        phis->clear();
        return phis;
      }
      for (ActionList::const_iterator a = actions->begin(); a != actions->end(); ++a)
      {
        if (actions->dimension() == 1)
          phis->set(projector->project(x), *a); // projection from whole space
        else
          phis->set(projector->project(x, (*a)->id()), *a);
      }
      return phis;
    }

    const ActionList* getActionList() const
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
