/*
 * Policy.h
 *
 *  Created on: Aug 19, 2012
 *      Author: sam
 */

#ifndef POLICY_H_
#define POLICY_H_

#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <iostream>

#include "Vector.h"
#include "Action.h"
#include "Predictor.h"
#include "Representation.h"
#include "Math.h"

using namespace std;
namespace RLLib
{

template<class T>
class Policy
{
  public:
    virtual ~Policy() {}
    virtual void update(const Representations<T>& phis) =0;
    virtual double pi(const Action& a) =0;
    virtual const Action& sampleAction() =0;
    virtual const Action& sampleBestAction() =0;
};

class Policies
{
  public:
    template<class T>
    static const Action& sampleAction(Policy<T>* policy, const Representations<T>& phis)
    {
      policy->update(phis);
      return policy->sampleAction();
    }

    template<class T>
    static const Action& sampleBestAction(Policy<T>* policy, const Representations<T>& phis)
    {
      policy->update(phis);
      return policy->sampleBestAction();
    }

};

// start with discrete action policy
template<class T>
class DiscreteActionPolicy: public virtual Policy<T>
{
  public:
    virtual ~DiscreteActionPolicy() {}

};

template<class T>
class PolicyDistribution: public virtual Policy<T>
{
  public:
    virtual ~PolicyDistribution() {}
    virtual const SparseVectors<T>& computeGradLog(const Representations<T>& phis,
        const Action& action) =0;
    virtual SparseVectors<T>* parameters() const =0;
};

template<class T>
class NormalDistribution: public PolicyDistribution<T>
{
  protected:
    double initialMean, initialStddev, sigma2;
    double mean, stddev, meanStep, stddevStep;
    SparseVector<T>* u_mean;
    SparseVector<T>* u_stddev;
    SparseVector<T>* gradMean;
    SparseVector<T>* gradStddev;
    SparseVector<T>* x;
    ActionList* actions;
    SparseVectors<T>* multiu;
    SparseVectors<T>* multigrad;
    const int defaultAction;
  public:

    NormalDistribution(const double& initialMean, const double& initialStddev,
        const int& nbFeatures, ActionList* actions) :
        initialMean(initialMean), initialStddev(initialStddev), sigma2(0), mean(0), stddev(0), meanStep(
            0), stddevStep(0), u_mean(new SparseVector<T>(nbFeatures)), u_stddev(
            new SparseVector<T>(nbFeatures)), gradMean(new SparseVector<T>(u_mean->dimension())), gradStddev(
            new SparseVector<T>(u_stddev->dimension())), x(new SparseVector<T>(nbFeatures)), actions(
            actions), multiu(new SparseVectors<T>()), multigrad(new SparseVectors<T>()), defaultAction(
            0)
    {
      multiu->push_back(u_mean);
      multiu->push_back(u_stddev);
      multigrad->push_back(gradMean);
      multigrad->push_back(gradStddev);
    }

    virtual ~NormalDistribution()
    {
      delete u_mean;
      delete u_stddev;
      delete gradMean;
      delete gradStddev;
      delete x;
      delete multiu;
      delete multigrad;
    }

  public:
    void update(const Representations<T>& xs_t)
    {
      // N(mu,var) for single action, single representation only
      assert((xs_t.dimension() == 1) && (actions->dimension() == 1));
      x->set(xs_t.at(actions->at(defaultAction)));
      mean = u_mean->dot(*x) + initialMean;
      stddev = exp(u_stddev->dot(*x)) * initialStddev + 10e-8;
      sigma2 = stddev * stddev;
    }

    double pi(const Action& a)
    {
      return Random::gaussianProbability(a.at(defaultAction), mean, stddev);
    }

  public:

    const Action& sampleAction()
    {
      actions->update(defaultAction, defaultAction, Random::nextNormalGaussian() * stddev + mean);
      return actions->at(defaultAction);
    }

    const Action& sampleBestAction()
    {
      return sampleAction();
    }

    virtual void updateStep(const Action& action)
    {
      double a = action.at(defaultAction);
      meanStep = (a - mean) / sigma2;
      stddevStep = (a - mean) * (a - mean) / sigma2 - 1.0;
    }

    const SparseVectors<T>& computeGradLog(const Representations<T>& xs, const Action& action)
    {
      assert((xs.dimension() == 1) && (actions->dimension() == 1));
      updateStep(action);
      x->set(xs.at(actions->at(defaultAction)));
      gradMean->set(*x).multiplyToSelf(meanStep);
      gradStddev->set(*x).multiplyToSelf(stddevStep);
      return *multigrad;
    }

    SparseVectors<T>* parameters() const
    {
      //return u;
      return multiu;
    }
};

template<class T>
class NormalDistributionScaled: public NormalDistribution<T>
{
  public:

    typedef NormalDistribution<T> super;

    NormalDistributionScaled(const double& initialMean, const double& initialStddev,
        const int& nbFeatures, ActionList* actions) :
        NormalDistribution<T>(initialMean, initialStddev, nbFeatures, actions)
    {
    }
    virtual ~NormalDistributionScaled()
    {
    }

    virtual void updateStep(const Action& action)
    {
      double a = action.at(super::defaultAction);
      super::meanStep = (a - super::mean);
      super::stddevStep = (a - super::mean) * (a - super::mean) - super::sigma2;
    }

};

template<class T>
class ScaledPolicyDistribution: public PolicyDistribution<T>
{
  protected:
    ActionList* actions;
    PolicyDistribution<T>* policy;
    Range<T>* policyRange;
    Range<T>* problemRange;
    Action* a_t;

  public:
    ScaledPolicyDistribution(ActionList* actions, PolicyDistribution<T>* policy,
        Range<T>* policyRange, Range<T>* problemRange) :
        actions(actions), policy(policy), policyRange(policyRange), problemRange(problemRange), a_t(
            new Action(0))
    {
      a_t->push_back(0.0);
    }

    virtual ~ScaledPolicyDistribution()
    {
      delete a_t;
    }

  private:
    // From (c, a, min(), max()) to (0, a', -1, 1)
    double normalize(const Range<T>* range, const double& a)
    {
      return (a - range->center()) / (range->length() / 2.0);
    }

    // From (0, a', -1, 1) to (c, a, min(), max())
    double scale(const Range<T>* range, const double& a)
    {
      return (a * range->length() / 2.0) + range->center();
    }

    const Action& problemToPolicy(const double& problemAction)
    {
      double normalizedAction = normalize(problemRange, problemAction);
      double scaledAction = scale(policyRange, normalizedAction);
      a_t->update(0, scaledAction);
      return *a_t;
    }

    const Action& policyToProblem(const double& policyAction)
    {
      double normalizedAction = normalize(policyRange, policyAction);
      double scaledAction = scale(problemRange, normalizedAction);
      a_t->update(0, scaledAction);
      return *a_t;
    }

  public:
    void update(const Representations<T>& phis)
    {
      policy->update(phis);
    }

    double pi(const Action& a)
    {
      return policy->pi(problemToPolicy(a.at(0)));
    }

    const Action& sampleAction()
    {
      actions->update(0, 0, policyToProblem(policy->sampleAction().at(0)).at(0));
      return actions->at(0);
    }

    const Action& sampleBestAction()
    {
      return sampleAction();
    }

    const SparseVectors<T>& computeGradLog(const Representations<T>& phis, const Action& action)
    {
      return policy->computeGradLog(phis, problemToPolicy(action.at(0)));
    }

    SparseVectors<T>* parameters() const
    {
      return policy->parameters();
    }
};

template<class T>
class StochasticPolicy: public virtual DiscreteActionPolicy<T>
{
  protected:
    ActionList* actions;
    DenseVector<double>* distribution;
  public:
    StochasticPolicy(ActionList* actions) :
        actions(actions), distribution(new DenseVector<double>(actions->dimension()))
    {
    }

    virtual ~StochasticPolicy()
    {
      delete distribution;
    }

    double pi(const Action& action)
    {
      return distribution->at(action.id());

    }

    const Action& sampleAction()
    {
      double random = Random::nextDouble();
      double sum = 0;
      for (ActionList::const_iterator a = actions->begin(); a != actions->end(); ++a)
      {
        sum += distribution->at((*a)->id());
        if (sum >= random)
          return **a;
      }
      return actions->at(actions->dimension() - 1);

    }

    const Action& sampleBestAction()
    {
      return sampleAction();
    }
};

template<class T>
class BoltzmannDistribution: public StochasticPolicy<T>, public PolicyDistribution<T>
{
  protected:
    SparseVector<T>* avg;
    SparseVector<T>* grad;
    SparseVector<T>* u;
    SparseVectors<T>* multiu;
    SparseVectors<T>* multigrad;
    typedef StochasticPolicy<T> super;
  public:
    BoltzmannDistribution(const int& numFeatures, ActionList* actions) :
        StochasticPolicy<T>(actions), avg(new SparseVector<T>(numFeatures)), grad(
            new SparseVector<T>(numFeatures)), u(new SparseVector<T>(numFeatures)), multiu(
            new SparseVectors<T>()), multigrad(new SparseVectors<T>())
    {
      // Parameter setting
      multiu->push_back(u);
      multigrad->push_back(grad);
    }

    virtual ~BoltzmannDistribution()
    {
      delete avg;
      delete grad;
      delete u;
      delete multiu;
      delete multigrad;
    }

    void update(const Representations<T>& xas)
    {
      assert(super::actions->dimension() == xas.dimension());
      super::distribution->clear();
      avg->clear();
      double sum = 0;
      // The exponential function may become very large and overflow.
      // Therefore, we multiply top and bottom of the hypothesis by the same
      // constant without changing the output.
      double maxValue = 0;
      for (ActionList::const_iterator a = super::actions->begin(); a != super::actions->end(); ++a)
      {
        double tmp = u->dot(xas.at(**a));
        if (tmp > maxValue)
          maxValue = tmp;
      }

      for (ActionList::const_iterator a = super::actions->begin(); a != super::actions->end(); ++a)
      {
        const int id = (*a)->id();
        super::distribution->at(id) = exp(u->dot(xas.at(id)) - maxValue);
        Boundedness::checkValue(super::distribution->at(id));
        sum += super::distribution->at(id);
        avg->addToSelf(super::distribution->at(id), xas.at(id));
      }

      for (ActionList::const_iterator a = super::actions->begin(); a != super::actions->end(); ++a)
      {
        const int id = (*a)->id();
        super::distribution->at(id) /= sum;
        Boundedness::checkValue(super::distribution->at(id));
      }
      avg->multiplyToSelf(1.0 / sum);
    }

    const SparseVectors<T>& computeGradLog(const Representations<T>& xas, const Action& action)
    {
      grad->set(xas.at(action));
      grad->substractToSelf(*avg);
      return *multigrad;
    }

    SparseVectors<T>* parameters() const
    {
      return multiu;
    }

    double pi(const Action& action)
    {
      return super::pi(action);
    }

    const Action& sampleAction()
    {
      return super::sampleAction();
    }

    const Action& sampleBestAction()
    {
      return super::sampleBestAction();
    }
};

template<class T>
class SoftMax: public StochasticPolicy<T>
{
  protected:
    Predictor<T>* predictor;
    double temperature;
    typedef StochasticPolicy<T> super;
  public:
    SoftMax(Predictor<T>* predictor, ActionList* actions, const double temperature = 1.0) :
        StochasticPolicy<T>(actions), predictor(predictor), temperature(temperature)
    {

    }

    virtual ~SoftMax()
    {
    }

    void update(const Representations<T>& xas)
    {
      assert(super::actions->dimension() == xas.dimension());
      super::distribution->clear();
      double sum = 0;
      // The exponential function may become very large and overflow.
      // Therefore, we multiply top and bottom of the hypothesis by the same
      // constant without changing the output.
      double maxValue = 0;
      for (ActionList::const_iterator a = super::actions->begin(); a != super::actions->end(); ++a)
      {
        double tmp = predictor->predict(xas.at(**a));
        if (tmp > maxValue)
          maxValue = tmp;
      }

      for (ActionList::const_iterator a = super::actions->begin(); a != super::actions->end(); ++a)
      {
        const int id = (*a)->id();
        super::distribution->at(id) = exp(
            (predictor->predict(xas.at(id)) - maxValue) / temperature);
        Boundedness::checkValue(super::distribution->at(id));
        sum += super::distribution->at(id);
      }

      for (ActionList::const_iterator a = super::actions->begin(); a != super::actions->end(); ++a)
      {
        const int id = (*a)->id();
        super::distribution->at(id) /= sum;
        Boundedness::checkValue(super::distribution->at(id));
      }
    }
};

template<class T>
class RandomPolicy: public Policy<T>
{
  protected:
    ActionList* actions;
  public:
    RandomPolicy(ActionList* actions) :
        actions(actions)
    {
    }

    virtual ~RandomPolicy()
    {
    }

    void update(const Representations<T>& xas)
    {
    }
    double pi(const Action& a)
    {
      return 1.0 / actions->dimension();
    }
    const Action& sampleAction()
    {
      return actions->at(rand() % actions->dimension());
    }
    const Action& sampleBestAction()
    {
      assert(false);
      return actions->at(0);
    }
};

template<class T>
class RandomBiasPolicy: public Policy<T>
{
  protected:
    ActionList* actions;
    const Action* prev;
    DenseVector<double>* distribution;
  public:
    RandomBiasPolicy(ActionList* actions) :
        actions(actions), prev(&actions->at(0)), distribution(
            new DenseVector<double>(actions->dimension()))
    {
    }

    virtual ~RandomBiasPolicy()
    {
      delete distribution;
    }

    void update(const Representations<T>& xas)
    {
      // 50% prev action
      distribution->clear();
      if (distribution->dimension() == 1)
        distribution->at(0) = 1.0;
      else
      {
        for (ActionList::const_iterator a = actions->begin(); a != actions->end(); ++a)
        {
          const int id = (*a)->id();
          if (id == prev->id())
            distribution->at(id) = 0.5;
          else
            distribution->at(id) = 0.5 / (actions->dimension() - 1);
        }
      }
      // chose an action
      double random = Random::nextDouble();
      double sum = 0;
      for (ActionList::const_iterator a = actions->begin(); a != actions->end(); ++a)
      {
        sum += distribution->at((*a)->id());
        if (sum >= random)
        {
          prev = *a;
          return;
        }
      }
      prev = &actions->at(actions->dimension() - 1);
    }

    double pi(const Action& action)
    {
      return distribution->at(action.id());
    }

    const Action& sampleAction()
    {
      return *prev;
    }
    const Action& sampleBestAction()
    {
      assert(false);
      return actions->at(0);
    }
};

template<class T>
class Greedy: public DiscreteActionPolicy<T>
{
  protected:
    Predictor<T>* predictor;
    ActionList* actions;
    double* actionValues;
    const Action* bestAction;

  public:
    Greedy(Predictor<T>* predictor, ActionList* actions) :
        predictor(predictor), actions(actions), actionValues(new double[actions->dimension()]), bestAction(
            0)
    {
    }

    virtual ~Greedy()
    {
      delete[] actionValues;
    }

  private:

    void updateActionValues(const Representations<T>& xas_tp1)
    {
      for (ActionList::const_iterator iter = actions->begin(); iter != actions->end(); ++iter)
      {
        const int id = (*iter)->id();
        actionValues[id] = predictor->predict(xas_tp1.at(id));
      }
    }

    void findBestAction()
    {
      bestAction = &actions->at(0);
      for (unsigned int i = 1; i < actions->dimension(); i++)
      {
        if (actionValues[i] > actionValues[bestAction->id()])
          bestAction = &actions->at(i);
      }
    }

  public:

    void update(const Representations<T>& xas_tp1)
    {
      updateActionValues(xas_tp1);
      findBestAction();
    }

    double pi(const Action& a)
    {
      return (bestAction == &a) ? 1.0 : 0;
    }

    const Action& sampleAction()
    {
      return *bestAction;
    }

    const Action& sampleBestAction()
    {
      return *bestAction;
    }

};

template<class T>
class EpsilonGreedy: public Greedy<T>
{
  protected:
    double epsilon;
  public:
    EpsilonGreedy(Predictor<T>* predictor, ActionList* actions, const double& epsilon) :
        Greedy<T>(predictor, actions), epsilon(epsilon)
    {
    }

    const Action& sampleAction()
    {
      if (Random::nextDouble() < epsilon)
        return (*Greedy<T>::actions)[rand() % Greedy<T>::actions->dimension()];
      else
        return *Greedy<T>::bestAction;
    }

    double pi(const Action& a)
    {
      double probability = (a == *Greedy<T>::bestAction) ? 1.0 - epsilon : 0.0;
      return probability + epsilon / Greedy<T>::actions->dimension();
    }

};

} // namespace RLLib

#endif /* POLICY_H_ */
