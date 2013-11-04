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
#include "StateToStateAction.h"
#include "Math.h"

using namespace std;
namespace RLLib
{

template<class T>
class Policy
{
  public:
    virtual ~Policy()
    {
    }
    virtual void update(const Representations<T>* phis) =0;
    virtual double pi(const Action* a) =0;
    virtual const Action* sampleAction() =0;
    virtual const Action* sampleBestAction() =0;
};

class Policies
{
  public:
    template<class T>
    static const Action* sampleAction(Policy<T>* policy, const Representations<T>* phis)
    {
      policy->update(phis);
      return policy->sampleAction();
    }

    template<class T>
    static const Action* sampleBestAction(Policy<T>* policy, const Representations<T>* phis)
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
    virtual ~DiscreteActionPolicy()
    {
    }
};

template<class T>
class PolicyDistribution: public virtual Policy<T>
{
  public:
    virtual ~PolicyDistribution()
    {
    }
    virtual const SparseVectors<T>& computeGradLog(const Representations<T>* phis,
        const Action* action) =0;
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
            0), stddevStep(0), u_mean(new SVector<T>(nbFeatures)), u_stddev(
            new SVector<T>(nbFeatures)), gradMean(new SVector<T>(u_mean->dimension())), gradStddev(
            new SVector<T>(u_stddev->dimension())), x(new SVector<T>(nbFeatures)), actions(actions), multiu(
            new SparseVectors<T>()), multigrad(new SparseVectors<T>()), defaultAction(0)
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
    void update(const Representations<T>* phi)
    {
      // N(mu,var) for single action, single representation only
      assert((phi->dimension() == 1) && (actions->dimension() == 1));
      x->set(phi->at(actions->at(defaultAction)));
      mean = u_mean->dot(x) + initialMean;
      stddev = exp(u_stddev->dot(x)) * initialStddev + 10e-8;
      sigma2 = stddev * stddev;
    }

    double pi(const Action* a)
    {
      return Probabilistic::gaussianProbability(a->at(defaultAction), mean, stddev);
    }

  public:

    const Action* sampleAction()
    {
      actions->update(defaultAction, defaultAction,
          Probabilistic::nextNormalGaussian() * stddev + mean);
      return actions->at(defaultAction);
    }

    const Action* sampleBestAction()
    {
      return sampleAction();
    }

    virtual void updateStep(const Action* action)
    {
      double a = action->at(defaultAction);
      meanStep = (a - mean) / sigma2;
      stddevStep = (a - mean) * (a - mean) / sigma2 - 1.0;
    }

    const SparseVectors<T>& computeGradLog(const Representations<T>* phi, const Action* action)
    {
      assert((phi->dimension() == 1) && (actions->dimension() == 1));
      updateStep(action);
      x->set(phi->at(actions->at(defaultAction)));
      gradMean->set(x)->mapMultiplyToSelf(meanStep);
      gradStddev->set(x)->mapMultiplyToSelf(stddevStep);
      return *multigrad;
    }

    SparseVectors<T>* parameters() const
    {
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

    virtual void updateStep(const Action* action)
    {
      double a = action->at(super::defaultAction);
      super::meanStep = (a - super::mean);
      super::stddevStep = (a - super::mean) * (a - super::mean) - super::sigma2;
    }

};

template<class T>
class NormalDistributionSkewed: public NormalDistribution<T>
{
  public:

    typedef NormalDistribution<T> super;

    NormalDistributionSkewed(const double& initialMean, const double& initialStddev,
        const int& nbFeatures, ActionList* actions) :
        NormalDistribution<T>(initialMean, initialStddev, nbFeatures, actions)
    {
    }

    virtual ~NormalDistributionSkewed()
    {
    }

    virtual void updateStep(const Action* action)
    {
      double a = action->at(super::defaultAction);
      super::meanStep = (a - super::mean);
      super::stddevStep = (a - super::mean) * (a - super::mean) / super::sigma2 - 1.0;
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

    const Action* problemToPolicy(const double& problemAction)
    {
      double normalizedAction = normalize(problemRange, problemAction);
      double scaledAction = scale(policyRange, normalizedAction);
      a_t->update(0, scaledAction);
      return a_t;
    }

    const Action* policyToProblem(const double& policyAction)
    {
      double normalizedAction = normalize(policyRange, policyAction);
      double scaledAction = scale(problemRange, normalizedAction);
      a_t->update(0, scaledAction);
      return a_t;
    }

  public:
    void update(const Representations<T>* phis)
    {
      policy->update(phis);
    }

    double pi(const Action* a)
    {
      return policy->pi(problemToPolicy(a->at(0)));
    }

    const Action* sampleAction()
    {
      actions->update(0, 0, policyToProblem(policy->sampleAction()->at(0))->at(0));
      return actions->at(0);
    }

    const Action* sampleBestAction()
    {
      return sampleAction();
    }

    const SparseVectors<T>& computeGradLog(const Representations<T>* phis, const Action* action)
    {
      return policy->computeGradLog(phis, problemToPolicy(action->at(0)));
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
    PVector<double>* distribution;
  public:
    StochasticPolicy(ActionList* actions) :
        actions(actions), distribution(new PVector<double>(actions->dimension()))
    {
    }

    StochasticPolicy(ActionList* actions, DenseVector<double>* distribution) :
        actions(actions), distribution(new PVector<double>(actions->dimension()))
    {
      assert((int )actions->dimension() == distribution->dimension());
      for (int i = 0; i < distribution->dimension(); i++)
        this->distribution->at(i) = distribution->at(i);
    }

    virtual ~StochasticPolicy()
    {
      delete distribution;
    }

    double pi(const Action* action)
    {
      return distribution->at(action->id());
    }

    const Action* sampleAction()
    {
      Boundedness::checkDistribution(*distribution);
      double random = Probabilistic::nextDouble();
      double sum = 0;
      for (ActionList::const_iterator a = actions->begin(); a != actions->end(); ++a)
      {
        sum += distribution->at((*a)->id());
        if (sum >= random)
          return *a;
      }
      return actions->at(actions->dimension() - 1);
    }

    const Action* sampleBestAction()
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
        StochasticPolicy<T>(actions), avg(new SVector<T>(numFeatures)), grad(
            new SVector<T>(numFeatures)), u(new SVector<T>(numFeatures)), multiu(
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

    void update(const Representations<T>* phi)
    {
      assert(super::actions->dimension() == phi->dimension());
      super::distribution->clear();
      avg->clear();
      double sum = 0;
      // The exponential function may become very large and overflow.
      // Therefore, we multiply top and bottom of the hypothesis by the same
      // constant without changing the output.
      double maxValue = 0;
      for (ActionList::const_iterator a = super::actions->begin(); a != super::actions->end(); ++a)
      {
        double tmp = u->dot(phi->at(*a));
        if (tmp > maxValue)
          maxValue = tmp;
      }

      for (ActionList::const_iterator a = super::actions->begin(); a != super::actions->end(); ++a)
      {
        const int id = (*a)->id();
        super::distribution->at(id) = exp(u->dot(phi->at(*a)) - maxValue);
        Boundedness::checkValue(super::distribution->at(id));
        sum += super::distribution->at(id);
        avg->addToSelf(super::distribution->at(id), phi->at(*a));
      }

      for (ActionList::const_iterator a = super::actions->begin(); a != super::actions->end(); ++a)
      {
        const int id = (*a)->id();
        super::distribution->at(id) /= sum;
        Boundedness::checkValue(super::distribution->at(id));
      }
      avg->mapMultiplyToSelf(1.0 / sum);
    }

    const SparseVectors<T>& computeGradLog(const Representations<T>* phi, const Action* action)
    {
      grad->set(phi->at(action));
      grad->subtractToSelf(avg);
      return *multigrad;
    }

    SparseVectors<T>* parameters() const
    {
      return multiu;
    }

    double pi(const Action* action)
    {
      return super::pi(action);
    }

    const Action* sampleAction()
    {
      return super::sampleAction();
    }

    const Action* sampleBestAction()
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

    void update(const Representations<T>* phi)
    {
      assert(super::actions->dimension() == phi->dimension());
      super::distribution->clear();
      double sum = 0;
      // The exponential function may become very large and overflow.
      // Therefore, we multiply top and bottom of the hypothesis by the same
      // constant without changing the output.
      double maxValue = 0;
      for (ActionList::const_iterator a = super::actions->begin(); a != super::actions->end(); ++a)
      {
        double tmp = predictor->predict(phi->at(*a));
        if (tmp > maxValue)
          maxValue = tmp;
      }

      for (ActionList::const_iterator a = super::actions->begin(); a != super::actions->end(); ++a)
      {
        const int id = (*a)->id();
        super::distribution->at(id) = exp(
            (predictor->predict(phi->at(*a)) - maxValue) / temperature);
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

    void update(const Representations<T>* phi)
    {
    }
    double pi(const Action* a)
    {
      return 1.0 / actions->dimension();
    }
    const Action* sampleAction()
    {
      return actions->at(rand() % actions->dimension());
    }
    const Action* sampleBestAction()
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
    PVector<double>* distribution;
  public:
    RandomBiasPolicy(ActionList* actions) :
        actions(actions), prev(&actions->at(0)), distribution(
            new PVector<double>(actions->dimension()))
    {
    }

    virtual ~RandomBiasPolicy()
    {
      delete distribution;
    }

    void update(const Representations<T>* phi)
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
      double random = Probabilistic::nextDouble();
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

    double pi(const Action* action)
    {
      return distribution->at(action->id());
    }

    const Action* sampleAction()
    {
      return *prev;
    }
    const Action* sampleBestAction()
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

    void updateActionValues(const Representations<T>* phi_tp1)
    {
      for (ActionList::const_iterator iter = actions->begin(); iter != actions->end(); ++iter)
      {
        const int id = (*iter)->id();
        actionValues[id] = predictor->predict(phi_tp1->at(*iter));
      }
    }

    void findBestAction()
    {
      bestAction = actions->at(0);
      for (int i = 1; i < actions->dimension(); i++)
      {
        if (actionValues[i] > actionValues[bestAction->id()])
          bestAction = actions->at(i);
      }
    }

  public:

    void update(const Representations<T>* phi_tp1)
    {
      updateActionValues(phi_tp1);
      findBestAction();
    }

    double pi(const Action* a)
    {
      return (bestAction->id() == a->id()) ? 1.0f : 0.0f;
    }

    const Action* sampleAction()
    {
      return bestAction;
    }

    const Action* sampleBestAction()
    {
      return bestAction;
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

    const Action* sampleAction()
    {
      if (Probabilistic::nextDouble() < epsilon)
        return Greedy<T>::actions->at(rand() % Greedy<T>::actions->dimension());
      else
        return Greedy<T>::bestAction;
    }

    double pi(const Action* a)
    {
      double probability = (a->id() == Greedy<T>::bestAction->id()) ? 1.0 - epsilon : 0.0;
      return probability + epsilon / Greedy<T>::actions->dimension();
    }

};

// Special behavior policies
template<class T>
class BoltzmannDistributionPerturbed: public Policy<T>
{
  protected:
    SparseVector<T>* u;
    ActionList* actions;
    PVector<double>* distribution;
    double epsilon;
    double perturbation;
  public:
    BoltzmannDistributionPerturbed(SparseVector<T>* u, ActionList* actions, const double& epsilon,
        const double& perturbation) :
        u(u), actions(actions), distribution(new PVector<double>(actions->dimension())), epsilon(
            epsilon), perturbation(perturbation)
    {
    }

    virtual ~BoltzmannDistributionPerturbed()
    {
      delete distribution;
    }

    void update(const Representations<T>* phis)
    {
      assert(actions->dimension() == phis->dimension());
      distribution->clear();
      double sum = 0;
      // The exponential function may become very large and overflow.
      // Therefore, we multiply top and bottom of the hypothesis by the same
      // constant without changing the output.
      double maxValue = 0;
      for (ActionList::const_iterator a = actions->begin(); a != actions->end(); ++a)
      {
        double tmp = u->dot(phis->at(*a));
        if (tmp > maxValue)
          maxValue = tmp;
      }

      for (ActionList::const_iterator a = actions->begin(); a != actions->end(); ++a)
      {
        const int id = (*a)->id();
        double perturb = Probabilistic::nextDouble() < epsilon ? perturbation : 0.0f;
        distribution->at(id) = exp(u->dot(phis->at(*a)) + perturb - maxValue);
        Boundedness::checkValue(distribution->at(id));
        sum += distribution->at(id);
      }

      for (ActionList::const_iterator a = actions->begin(); a != actions->end(); ++a)
      {
        const int id = (*a)->id();
        distribution->at(id) /= sum;
        Boundedness::checkValue(distribution->at(id));
      }

    }

    double pi(const Action* action)
    {
      return distribution->at(action->id());
    }

    const Action* sampleAction()
    {
      double random = Probabilistic::nextDouble();
      double sum = 0;
      for (ActionList::const_iterator a = actions->begin(); a != actions->end(); ++a)
      {
        sum += distribution->at((*a)->id());
        if (sum >= random)
          return *a;
      }
      return actions->at(actions->dimension() - 1);
    }

    const Action* sampleBestAction()
    {
      return sampleAction();
    }

};

template<class T>
class SingleActionPolicy: public Policy<T>
{
  private:
    ActionList* actions;
  public:
    SingleActionPolicy(ActionList* actions) :
        actions(actions)
    {
      assert(actions->dimension() == 1);
    }

    void update(const Representations<T>* phis)
    {
    }

    double pi(const Action* a)
    {
      return a->id() == actions->at(0)->id() ? 1.0 : 0.0;
    }

    const Action* sampleAction()
    {
      return actions->at(0);
    }

    const Action* sampleBestAction()
    {
      return sampleAction();
    }
};

template<class T>
class ConstantPolicy: public StochasticPolicy<T>
{
  public:
    ConstantPolicy(ActionList* actions, DenseVector<double>* distribution) :
        StochasticPolicy<T>(actions, distribution)
    {
    }

    virtual ~ConstantPolicy()
    {
    }

    void update(const Representations<T>* phi)
    {
    }
};

} // namespace RLLib

#endif /* POLICY_H_ */
