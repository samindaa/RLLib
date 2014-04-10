/*
 * Copyright 2014 Saminda Abeyruwan (saminda@cs.miami.edu)
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

#include "Vector.h"
#include "Action.h"
#include "Predictor.h"
#include "StateToStateAction.h"
#include "Math.h"

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
    virtual T pi(const Action<T>* a) =0;
    virtual const Action<T>* sampleAction() =0;
    virtual const Action<T>* sampleBestAction() =0;
};

class Policies
{
  public:
    template<class T>
    static const Action<T>* sampleAction(Policy<T>* policy, const Representations<T>* phis)
    {
      policy->update(phis);
      return policy->sampleAction();
    }

    template<class T>
    static const Action<T>* sampleBestAction(Policy<T>* policy, const Representations<T>* phis)
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
    virtual const Vectors<T>& computeGradLog(const Representations<T>* phis,
        const Action<T>* action) =0;
    virtual Vectors<T>* parameters() const =0;
};

template<class T>
class NormalDistribution: public PolicyDistribution<T>
{
  protected:
    Random<T>* random;
    Actions<T>* actions;
    T initialMean, initialStddev, sigma2;
    T mean, stddev, meanStep, stddevStep;
    Vector<T>* u_mean;
    Vector<T>* u_stddev;
    Vector<T>* gradMean;
    Vector<T>* gradStddev;
    Vector<T>* x;
    Vectors<T>* multiu;
    Vectors<T>* multigrad;
    const int defaultAction;

  public:

    NormalDistribution(Random<T>* random, Actions<T>* actions, const T& initialMean,
        const T& initialStddev, const int& nbFeatures) :
        random(random), actions(actions), initialMean(initialMean), initialStddev(initialStddev), sigma2(
            0), mean(0), stddev(0), meanStep(0), stddevStep(0), u_mean(new PVector<T>(nbFeatures)), u_stddev(
            new PVector<T>(nbFeatures)), gradMean(new SVector<T>(u_mean->dimension())), gradStddev(
            new SVector<T>(u_stddev->dimension())), x(new SVector<T>(nbFeatures)), multiu(
            new Vectors<T>()), multigrad(new Vectors<T>()), defaultAction(0)
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
      ASSERT((phi->dimension() == 1) && (actions->dimension() == 1));
      x->set(phi->at(actions->at(defaultAction)));
      mean = u_mean->dot(x) + initialMean;
      stddev = exp(u_stddev->dot(x)) * initialStddev + 10e-8;
      Boundedness::checkValue(stddev);
      sigma2 = stddev * stddev;
      Boundedness::checkValue(sigma2);
    }

    T pi(const Action<T>* a)
    {
      return random->gaussianProbability(a->at(defaultAction), mean, stddev);
    }

  public:

    const Action<T>* sampleAction()
    {
      actions->update(defaultAction, defaultAction, random->nextNormalGaussian() * stddev + mean);
      return actions->at(defaultAction);
    }

    const Action<T>* sampleBestAction()
    {
      return sampleAction();
    }

    virtual void updateStep(const Action<T>* action)
    {
      T a = action->at(defaultAction);
      meanStep = (a - mean) / sigma2;
      stddevStep = (a - mean) * (a - mean) / sigma2 - 1.0;
    }

    const Vectors<T>& computeGradLog(const Representations<T>* phi, const Action<T>* action)
    {
      ASSERT((phi->dimension() == 1) && (actions->dimension() == 1));
      updateStep(action);
      x->set(phi->at(actions->at(defaultAction)));
      gradMean->set(x)->mapMultiplyToSelf(meanStep);
      gradStddev->set(x)->mapMultiplyToSelf(stddevStep);
      return *multigrad;
    }

    Vectors<T>* parameters() const
    {
      return multiu;
    }
};

template<class T>
class NormalDistributionScaled: public NormalDistribution<T>
{
  public:

    typedef NormalDistribution<T> Base;

    NormalDistributionScaled(Random<T>* random, Actions<T>* actions, const T& initialMean,
        const T& initialStddev, const int& nbFeatures) :
        NormalDistribution<T>(random, actions, initialMean, initialStddev, nbFeatures)
    {
    }

    virtual ~NormalDistributionScaled()
    {
    }

    virtual void updateStep(const Action<T>* action)
    {
      T a = action->at(Base::defaultAction);
      Base::meanStep = (a - Base::mean);
      Base::stddevStep = (a - Base::mean) * (a - Base::mean) - Base::sigma2;
    }

};

template<class T>
class NormalDistributionSkewed: public NormalDistribution<T>
{
  public:

    typedef NormalDistribution<T> Base;

    NormalDistributionSkewed(Random<T>* random, Actions<T>* actions, const T& initialMean,
        const T& initialStddev, const int& nbFeatures) :
        NormalDistribution<T>(random, actions, initialMean, initialStddev, nbFeatures)
    {
    }

    virtual ~NormalDistributionSkewed()
    {
    }

    virtual void updateStep(const Action<T>* action)
    {
      T a = action->at(Base::defaultAction);
      Base::meanStep = (a - Base::mean);
      Base::stddevStep = (a - Base::mean) * (a - Base::mean) / Base::sigma2 - 1.0;
    }

};

template<class T>
class ScaledPolicyDistribution: public PolicyDistribution<T>
{
  protected:
    Actions<T>* actions;
    PolicyDistribution<T>* policy;
    Range<T>* policyRange;
    Range<T>* problemRange;
    Action<T>* a_t;

  public:
    ScaledPolicyDistribution(Actions<T>* actions, PolicyDistribution<T>* policy,
        Range<T>* policyRange, Range<T>* problemRange) :
        actions(actions), policy(policy), policyRange(policyRange), problemRange(problemRange), a_t(
            new Action<T>(0))
    {
      a_t->push_back(0.0);
    }

    virtual ~ScaledPolicyDistribution()
    {
      delete a_t;
    }

  private:
    // From (c, a, min(), max()) to (0, a', -1, 1)
    T normalize(const Range<T>* range, const T& a)
    {
      return (a - range->center()) / (range->length() / 2.0);
    }

    // From (0, a', -1, 1) to (c, a, min(), max())
    T scale(const Range<T>* range, const T& a)
    {
      return (a * range->length() / 2.0) + range->center();
    }

    const Action<T>* problemToPolicy(const T& problemAction)
    {
      T normalizedAction = normalize(problemRange, problemAction);
      T scaledAction = scale(policyRange, normalizedAction);
      a_t->update(0, scaledAction);
      return a_t;
    }

    const Action<T>* policyToProblem(const T& policyAction)
    {
      T normalizedAction = normalize(policyRange, policyAction);
      T scaledAction = scale(problemRange, normalizedAction);
      a_t->update(0, scaledAction);
      return a_t;
    }

  public:
    void update(const Representations<T>* phis)
    {
      policy->update(phis);
    }

    T pi(const Action<T>* a)
    {
      return policy->pi(problemToPolicy(a->at(0)));
    }

    const Action<T>* sampleAction()
    {
      actions->update(0, 0, policyToProblem(policy->sampleAction()->at(0))->at(0));
      return actions->at(0);
    }

    const Action<T>* sampleBestAction()
    {
      return sampleAction();
    }

    const Vectors<T>& computeGradLog(const Representations<T>* phis, const Action<T>* action)
    {
      return policy->computeGradLog(phis, problemToPolicy(action->at(0)));
    }

    Vectors<T>* parameters() const
    {
      return policy->parameters();
    }
};

template<class T>
class StochasticPolicy: public virtual DiscreteActionPolicy<T>
{
  protected:
    Random<T>* random;
    Actions<T>* actions;
    PVector<T>* distribution;
  public:
    StochasticPolicy(Random<T>* random, Actions<T>* actions) :
        random(random), actions(actions), distribution(new PVector<T>(actions->dimension()))
    {
    }

    virtual ~StochasticPolicy()
    {
      delete distribution;
    }

    T pi(const Action<T>* action)
    {
      return distribution->at(action->id());
    }

    const Action<T>* sampleAction()
    {
      Boundedness::checkDistribution(distribution);
      T rand = random->nextReal();
      T sum = T(0);
      for (typename Actions<T>::const_iterator a = actions->begin(); a != actions->end(); ++a)
      {
        sum += distribution->at((*a)->id());
        if (sum >= rand)
          return *a;
      }
      return actions->at(actions->dimension() - 1);
    }

    const Action<T>* sampleBestAction()
    {
      return sampleAction();
    }
};

template<class T>
class BoltzmannDistribution: public StochasticPolicy<T>, public PolicyDistribution<T>
{
  protected:
    Vector<T>* avg;
    Vector<T>* grad;
    Vector<T>* u;
    Vectors<T>* multiu;
    Vectors<T>* multigrad;
    typedef StochasticPolicy<T> Base;
  public:
    BoltzmannDistribution(Random<T>* random, Actions<T>* actions, const int& numFeatures) :
        StochasticPolicy<T>(random, actions), avg(new SVector<T>(numFeatures)), grad(
            new SVector<T>(numFeatures)), u(new PVector<T>(numFeatures)), multiu(new Vectors<T>()), multigrad(
            new Vectors<T>())
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
      ASSERT(Base::actions->dimension() == phi->dimension());
      Base::distribution->clear();
      avg->clear();
      T sum = T(0);
      // The exponential function may become very large and overflow.
      // Therefore, we multiply top and bottom of the hypothesis by the same
      // constant without changing the output.
      T maxValue = T(0);
      for (typename Actions<T>::const_iterator a = Base::actions->begin();
          a != Base::actions->end(); ++a)
      {
        T tmp = u->dot(phi->at(*a));
        if (tmp > maxValue)
          maxValue = tmp;
      }

      for (typename Actions<T>::const_iterator a = Base::actions->begin();
          a != Base::actions->end(); ++a)
      {
        const int id = (*a)->id();
        Base::distribution->at(id) = exp(u->dot(phi->at(*a)) - maxValue);
        Boundedness::checkValue(Base::distribution->at(id));
        sum += Base::distribution->at(id);
        avg->addToSelf(Base::distribution->at(id), phi->at(*a));
      }

      for (typename Actions<T>::const_iterator a = Base::actions->begin();
          a != Base::actions->end(); ++a)
      {
        const int id = (*a)->id();
        Base::distribution->at(id) /= sum;
        Boundedness::checkValue(Base::distribution->at(id));
      }
      avg->mapMultiplyToSelf(1.0 / sum);
    }

    const Vectors<T>& computeGradLog(const Representations<T>* phi, const Action<T>* action)
    {
      grad->set(phi->at(action));
      grad->subtractToSelf(avg);
      return *multigrad;
    }

    Vectors<T>* parameters() const
    {
      return multiu;
    }

    T pi(const Action<T>* action)
    {
      return Base::pi(action);
    }

    const Action<T>* sampleAction()
    {
      return Base::sampleAction();
    }

    const Action<T>* sampleBestAction()
    {
      return Base::sampleBestAction();
    }
};

template<class T>
class SoftMax: public StochasticPolicy<T>
{
  protected:
    Predictor<T>* predictor;
    T temperature;
    typedef StochasticPolicy<T> Base;
  public:
    SoftMax(Random<T>* random, Actions<T>* actions, Predictor<T>* predictor, const T temperature =
        T(1)) :
        StochasticPolicy<T>(random, actions), predictor(predictor), temperature(temperature)
    {
    }

    virtual ~SoftMax()
    {
    }

    void update(const Representations<T>* phi)
    {
      ASSERT(Base::actions->dimension() == phi->dimension());
      Base::distribution->clear();
      T sum = T(0);
      // The exponential function may become very large and overflow.
      // Therefore, we multiply top and bottom of the hypothesis by the same
      // constant without changing the output.
      T maxValue = T(0);
      for (typename Actions<T>::const_iterator a = Base::actions->begin();
          a != Base::actions->end(); ++a)
      {
        T tmp = predictor->predict(phi->at(*a));
        if (tmp > maxValue)
          maxValue = tmp;
      }

      for (typename Actions<T>::const_iterator a = Base::actions->begin();
          a != Base::actions->end(); ++a)
      {
        const int id = (*a)->id();
        Base::distribution->at(id) = exp(
            (predictor->predict(phi->at(*a)) - maxValue) / temperature);
        Boundedness::checkValue(Base::distribution->at(id));
        sum += Base::distribution->at(id);
      }

      for (typename Actions<T>::const_iterator a = Base::actions->begin();
          a != Base::actions->end(); ++a)
      {
        const int id = (*a)->id();
        Base::distribution->at(id) /= sum;
        Boundedness::checkValue(Base::distribution->at(id));
      }
    }
};

template<class T>
class RandomPolicy: public Policy<T>
{
  protected:
    Random<T>* random;
    Actions<T>* actions;
  public:
    RandomPolicy(Random<T>* random, Actions<T>* actions) :
        random(random), actions(actions)
    {
    }

    virtual ~RandomPolicy()
    {
    }

    void update(const Representations<T>* phi)
    {
    }
    T pi(const Action<T>* a)
    {
      return T(1) / actions->dimension();
    }
    const Action<T>* sampleAction()
    {
      return actions->at(random->nextInt(actions->dimension()));
    }
    const Action<T>* sampleBestAction()
    {
      ASSERT(false);
      return actions->at(0);
    }
};

template<class T>
class RandomBiasPolicy: public Policy<T>
{
  protected:
    Random<T>* random;
    Actions<T>* actions;
    const Action<T>* prev;
    PVector<T>* distribution;
  public:
    RandomBiasPolicy(Random<T>* random, Actions<T>* actions) :
        random(random), actions(actions), prev(&actions->at(0)), distribution(
            new PVector<T>(actions->dimension()))
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
        for (typename Actions<T>::const_iterator a = actions->begin(); a != actions->end(); ++a)
        {
          const int id = (*a)->id();
          if (id == prev->id())
            distribution->at(id) = 0.5;
          else
            distribution->at(id) = 0.5 / (actions->dimension() - 1);
        }
      }
      // chose an action
      T rand = random->nextDouble();
      T sum = T(0);
      for (typename Actions<T>::const_iterator a = actions->begin(); a != actions->end(); ++a)
      {
        sum += distribution->at((*a)->id());
        if (sum >= rand)
        {
          prev = *a;
          return;
        }
      }
      prev = &actions->at(actions->dimension() - 1);
    }

    T pi(const Action<T>* action)
    {
      return distribution->at(action->id());
    }

    const Action<T>* sampleAction()
    {
      return *prev;
    }
    const Action<T>* sampleBestAction()
    {
      ASSERT(false);
      return actions->at(0);
    }
};

template<class T>
class Greedy: public DiscreteActionPolicy<T>
{
  protected:
    Actions<T>* actions;
    Predictor<T>* predictor;
    T* actionValues;
    const Action<T>* bestAction;

  public:
    Greedy(Actions<T>* actions, Predictor<T>* predictor) :
        actions(actions), predictor(predictor), actionValues(new T[actions->dimension()]), bestAction(
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
      for (typename Actions<T>::const_iterator iter = actions->begin(); iter != actions->end();
          ++iter)
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

    T pi(const Action<T>* a)
    {
      return (bestAction->id() == a->id()) ? T(1) : T(0);
    }

    const Action<T>* sampleAction()
    {
      return bestAction;
    }

    const Action<T>* sampleBestAction()
    {
      return bestAction;
    }

};

template<class T>
class EpsilonGreedy: public Greedy<T>
{
  protected:
    Random<T>* random;
    T epsilon;
  public:
    EpsilonGreedy(Random<T>* random, Actions<T>* actions, Predictor<T>* predictor, const T& epsilon) :
        Greedy<T>(actions, predictor), random(random), epsilon(epsilon)
    {
    }

    const Action<T>* sampleAction()
    {
      if (random->nextReal() < epsilon)
        return Greedy<T>::actions->at(random->nextInt(Greedy<T>::actions->dimension()));
      else
        return Greedy<T>::bestAction;
    }

    T pi(const Action<T>* a)
    {
      T probability = (a->id() == Greedy<T>::bestAction->id()) ? T(1) - epsilon : T(0);
      return probability + epsilon / Greedy<T>::actions->dimension();
    }

};

// Special behavior policies
template<class T>
class BoltzmannDistributionPerturbed: public Policy<T>
{
  protected:
    Random<T>* random;
    Actions<T>* actions;
    Vector<T>* u;
    PVector<T>* distribution;
    T epsilon;
    T perturbation;

  public:
    BoltzmannDistributionPerturbed(Random<T>* random, Actions<T>* actions, Vector<T>* u,
        const T& epsilon, const T& perturbation) :
        random(random), actions(actions), u(u), distribution(new PVector<T>(actions->dimension())), epsilon(
            epsilon), perturbation(perturbation)
    {
    }

    virtual ~BoltzmannDistributionPerturbed()
    {
      delete distribution;
    }

    void update(const Representations<T>* phis)
    {
      ASSERT(actions->dimension() == phis->dimension());
      distribution->clear();
      T sum = T(0);
      // The exponential function may become very large and overflow.
      // Therefore, we multiply top and bottom of the hypothesis by the same
      // constant without changing the output.
      T maxValue = T(0);
      for (typename Actions<T>::const_iterator a = actions->begin(); a != actions->end(); ++a)
      {
        T tmp = u->dot(phis->at(*a));
        if (tmp > maxValue)
          maxValue = tmp;
      }

      for (typename Actions<T>::const_iterator a = actions->begin(); a != actions->end(); ++a)
      {
        const int id = (*a)->id();
        T perturb = random->nextReal() < epsilon ? perturbation : T(0);
        distribution->at(id) = exp(u->dot(phis->at(*a)) + perturb - maxValue);
        Boundedness::checkValue(distribution->at(id));
        sum += distribution->at(id);
      }

      for (typename Actions<T>::const_iterator a = actions->begin(); a != actions->end(); ++a)
      {
        const int id = (*a)->id();
        distribution->at(id) /= sum;
        Boundedness::checkValue(distribution->at(id));
      }

    }

    T pi(const Action<T>* action)
    {
      return distribution->at(action->id());
    }

    const Action<T>* sampleAction()
    {
      T rand = random->nextReal();
      T sum = T(0);
      for (typename Actions<T>::const_iterator a = actions->begin(); a != actions->end(); ++a)
      {
        sum += distribution->at((*a)->id());
        if (sum >= rand)
          return *a;
      }
      return actions->at(actions->dimension() - 1);
    }

    const Action<T>* sampleBestAction()
    {
      return sampleAction();
    }

};

template<class T>
class SingleActionPolicy: public Policy<T>
{
  private:
    Actions<T>* actions;
  public:
    SingleActionPolicy(Actions<T>* actions) :
        actions(actions)
    {
      ASSERT(actions->dimension() == 1);
    }

    void update(const Representations<T>* phis)
    {
    }

    T pi(const Action<T>* a)
    {
      return a->id() == actions->at(0)->id() ? T(1) : T(0);
    }

    const Action<T>* sampleAction()
    {
      return actions->at(0);
    }

    const Action<T>* sampleBestAction()
    {
      return sampleAction();
    }
};

template<class T>
class ConstantPolicy: public StochasticPolicy<T>
{
  private:
    typedef StochasticPolicy<T> Base;
  public:
    ConstantPolicy(Random<T>* random, Actions<T>* actions, const Vector<T>* distribution) :
        StochasticPolicy<T>(random, actions)
    {
      ASSERT(actions->dimension() == distribution->dimension());
      for (int i = 0; i < distribution->dimension(); i++)
        Base::distribution->at(i) = distribution->getEntry(i);
    }

    virtual ~ConstantPolicy()
    {
    }

    void update(const Representations<T>* phi)
    {/*Empty*/
    }

    T pi(const Action<T>* action)
    {
      return Base::pi(action);
    }

    const Action<T>* sampleAction()
    {
      return Base::sampleAction();
    }

    const Action<T>* sampleBestAction()
    {
      return Base::sampleBestAction();
    }

};

} // namespace RLLib

#endif /* POLICY_H_ */
