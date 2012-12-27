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

#include "Vector.h"
#include "Action.h"
#include "Predictor.h"
#include "Representation.h"
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
    virtual void update(const Representations<T>& phis) =0;
    virtual double pi(const Action& a) const =0;
    virtual const Action& sampleAction() =0;
    virtual const Action& sampleBestAction() =0;

    const Action& decide(const Representations<T>& phis)
    {
      update(phis);
      return sampleAction();
    }
};

// start with discrete action policy
template<class T>
class DiscreteActionPolicy: public Policy<T>
{
  public:
    virtual ~DiscreteActionPolicy()
    {
    }

};

template<class T>
class PolicyDistribution: public Policy<T>
{
  public:
    virtual ~PolicyDistribution()
    {
    }
    virtual const SparseVectors<T>& computeGradLog(
        const Representations<T>& phis, const Action& action) =0;
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
  public:

    NormalDistribution(const double& initialMean, const double& initialStddev,
        const int& nbFeatures, ActionList* actions) :
        initialMean(initialMean), initialStddev(initialStddev), sigma2(0), mean(
            0), stddev(0), meanStep(0), stddevStep(0), u_mean(
            new SparseVector<T>(nbFeatures)), u_stddev(
            new SparseVector<T>(nbFeatures)), gradMean(
            new SparseVector<T>(u_mean->dimension())), gradStddev(
            new SparseVector<T>(u_stddev->dimension())), x(
            new SparseVector<T>(nbFeatures)), actions(actions), multiu(
            new SparseVectors<T>()), multigrad(new SparseVectors<T>())
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
      x->set(xs_t.at(actions->at(0)));
      mean = u_mean->dot(*x) + initialMean;
      stddev = exp(u_stddev->dot(*x)) * initialStddev + 10e-8;
      sigma2 = stddev * stddev;
    }

    double pi(const Action& a) const
    {
      return Random::gaussianProbability(a.at(0), mean, stddev);
    }

  public:

    const Action& sampleAction()
    {
      actions->update(0, 0, Random::nextNormalGaussian() * stddev + mean);
      return actions->at(0);
    }
    const Action& sampleBestAction()
    {
      actions->update(0, 0, Random::nextNormalGaussian() * stddev + mean);
      return actions->at(0);
    }

    virtual void updateStep(const Action& action)
    {
      double a = action.at(0);
      meanStep = (a - mean) / sigma2;
      stddevStep = (a - mean) * (a - mean) / sigma2 - 1.0;
    }

    const SparseVectors<T>& computeGradLog(const Representations<T>& xs,
        const Action& action)
    {
      assert((xs.dimension() == 1) && (actions->dimension() == 1));
      updateStep(action);
      x->set(xs.at(actions->at(0)));
      gradMean->set(*x);
      gradMean->multiplyToSelf(meanStep);
      gradStddev->set(*x);
      gradStddev->multiplyToSelf(meanStep);

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

    typedef NormalDistribution<T> Base;

    NormalDistributionScaled(const double& initialMean,
        const double& initialStddev, const int& nbFeatures, ActionList* actions) :
        NormalDistribution<T>(initialMean, initialStddev, nbFeatures, actions)
    {
    }
    virtual ~NormalDistributionScaled()
    {
    }

    virtual void updateStep(const Action& action)
    {
      double a = action.at(0);
      Base::meanStep = (a - Base::mean);
      Base::stddevStep = (a - Base::mean) * (a - Base::mean) - Base::sigma2;
    }

};

template<class T>
class BoltzmannDistribution: public PolicyDistribution<T>
{
  protected:
    ActionList* actions;
    SparseVector<T>* avg;
    SparseVector<T>* grad;
    DenseVector<double>* distribution;
    SparseVector<T>* u;
    SparseVectors<T>* multiu;
    SparseVectors<T>* multigrad;
  public:
    BoltzmannDistribution(const int& numFeatures, ActionList* actions) :
        actions(actions), avg(new SparseVector<T>(numFeatures)), grad(
            new SparseVector<T>(numFeatures)), distribution(
            new DenseVector<double>(actions->dimension())), u(
            new SparseVector<T>(numFeatures)), multiu(
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
      delete distribution;
      delete u;
      delete multiu;
      delete multigrad;
    }

    void update(const Representations<T>& xas)
    {
      assert(actions->dimension() == xas.dimension());
      distribution->clear();
      avg->clear();
      double sum = 0;
      // The exponential function may become very large and overflow.
      // Therefore, we multiply top and bottom of the hypothesis by the same
      // constant without changing the output.
      double maxValue = 0;
      for (ActionList::const_iterator a = actions->begin(); a != actions->end();
          ++a)
      {
        double tmp = u->dot(xas.at(**a));
        if (tmp > maxValue)
          maxValue = tmp;
      }

      for (ActionList::const_iterator a = actions->begin(); a != actions->end();
          ++a)
      {
        distribution->at(**a) = exp(u->dot(xas.at(**a)) - maxValue);
        Boundedness::checkValue(distribution->at(**a));
        sum += distribution->at(**a);
        avg->addToSelf(distribution->at(**a), xas.at(**a));
      }

      for (ActionList::const_iterator a = actions->begin(); a != actions->end();
          ++a)
      {
        distribution->at(**a) /= sum;
        Boundedness::checkValue(distribution->at(**a));
      }
      avg->multiplyToSelf(1.0 / sum);
    }

    const SparseVectors<T>& computeGradLog(const Representations<T>& xas,
        const Action& action)
    {
      grad->set(xas.at(action));
      grad->substractToSelf(*avg);
      return *multigrad;
    }

    double pi(const Action& action) const
    {
      return distribution->at(action);

    }
    const Action& sampleAction()
    {
      double random = Random::nextDouble();
      double sum = 0;
      for (ActionList::const_iterator a = actions->begin(); a != actions->end();
          ++a)
      {
        sum += distribution->at(**a);
        if (sum >= random)
          return **a;
      }
      return actions->at(actions->dimension() - 1);

    }
    const Action& sampleBestAction()
    {
      return sampleAction();
    }

    SparseVectors<T>* parameters() const
    {
      return multiu;
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
    double pi(const Action& a) const
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
        for (ActionList::const_iterator a = actions->begin();
            a != actions->end(); ++a)
        {
          if (**a == *prev)
            distribution->at(**a) = 0.5;
          else
            distribution->at(**a) = 0.5 / (actions->dimension() - 1);
        }
      }
      // chose an action
      double random = Random::nextDouble();
      double sum = 0;
      for (ActionList::const_iterator a = actions->begin(); a != actions->end();
          ++a)
      {
        sum += distribution->at(**a);
        if (sum >= random)
        {
          prev = *a;
          return;
        }
      }
      prev = &actions->at(actions->dimension() - 1);
    }

    double pi(const Action& action) const
    {
      return distribution->at(action);
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
        predictor(predictor), actions(actions), actionValues(
            new double[actions->dimension()]), bestAction(0)
    {
    }

    virtual ~Greedy()
    {
      delete[] actionValues;
    }

  private:

    void updateActionValues(const Representations<T>& xas_tp1)
    {
      for (ActionList::const_iterator iter = actions->begin();
          iter != actions->end(); ++iter)
        actionValues[**iter] = predictor->predict(xas_tp1.at(**iter));
    }

    void findBestAction()
    {
      bestAction = &actions->at(0);
      for (unsigned int i = 1; i < actions->dimension(); i++)
      {
        if (actionValues[i] > actionValues[*bestAction])
          bestAction = &actions->at(i);
      }
    }

  public:

    void update(const Representations<T>& xas_tp1)
    {
      updateActionValues(xas_tp1);
      findBestAction();
    }

    double pi(const Action& a) const
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
    EpsilonGreedy(Predictor<T>* predictor, ActionList* actions,
        const double& epsilon) :
        Greedy<T>(predictor, actions), epsilon(epsilon)
    {
    }

    const Action& sampleAction() const
    {
      if (Random::nextDouble() < epsilon)
        return (*Greedy<T>::actions)[rand() % Greedy<T>::actions->dimension()];
      else
        return *Greedy<T>::bestAction;
    }

    double pi(const Action& a) const
    {
      double probability = (a == *Greedy<T>::bestAction) ? 1.0 - epsilon : 0.0;
      return probability + epsilon / Greedy<T>::actions->dimension();
    }

};

} // namespace RLLib

#endif /* POLICY_H_ */
