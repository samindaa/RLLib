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
    virtual const Action& sampleAction() const =0;
    virtual const Action& sampleBestAction() const =0;

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
    virtual const SparseVector<T>& computeGradLog(
        const Representations<T>& phis, const Action& action) =0;
    virtual SparseVector<T>* parameters() =0;
};

template<class T>
class NormalDistribution: public PolicyDistribution<T>
{
  protected:
    double mean0, stddev0, mean, stddev, meanStep, stddevStep;
    SparseVector<T>* u; // Policy gradient algorithms
    SparseVector<T>* x; // features
    SparseVector<T>* grad;
    ActionList* actions;
  public:

    NormalDistribution(double mean0, double stddev0, const int& numFeatures,
        ActionList* actions) :
        mean0(mean0), stddev0(stddev0), mean(0), stddev(0), meanStep(0),
            stddevStep(0), u(new SparseVector<T>(numFeatures + 1)),
            x(new SparseVector<T>(u->dimension())),
            grad(new SparseVector<T>(u->dimension())), actions(actions)
    {
    }

    virtual ~NormalDistribution()
    {
      delete u;
      delete x;
      delete grad;
    }

  public:
    void update(const Representations<T>& xs)
    {
      // N(mu,var) for single action, single representation only
      assert((xs.dimension() == 1) && (actions->dimension() == 1));
      x->set(xs.at(actions->at(0)));
      mean = u->dot(*x) + mean0;
      stddev = u->getEntry(u->dimension() - 1);
      actions->update(0, 0, Random::nextGaussian(mean, stddev));
    }

    double pi(const Action& a) const
    {
      return Random::gaussianProbability(a.at(0), mean, stddev);
    }

  public:

    const Action& sampleAction() const
    {
      return actions->at(0);
    }
    const Action& sampleBestAction() const
    {
      return actions->at(0); // small std after learning
    }

    virtual void updateStep(const Action& action)
    {
      double a = action.at(0);
      meanStep = (a - mean) / (pow(stddev, 2) + 0.000001);
      stddevStep = (pow((a - mean), 2) - pow(stddev, 2))
          / (pow(stddev, 3) + 0.000001);
    }

    const SparseVector<T>& computeGradLog(const Representations<T>& xs,
        const Action& action)
    {
      assert((xs.dimension() == 1) && (actions->dimension() == 1));
      updateStep(action);
      x->set(xs.at(actions->at(0)));

      grad->clear();
      grad->addToSelf(*x);
      grad->multiplyToSelf(meanStep);
      grad->insertLast(stddevStep);
      return *grad;
    }

    SparseVector<T>* parameters()
    {
      return u;
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
  public:
    BoltzmannDistribution(const int& numFeatures, ActionList* actions) :
        actions(actions), avg(new SparseVector<T>(numFeatures)),
            grad(new SparseVector<T>(numFeatures)),
            distribution(new DenseVector<double>(actions->dimension())),
            u(new SparseVector<T>(numFeatures))
    {
    }
    virtual ~BoltzmannDistribution()
    {
      delete avg;
      delete grad;
      delete distribution;
      delete u;
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

    const SparseVector<T>& computeGradLog(const Representations<T>& xas,
        const Action& action)
    {
      grad->set(xas.at(action));
      grad->substractToSelf(*avg);
      return *grad;
    }

    double pi(const Action& action) const
    {
      return distribution->at(action);

    }
    const Action& sampleAction() const
    {
      double random = drand48();
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
    const Action& sampleBestAction() const
    {
      return sampleAction();
    }

    SparseVector<T>* parameters()
    {
      return u;
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
    const Action& sampleAction() const
    {
      return actions->at(rand() % actions->dimension());
    }
    const Action& sampleBestAction() const
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
        actions(actions), prev(&actions->at(0)),
            distribution(new DenseVector<double>(actions->dimension()))
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
      double random = drand48();
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

    const Action& sampleAction() const
    {
      return *prev;
    }
    const Action& sampleBestAction() const
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
        predictor(predictor), actions(actions),
            actionValues(new double[actions->dimension()]), bestAction(0)
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
      return (bestAction == &a) ?
          1.0 : 0;
    }

    const Action& sampleAction() const
    {
      return *bestAction;
    }

    const Action& sampleBestAction() const
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
      if (drand48() < epsilon)
        return (*Greedy<T>::actions)[rand() % Greedy<T>::actions->dimension()];
      else
        return *Greedy<T>::bestAction;
    }

    double pi(const Action& a) const
    {
      double probability = (a == *Greedy<T>::bestAction) ?
          1.0 - epsilon : 0.0;
      return probability + epsilon / Greedy<T>::actions->dimension();
    }

};

} // namespace RLLib

#endif /* POLICY_H_ */
