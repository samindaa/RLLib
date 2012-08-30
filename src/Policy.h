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

#include "Vector.h"
#include "Action.h"
#include "Predictor.h"

template<class T>
class Policy
{
  public:
    virtual ~Policy()
    {
    }
    virtual void update(const std::vector<SparseVector<T>*>& xas) =0;
    virtual double pi(const Action& a) const =0;
    virtual const Action& sampleAction() const =0;
    virtual const Action& sampleBestAction() const =0;

    const Action& decide(const std::vector<SparseVector<T>*>& xas)
    {
      update(xas);
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
        const std::vector<SparseVector<T>*>& xas, const Action& action) =0;
    virtual std::vector<SparseVector<T>*>* parameters() =0;
};

template<class T>
class BoltzmannDistribution: public PolicyDistribution<T>
{
  protected:
    ActionList* actions;
    SparseVector<T>* avg;
    DenseVector<double>* distribution;
    std::vector<SparseVector<T>*>* u;
  public:
    BoltzmannDistribution(const int& numFeatures, ActionList* actions) :
        actions(actions), avg(new SparseVector<T>(numFeatures)),
            distribution(new DenseVector<double>(actions->dimension())),
            u(new std::vector<SparseVector<T>*>)
    {

      u->push_back(new SparseVector<T>(numFeatures));
    }
    virtual ~BoltzmannDistribution()
    {
      delete avg;
      delete distribution;
      for (typename std::vector<SparseVector<T>*>::iterator iter = u->begin();
          iter != u->end(); ++iter)
        delete *iter;
      u->clear();
      delete u;
    }

    void update(const std::vector<SparseVector<T>*>& xas)
    {
      distribution->clear();
      SparseVector<T>* _u = u->at(0);
      double sum = 0;
      for (unsigned int a = 0; a < xas.size(); a++)
      {
        distribution->at(a) = exp(_u->dot(*xas.at(a)));
        sum += distribution->at(a);
      }
      assert(sum);
      for (unsigned int a = 0; a < xas.size(); a++)
        distribution->at(a) /= sum;

    }

    const SparseVector<T>& computeGradLog(
        const std::vector<SparseVector<T>*>& xas, const Action& action)
    {
      avg->clear();
      avg->addToSelf(*xas.at(action));
      for (unsigned int b = 0; b < xas.size(); b++)
        avg->addToSelf(-distribution->at(b), *xas.at(b));
      return *avg;
    }

    double pi(const Action& action) const
    {
      for (unsigned int a = 0; a < actions->dimension(); a++)
      {
        if (action == actions->at(a)) return distribution->at(a);
      }
      assert(false);
      return distribution->at(0);

    }
    const Action& sampleAction() const
    {
      double random = drand48();
      double sum = 0;
      for (unsigned int a = 0; a < actions->dimension(); a++)
      {
        sum += distribution->at(a);
        if (sum >= random) return actions->at(a);
      }
      return actions->at(actions->dimension() - 1);

    }
    const Action& sampleBestAction() const
    {
      return sampleAction();
    }

    std::vector<SparseVector<T>*>* parameters()
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

    void update(const std::vector<SparseVector<T>*>& xas)
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

    void update(const std::vector<SparseVector<T>*>& xas)
    {
      // 50% prev action
      distribution->clear();
      if (distribution->dimension() == 1) distribution->at(0) = 1.0;
      else
      {
        for (unsigned int a = 0; a < actions->dimension(); a++)
        {
          if (actions->at(a) == prev) distribution->at(a) = 0.5;
          else distribution->at(a) = 0.5 / (actions->dimension() - 1);
        }
      }
      // chose an action
      double random = drand48();
      double sum = 0;
      for (unsigned int a = 0; a < actions->dimension(); a++)
      {
        sum += distribution->at(a);
        if (sum >= random)
        {
          prev = &actions->at(a);
          return;
        }
      }
      prev = &actions->at(actions->dimension() - 1);
    }
    double pi(const Action& action) const
    {
      for (unsigned int a = 0; a < actions->dimension(); a++)
      {
        if (action == actions->at(a)) return distribution->at(a);
      }
      assert(false);
      return actions->at(0);
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

    void updateActionValues(const std::vector<SparseVector<T>*>& xas_tp1)
    {
      for (ActionList::const_iterator iter = actions->begin();
          iter != actions->end(); ++iter)
        actionValues[**iter] = predictor->predict(*xas_tp1.at(**iter));
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

    void update(const std::vector<SparseVector<T>*>& xas_tp1)
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
      if (drand48() < epsilon) return (*Greedy<T>::actions)[rand()
          % Greedy<T>::actions->dimension()];
      else return *Greedy<T>::bestAction;
    }

    double pi(const Action& a) const
    {
      double probability = (a == *Greedy<T>::bestAction) ?
          1.0 - epsilon : 0.0;
      return probability + epsilon / Greedy<T>::actions->dimension();
    }

};

#endif /* POLICY_H_ */
