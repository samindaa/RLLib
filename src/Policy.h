/*
 * Policy.h
 *
 *  Created on: Aug 19, 2012
 *      Author: sam
 */

#ifndef POLICY_H_
#define POLICY_H_

#include <cstdlib>
#include <vector>

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
    virtual const double* getValues() const =0;
};

template<class T>
class Greedy: public DiscreteActionPolicy<T>
{
  protected:
    Predictor<T>* predictor;
    ActionList* actions;
    double* actionValues;
    double bestValue;
    const Action* bestAction;

  public:
    Greedy(Predictor<T>* predictor, ActionList* actions) :
        predictor(predictor), actions(actions),
            actionValues(new double[actions->getNumActions()]), bestValue(0),
            bestAction(0)
    {
    }

    virtual ~Greedy()
    {
      delete[] actionValues;
    }

  private:

    void updateActionValues(const std::vector<SparseVector<T>*>& xas_tp1)
    {
      for (unsigned int i = 0; i < actions->getNumActions(); i++)
        actionValues[i] = predictor->predict(*xas_tp1.at(i));
    }

    void findBestAction()
    {
      bestValue = actionValues[0];
      bestAction = &actions->at(0);
      for (unsigned int i = 0; i < actions->getNumActions(); i++)
      {
        double value = actionValues[i];
        if (value > bestValue)
        {
          bestValue = value;
          bestAction = &actions->at(i);
        }
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

    const double* getValues() const
    {
      return actionValues;
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
          % Greedy<T>::actions->getNumActions()];
      else return *Greedy<T>::bestAction;
    }

    double pi(const Action& a) const
    {
      double probability = (a == Greedy<T>::bestAction) ?
          1.0 - epsilon : 0.0;
      return probability + epsilon / Greedy<T>::actions->getNumActions();
    }

};

#endif /* POLICY_H_ */
