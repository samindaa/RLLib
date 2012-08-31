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
#include "Representation.h"

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
    virtual const SparseVector<T>& computeGradLog(const Representations<T>& phis,
        const Action& action) =0;
    virtual SparseVector<T>* parameters() =0;
};

template<class T>
class NormalDistribution: public PolicyDistribution<T>
{
  protected:
    double mean0, sigma0, mean, sigma, meanStep, sigmaStep, sigma2, a_t;
    int numFeatures;
    ActionList* actions;
    SparseVector<T>* u;
    SparseVector<T>* grad;
    SparseVector<T>* u_mean;
    SparseVector<T>* u_sigma;
    SparseVector<T>* x_mean;
    SparseVector<T>* x_sigma;
  public:

    NormalDistribution(double mean0, double sigma0, const int& numFeatures,
        ActionList* actions) :
        mean0(mean0), sigma0(sigma0), mean(0), sigma(0), meanStep(0),
            sigmaStep(0), sigma2(0), a_t(0), numFeatures(numFeatures),
            actions(actions), u(new SparseVector<T>(2 * numFeatures)),
            grad(new SparseVector<T>(2 * numFeatures)),
            u_mean(new SparseVector<T>(numFeatures)),
            u_sigma(new SparseVector<T>(numFeatures)),
            x_mean(new SparseVector<T>(numFeatures)),
            x_sigma(new SparseVector<T>(numFeatures))
    {
    }

    virtual ~NormalDistribution()
    {
      delete u;
      delete grad;
      delete u_mean;
      delete u_sigma;
      delete x_mean;
      delete x_sigma;
    }

    void update(const Representations<T>& xs)
    {
      assert(xs.dimension()==actions->dimension());
      mean = u_mean->dot(xs.at(actions->at(0))) + mean0;
      sigma = exp(u_sigma->dot(xs.at(actions->at(0)))) * sigma0 + 10e-8;
      sigma2 = pow(sigma, 2);

      a_t = nextGaussian(mean, sigma);
      assert(actions->dimension() == 1);
      actions->update(0, 0, a_t);
    }

    double pi(const Action& a) const
    {
      return exp(-0.5 * pow(a.at() - mean, 2) / sigma2)
          / sqrt(2 * M_PI * sigma2);
    }

  private:
    // http://en.literateprograms.org/Box-Muller_transform_(C)
    double nextGaussian(const double& mean = 0, const double& stddev = 1.0)
    {
      static double n2 = 0.0;
      static int n2_cached = 0;
      if (!n2_cached)
      {
        double x, y, r;
        do
        {
          x = drand48() - 1;
          y = drand48() - 1;

          r = x * x + y * y;
        } while (r == 0.0 || r > 1.0);
        {
          double d = sqrt(-2.0 * log(r) / r);
          double n1 = x * d;
          n2 = y * d;
          double result = n1 * stddev + mean;
          n2_cached = 1;
          return result;
        }
      }
      else
      {
        n2_cached = 0;
        return n2 * stddev + mean;
      }
    }
  public:

    const Action& sampleAction() const
    {
      return actions->at(0);
    }
    const Action& sampleBestAction() const
    {
      return actions->at(0);
    }

    const SparseVector<T>& computeGradLog(const Representations<T>& xs,
        const Action& action)
    {
      assert(actions->dimension() == 1);
      meanStep = (action.at() - mean) / sigma2;
      sigmaStep = pow(action.at() - mean, 2) / sigma2 - 1.0;

      x_mean->set(xs.at(action));
      x_sigma->set(xs.at(action));

      x_mean->multiplyToSelf(meanStep);
      x_sigma->multiplyToSelf(sigmaStep);

      grad->clear();
      // 1st vector
      grad->addToSelf(*x_mean);
      // 2nd vector
      grad->addToSelf(1, *x_sigma, numFeatures);
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
    DenseVector<double>* distribution;
    SparseVector<T>* u;
  public:
    BoltzmannDistribution(const int& numFeatures, ActionList* actions) :
        actions(actions), avg(new SparseVector<T>(numFeatures)),
            distribution(new DenseVector<double>(actions->dimension())),
            u(new SparseVector<T>(numFeatures))
    {
    }
    virtual ~BoltzmannDistribution()
    {
      delete avg;
      delete distribution;
      delete u;
    }

    void update(const Representations<T>& xas)
    {
      assert(actions->dimension() == xas.dimension());
      distribution->clear();
      double sum = 0;
      for (ActionList::const_iterator a = actions->begin(); a != actions->end();
          ++a)
      {
        distribution->at(**a) = exp(u->dot(xas.at(**a)));
        sum += distribution->at(**a);
      }
      assert(sum);
      for (ActionList::const_iterator a = actions->begin(); a != actions->end();
          ++a)
        distribution->at(**a) /= sum;

    }

    const SparseVector<T>& computeGradLog(const Representations<T>& xas,
        const Action& action)
    {
      avg->clear();
      avg->addToSelf(xas.at(action));
      for (ActionList::const_iterator b = actions->begin(); b != actions->end();
          ++b)
        avg->addToSelf(-distribution->at(**b), xas.at(**b));
      return *avg;
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
        if (sum >= random) return **a;
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
      if (distribution->dimension() == 1) distribution->at(0) = 1.0;
      else
      {
        for (ActionList::const_iterator a = actions->begin();
            a != actions->end(); ++a)
        {
          if (**a == *prev) distribution->at(**a) = 0.5;
          else distribution->at(**a) = 0.5 / (actions->dimension() - 1);
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
