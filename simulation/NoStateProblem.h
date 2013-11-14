/*
 * NoStateProblem.h
 *
 *  Created on: Oct 28, 2013
 *      Author: sam
 */

#ifndef NOSTATEPROBLEM_H_
#define NOSTATEPROBLEM_H_

#include "RL.h"

class NoStateProblem: public RLProblem<>
{
  protected:
    double mu;
    double sigma;
    const Range<double>* range;
    double currentA;
  public:
    NoStateProblem(const double& mu, const double& sigma, const Range<double>* range = 0) :
        RLProblem<>(1, 1, 1), mu(mu), sigma(sigma), range(range), currentA(0)
    {
      discreteActions->push_back(0, 0);
      continuousActions->push_back(0, 0);
    }

    void initialize()
    {
      updateRTStep();
    }

    void step(const Action<double>* action)
    {
      currentA = action->at(0);
      if (range != 0)
        currentA = range->bound(currentA);
      updateRTStep();
    }

    void updateRTStep()
    {
      observations->at(0) = output->o_tp1->at(0) = 1.0;
      output->updateRTStep(r(), z(), endOfEpisode());
    }

    bool endOfEpisode() const
    {
      return false;
    }

    float r() const
    {
      return Probabilistic::gaussianProbability(currentA, mu, sigma);
    }

    float z() const
    {
      return 0;
    }
};

#endif /* NOSTATEPROBLEM_H_ */
