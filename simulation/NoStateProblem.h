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
 * NoStateProblem.h
 *
 *  Created on: Oct 28, 2013
 *      Author: sam
 */

#ifndef NOSTATEPROBLEM_H_
#define NOSTATEPROBLEM_H_

#include "RL.h"

class NoStateProblem: public RLProblem<double>
{
  protected:
    double mu;
    double sigma;
    const Range<double>* range;
    double currentA;

  public:
    NoStateProblem(Random<double>* random, const double& mu, const double& sigma, const Range<double>* range = 0) :
        RLProblem<double>(random,1, 1, 1), mu(mu), sigma(sigma), range(range), currentA(0)
    {
      discreteActions->push_back(0, 0);
      continuousActions->push_back(0, 0);
    }

    virtual ~NoStateProblem()
    {
    }

    void initialize()
    {
    }

    void step(const Action<double>* action)
    {
      currentA = action->at(0);
      if (range != 0)
        currentA = range->bound(currentA);
    }

    void updateTRStep()
    {
      observations->at(0) = output->o_tp1->at(0) = 1.0;
      output->updateTRStep(r(), z(), endOfEpisode());
    }

    bool endOfEpisode() const
    {
      return false;
    }

    double r() const
    {
      return random->gaussianProbability(currentA, mu, sigma);
    }

    double z() const
    {
      return 0;
    }
};

#endif /* NOSTATEPROBLEM_H_ */
