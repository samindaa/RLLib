/*
 * Copyright 2015 Saminda Abeyruwan (saminda@cs.miami.edu)
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

class NoStateProblem: public RLLib::RLProblem<double>
{
  protected:
    double mu;
    double sigma;
    const RLLib::Range<double>* range;
    double currentA;

  public:
    NoStateProblem(RLLib::Random<double>* random, const double& mu, const double& sigma,
        const RLLib::Range<double>* range = 0) :
        RLLib::RLProblem<double>(random, 1, 1, 1), mu(mu), sigma(sigma), range(range), currentA(0)
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

    void step(const RLLib::Action<double>* action)
    {
      currentA = action->getEntry(0);
      if (range != 0)
        currentA = range->bound(currentA);
    }

    void updateTRStep()
    {
      output->observation_tp1->setEntry(0, 1.0f);
      output->o_tp1->setEntry(0, output->observation_tp1->getEntry(0));
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
