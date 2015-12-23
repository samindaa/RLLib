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
 * ContinuousGridworld.h
 *
 *  Created on: Sep 1, 2012
 *      Author: sam
 */

#ifndef CONTINUOUSGRIDWORLD_H_
#define CONTINUOUSGRIDWORLD_H_

#include "RL.h"

template<typename T>
class ContinuousGridworld: public RLLib::RLProblem<T>
{
    typedef RLLib::RLProblem<T> Base;
  protected:
    RLLib::Range<T>* observationRange;
    RLLib::Range<T>* actionRange;
    float absoluteNoise;
    std::ofstream outpath;

  public:
    ContinuousGridworld(RLLib::Random<T>* random) :
        RLLib::RLProblem<T>(random, 2, 2 * 2 + 1, 1), //
        observationRange(new RLLib::Range<T>(0, 1.0)), //
        actionRange(new RLLib::Range<T>(-0.05, 0.05)), absoluteNoise(0.025)
    {
      // discrete actions
      for (int i = 0; i < Base::discreteActions->dimension(); i++)
      {
        for (int k = 0; k < 2; k++)
          Base::discreteActions->push_back(i, 0);
      }
      for (int i = 0; i < Base::discreteActions->dimension() - 1; i++)
      {
        int dimension = i / 2;
        int dimensionAction = i % 2;
        if (dimensionAction == 0)
          Base::discreteActions->update(i, dimension, -1.0);
        else
          Base::discreteActions->update(i, dimension, 1.0);
      }
      // continuous actions are not setup for this problem
      Base::observationRanges->push_back(observationRange);
      Base::observationRanges->push_back(observationRange);
    }

    virtual ~ContinuousGridworld()
    {
      delete observationRange;
      delete actionRange;
    }

    void initialize()
    {
      Base::output->observation_tp1->setEntry(0, 0.2);
      Base::output->observation_tp1->setEntry(1, 0.4);
    }

    void updateTRStep()
    { // nothing
      // unit generalization
      for (int i = 0; i < Base::output->o_tp1->dimension(); i++)
        Base::output->o_tp1->setEntry(i, Base::output->observation_tp1->getEntry(i));
    }

    void step(const RLLib::Action<T>* action)
    {
      float noise = Base::random->nextReal() * absoluteNoise - (absoluteNoise / 2.0f);
      for (int i = 0; i < Base::output->observation_tp1->dimension(); i++)
        Base::output->observation_tp1->setEntry(i,
            observationRange->bound(Base::output->observation_tp1->getEntry(i) //
            + actionRange->bound(action->getEntry(i) + noise)));
    }

    bool endOfEpisode() const
    {
      // L1-norm
      float distance = 0;
      for (int i = 0; i < Base::output->observation_tp1->dimension(); i++)
        distance += fabs(1.0 - Base::output->observation_tp1->getEntry(i));
      return distance < 0.1;
    }

    float N(const float& p, const float& mu, const float& sigma) const
    {
      return Base::random->gaussianProbability(p, mu, sigma);
    }

    T r() const
    {
      float px = Base::output->observation_tp1->getEntry(0);
      float py = Base::output->observation_tp1->getEntry(1);
      return -1.0f - 2.0f * (N(px, 0.3f, 0.1f) * N(py, 0.6f, 0.03f) + //
          N(px, 0.4f, 0.03f) * N(py, 0.5f, 0.1f) + N(px, 0.8f, 0.03f) * N(py, 0.9f, 0.1f));
    }

    T z() const
    {
      return 0.0f;
    }

    void draw() const
    {
      std::ofstream oute("visualization/continuousGridworld.txt");
      for (float px = observationRange->min(); px <= observationRange->max(); px += 0.01f)
      {
        for (float py = observationRange->min(); py <= observationRange->max(); py += 0.01f)
          oute << (N(px, 0.3f, 0.1f) * N(py, 0.6f, 0.03f) + //
              N(px, 0.4f, 0.03f) * N(py, 0.5f, 0.1f) + N(px, 0.8f, 0.03f) * N(py, 0.9f, 0.1f))
              << " ";
        oute << std::endl;
      }
      oute.close();
    }
};

#endif /* CONTINUOUSGRIDWORLD_H_ */
