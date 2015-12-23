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
 * NoisyInputSum.h
 *
 *  Created on: Nov 18, 2013
 *      Author: sam
 */

#ifndef NOISYINPUTSUM_H_
#define NOISYINPUTSUM_H_

#include "Vector.h"
#include "Mathema.h"
#include "Supervised.h"
#include "Timer.h"

namespace RLLib
{

  template<typename T>
  class PredictionProblem
  {
    public:
      virtual ~PredictionProblem()
      {
      }
      virtual bool update() =0;
      virtual T getTarget() const =0;
      virtual const Vector<T>* getInputs() const =0;
  };

  class NoisyInputSum: public PredictionProblem<double>
  {
    protected:
      double target;
      int nbChangingWeights;
      int changePeriod;
      int nbSteps;
      Vector<double>* inputs;
      Vector<double>* w;
      Random<double>* random;

    public:
      NoisyInputSum(const int& nbNonZeroWeights, const int& nbInputs) :
          target(0), nbChangingWeights(nbNonZeroWeights), changePeriod(20), nbSteps(0), //
          inputs(new PVector<double>(nbInputs)), w(new PVector<double>(nbInputs)), //
          random(new Random<double>())
      {
        for (int i = 0; i < w->dimension(); i++)
        {
          if (i < nbNonZeroWeights)
            w->setEntry(i, random->nextReal() > 0.5f ? 1.0f : -1.0f);
        }
      }

      virtual ~NoisyInputSum()
      {
        delete inputs;
        delete w;
        delete random;
      }

    private:
      void changeWeight()
      {
        const int i = random->nextInt(nbChangingWeights);
        w->setEntry(i, w->getEntry(i) * -1.0f);
      }

    public:
      bool update()
      {
        ++nbSteps;
        if ((nbSteps % changePeriod) == 0)
          changeWeight();
        inputs->clear();
        for (int i = 0; i < inputs->dimension(); i++)
          inputs->setEntry(i, random->nextNormalGaussian());
        target = w->dot(inputs);
        return true;
      }

      double getTarget() const
      {
        return target;
      }

      const Vector<double>* getInputs() const
      {
        return inputs;
      }
  };

  class NoisyInputSumEvaluation
  {
    public:
      int nbInputs;
      int nbNonZeroWeights;

      NoisyInputSumEvaluation() :
          nbInputs(20), nbNonZeroWeights(5)
      {
      }

      double evaluateLearner(LearningAlgorithm<double>* algorithm, const int& learningEpisodes,
          const int& evaluationEpisodes)
      {
        NoisyInputSum noisyInputSum(nbNonZeroWeights, nbInputs);
        Timer timer;
        timer.start();
        for (int i = 0; i < learningEpisodes; i++)
        {
          noisyInputSum.update();
          algorithm->learn(noisyInputSum.getInputs(), noisyInputSum.getTarget());
        }
        timer.stop();
        double elapsedTime = timer.getElapsedTimeInMilliSec();
        PVector<double> errors(evaluationEpisodes);
        timer.start();
        for (int i = 0; i < evaluationEpisodes; i++)
        {
          noisyInputSum.update();
          errors[i] = algorithm->learn(noisyInputSum.getInputs(), noisyInputSum.getTarget());
          ASSERT(Boundedness::checkValue(errors[i]));
        }
        timer.stop();
        elapsedTime += timer.getElapsedTimeInMilliSec();
        std::cout << "Time=" << (elapsedTime / (learningEpisodes + evaluationEpisodes))
            << std::endl;
        double mrse = errors.dot(&errors) / errors.dimension();
        ASSERT(Boundedness::checkValue(mrse));
        return mrse;
      }

      double evaluateLearner(LearningAlgorithm<double>* algorithm)
      {
        return evaluateLearner(algorithm, 20000, 10000);
      }
  };

}  // namespace RLLib

#endif /* NOISYINPUTSUM_H_ */
