/*
 * Copyright 2013 Saminda Abeyruwan (saminda@cs.miami.edu)
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
 * Simulator.h
 *
 *  Created on: Jun 29, 2012
 *      Author: sam
 */

#ifndef SIMULATOR_H_
#define SIMULATOR_H_

#include <iostream>
#include <fstream>
#include <cmath>
#include <numeric>
#include <typeinfo>

#include "Control.h"
#include "Env.h"

#include <iostream>
#include <vector>

template<class T, class O>
class Simulator
{
  protected:
    int maxTestRuns;

    Control<T, O>* agent;
    Env<O>* env;

    const Action* action;
    DenseVector<O>* x_t;
    DenseVector<O>* x_tp1;

    std::vector<double> xTest;

  public:
    double episodeR;
    double episodeZ;
    double time;

    Simulator(Control<T, O>* agent, Env<O>* env, int maxTestRuns = 20) :
        maxTestRuns(maxTestRuns), agent(agent), env(env), action(0), x_t(
            new DenseVector<O>(env->getVars().dimension())), x_tp1(
            new DenseVector<O>(env->getVars().dimension())), episodeR(0), episodeZ(0), time(0)
    {
    }

    ~Simulator()
    {
      delete x_t;
      delete x_tp1;
    }
    void run(const int& maxRuns, const int& maxSteps, const int& maxEpisodes, const bool& runTest =
        true, const bool& verbose = true)
    {
      if (verbose)
        std::cout << "## ControlLearner=" << typeid(*agent).name() << std::endl;
      xTest.clear();

      for (int run = 0; run < maxRuns; run++)
      {
        if (verbose)
          std::cout << "## run=" << run << std::endl;
        agent->reset();
        action = 0;

        for (int episode = 0; episode < maxEpisodes; episode++)
        {
          env->setOn(false);
          env->initialize();
          x_t->set(env->getVars());
          action = &agent->initialize(*x_t);
          int steps = 0;
          episodeR = 0;
          episodeZ = 0;
          time = 0;
          do
          {
            env->step(*action);
            x_tp1->set(env->getVars());
            ++steps;
            episodeR += env->r();
            episodeZ += env->z();
            action = &agent->step(*x_t, *action, *x_tp1, env->r(), env->z());
            x_t->set(*x_tp1);
          } while (!env->endOfEpisode() && steps < maxSteps);

          time = steps;

          if (verbose)
          {
            std::cout << steps << " (" << episodeR << "," << episodeZ << ") ";
            //std::cout << ".";
            std::cout.flush();
          }

        }
        if (verbose)
          std::cout << std::endl;

        if (runTest)
        {
          if (verbose)
            std::cout << "## test" << std::endl;
          test(maxTestRuns, maxSteps);
        }
      }

      if (runTest)
      {
        double xbar = std::accumulate(xTest.begin(), xTest.end(), 0.0) / (double(xTest.size()));
        std::cout << "## avg length=" << xbar << std::endl;
        double sigmabar = 0;
        for (std::vector<double>::const_iterator x = xTest.begin(); x != xTest.end(); ++x)
          sigmabar += pow((*x - xbar), 2);
        sigmabar = sqrt(sigmabar) / double(xTest.size());
        double se/*standard error*/= sigmabar / sqrt(double(xTest.size()));
        std::cout << "## (+- 95%) =" << (se * 2) << std::endl;
      }
    }

    void test(int maxRuns, int maxSteps)
    {
      for (int run = 0; run < maxRuns; run++)
      {
        env->setOn(true);
        env->initialize();
        action = &agent->proposeAction(env->getVars());
        episodeR = episodeZ = 0;
        int steps = 0;
        do
        {
          env->step(*action);
          ++steps;
          episodeR += env->r();
          episodeZ += env->z();
          action = &agent->proposeAction(env->getVars());
        } while (!env->endOfEpisode() && steps < maxSteps);

        xTest.push_back(steps);
        std::cout << steps << " (" << episodeR << "," << episodeZ << ") ";
        std::cout.flush();
      }
      std::cout << std::endl;
    }

    void computeValueFunction(const char* outFile = "visualization/valueFunction.txt") const
    {
      if (env->getVars().dimension() == 2) // only for two state variables
      {
        std::ofstream out(outFile);
        DenseVector<float> x_t(2);
        for (float x = 0; x <= 10; x += 0.1)
        {
          for (float y = 0; y <= 10; y += 0.1)
          {
            x_t[0] = x;
            x_t[1] = y;
            out << agent->computeValueFunction(x_t) << " ";
          }
          out << std::endl;
        }
        out.close();
      }

      // draw
      env->draw();
    }
};

#endif /* SIMULATOR_H_ */
