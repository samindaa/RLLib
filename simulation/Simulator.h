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
#include "Environment.h"
#include "Timer.h"

#include <iostream>
#include <vector>

template<class T, class O>
class Simulator
{
  protected:
    Control<T, O>* agent;
    Environment<O>* environment;

    const Action* a_t;
    DenseVector<O>* x_t;
    DenseVector<O>* x_tp1;

    int nbRuns;
    int maxEpisodeTimeSteps;
    int nbEpisodes;
    int nbEpisodeDone;
    bool beginingOfEpisode;
    bool evaluate;
    bool verbose;

    Timer timer;
    double totalTimeInMilliseconds;

    std::vector<double> statistics;
    bool enableStatistics;

    bool enableTestEpisodesAfterEachRun;
    int maxTestEpisodesAfterEachRun;
  public:
    int timeStep;
    double episodeR;
    double episodeZ;

    Simulator(Control<T, O>* agent, Environment<O>* environment, int nbRuns,
        int maxEpisodeTimeSteps, int nbEpisodes) :
        agent(agent), environment(environment), a_t(0), x_t(
            new DenseVector<O>(environment->getVars().dimension())), x_tp1(
            new DenseVector<O>(environment->getVars().dimension())), nbRuns(nbRuns), maxEpisodeTimeSteps(
            maxEpisodeTimeSteps), nbEpisodes(nbEpisodes), nbEpisodeDone(0), beginingOfEpisode(true), evaluate(
            false), verbose(true), totalTimeInMilliseconds(0), enableStatistics(false), enableTestEpisodesAfterEachRun(
            false), maxTestEpisodesAfterEachRun(20), timeStep(0), episodeR(0), episodeZ(0)
    {
    }

    ~Simulator()
    {
      delete x_t;
      delete x_tp1;
    }

    void setVerbose(const bool& verbose)
    {
      this->verbose = verbose;
    }

    void setEvaluate(const bool& evaluate)
    {
      this->evaluate = evaluate;
    }

    void setRuns(const int& nbRuns)
    {
      this->nbRuns = nbRuns;
    }

    void setEpisodes(const int& nbEpisodes)
    {
      this->nbEpisodes = nbEpisodes;
    }

    void setTestEpisodesAfterEachRun(const bool& enableTestEpisodesAfterEachRun)
    {
      this->enableTestEpisodesAfterEachRun = enableTestEpisodesAfterEachRun;
    }

    void benchmark()
    {
      double xbar = std::accumulate(statistics.begin(), statistics.end(), 0.0)
          / (double(statistics.size()));
      std::cout << std::endl;
      std::cout << "## Average: length=" << xbar << std::endl;
      double sigmabar = 0;
      for (std::vector<double>::const_iterator x = statistics.begin(); x != statistics.end(); ++x)
        sigmabar += pow((*x - xbar), 2);
      sigmabar = sqrt(sigmabar) / double(statistics.size());
      double se/*standard error*/= sigmabar / sqrt(double(statistics.size()));
      std::cout << "## (+- 95%) =" << (se * 2) << std::endl;
      statistics.clear();
    }

    void step()
    {
      if (beginingOfEpisode)
      {
        environment->setOn(false);
        environment->initialize();
        x_t->set(environment->getVars());
        a_t = &agent->initialize(*x_t);
        timeStep = 0;
        episodeR = 0;
        episodeZ = 0;
        totalTimeInMilliseconds = 0;
        beginingOfEpisode = false;
      }
      else
      {
        environment->step(*a_t);
        const TRStep<O>& step = environment->getTRStep();
        x_tp1->set(environment->getVars());
        ++timeStep;
        episodeR += step.r_tp1;
        episodeZ += step.z_tp1;
        timer.start();
        a_t =
            evaluate ?
                &agent->proposeAction(environment->getVars()) :
                &agent->step(*x_t, *a_t, *x_tp1, step.r_tp1, step.z_tp1);
        x_t->set(*x_tp1);
        timer.stop();
        totalTimeInMilliseconds += timer.getElapsedTimeInMilliSec();
      }

      if (environment->getTRStep().endOfEpisode || (timeStep == maxEpisodeTimeSteps))
      {
        // Episode ended
        if (verbose)
        {
          double averageTimePerStep = totalTimeInMilliseconds / timeStep;
          std::cout << timeStep << " (" << episodeR << "," << episodeZ << "," << averageTimePerStep
              << ") ";
          //std::cout << ".";
          std::cout.flush();
        }
        if (enableStatistics)
          statistics.push_back(timeStep);
        timeStep = 0;
        ++nbEpisodeDone;
        beginingOfEpisode = true;
      }
    }

    void runEpisodes()
    {
      do
      {
        step();
      } while (nbEpisodeDone < nbEpisodes);
    }

    void run()
    {
      if (verbose)
        std::cout << "## ControlLearner=" << typeid(*agent).name() << std::endl;
      for (int run = 0; run < nbRuns; run++)
      {
        std::cout << "## nbRun=" << run << std::endl;
        enableStatistics = true;
        statistics.clear();
        nbEpisodeDone = 0;
        // For each run
        if (!evaluate)
          agent->reset();
        runEpisodes();
        benchmark();

        if (enableTestEpisodesAfterEachRun)
        {
          // test run
          std::cout << "## enableTestEpisodesAfterEachRun=" << enableTestEpisodesAfterEachRun
              << std::endl;
          Simulator<T, O>* testSimulator = new Simulator<T, O>(agent, environment, 1,
              maxEpisodeTimeSteps, maxTestEpisodesAfterEachRun);
          testSimulator->setEvaluate(true);
          testSimulator->run();
          delete testSimulator;
        }
      }

    }

    void computeValueFunction(const char* outFile = "visualization/valueFunction.txt") const
    {
      if (environment->getVars().dimension() == 2) // only for two state variables
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
      environment->draw();
    }
};

#endif /* SIMULATOR_H_ */
