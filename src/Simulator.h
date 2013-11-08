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

#include <cmath>
#include <numeric>
#include <typeinfo>

#include <iostream>
#include <vector>

#include "Control.h"
#include "Environment.h"
#include "Timer.h"

namespace RLLib
{

template<class T>
class Simulator
{
  public:
    class Event
    {
        friend class Simulator;
      protected:
        int nbTotalTimeSteps;
        int nbEpisodeDone;
        double averageTimePerStep;
        double episodeR;
        double episodeZ;

      public:
        Event() :
            nbTotalTimeSteps(0), nbEpisodeDone(0), averageTimePerStep(0), episodeR(0), episodeZ(0)
        {
        }

        virtual ~Event()
        {
        }

        virtual void update() const=0;

    };
  protected:
    Control<T>* agent;
    Environment<T>* environment;

    const Action<T>* a_t;
    Vector<T>* x_0; // << this is the terminal state
    Vector<T>* x_t;
    Vector<T>* x_tp1;

    int maxEpisodeTimeSteps;
    int nbEpisodes;
    int nbRuns;
    int nbEpisodeDone;
    bool endingOfEpisode;
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
    std::vector<Event*> onEpisodeEnd;

    Simulator(Control<T>* agent, Environment<T>* environment, const int& maxEpisodeTimeSteps,
        const int nbEpisodes = -1, const int nbRuns = -1) :
        agent(agent), environment(environment), a_t(0), x_0(new PVector<T>(0)), x_t(
            new PVector<T>(environment->dimension())), x_tp1(
            new PVector<T>(environment->dimension())), maxEpisodeTimeSteps(maxEpisodeTimeSteps), nbEpisodes(
            nbEpisodes), nbRuns(nbRuns), nbEpisodeDone(0), endingOfEpisode(false), evaluate(false), verbose(
            true), totalTimeInMilliseconds(0), enableStatistics(false), enableTestEpisodesAfterEachRun(
            false), maxTestEpisodesAfterEachRun(20), timeStep(0), episodeR(0), episodeZ(0)
    {
    }

    ~Simulator()
    {
      delete x_t;
      delete x_tp1;
      onEpisodeEnd.clear();
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

    void setEnableStatistics(const bool& enableStatistics)
    {
      this->enableStatistics = enableStatistics;
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
      if (!a_t)
      {
        /*Initialize the problem*/
        environment->initialize();
        x_t->set(environment->getTRStep()->o_tp1);
        /*Statistic variables*/
        timeStep = 0;
        episodeR = 0;
        episodeZ = 0;
        totalTimeInMilliseconds = 0;
        /*The episode is just started*/
        endingOfEpisode = false;
      }
      else
      {
        /*Step through the problem*/
        environment->step(a_t);
      }

      if (!a_t)
      {
        /*Initialize the control agent and get the first action*/
        a_t = agent->initialize(x_t);
      }
      else
      {
        const TRStep<T>* step = environment->getTRStep();
        x_tp1->set(step->o_tp1);
        ++timeStep;
        episodeR += step->r_tp1;
        episodeZ += step->z_tp1;
        endingOfEpisode = step->endOfEpisode;// || (timeStep == maxEpisodeTimeSteps);
        timer.start();
        a_t =
            evaluate ?
                agent->proposeAction(x_tp1) :
                agent->step(x_t, a_t, (!endingOfEpisode ? x_tp1 : x_0), step->r_tp1, step->z_tp1);
        x_t->set(x_tp1);
        timer.stop();
        totalTimeInMilliseconds += timer.getElapsedTimeInMilliSec();

      }

      if (endingOfEpisode/*The episode is just ended*/|| (timeStep == maxEpisodeTimeSteps))
      {
        if (verbose)
        {
          double averageTimePerStep = totalTimeInMilliseconds / timeStep;
          std::cout << "{" << nbEpisodeDone << " [" << timeStep << " (" << episodeR << ","
              << episodeZ << "," << averageTimePerStep << ")]} ";
          //std::cout << ".";
          std::cout.flush();
        }
        if (enableStatistics)
          statistics.push_back(timeStep);
        ++nbEpisodeDone;
        /*Set the initial marker*/
        a_t = 0;
        // Fire the events
        for (typename std::vector<Event*>::iterator iter = onEpisodeEnd.begin();
            iter != onEpisodeEnd.end(); ++iter)
        {
          Event* e = *iter;
          e->nbTotalTimeSteps = timeStep;
          e->nbEpisodeDone = nbEpisodeDone;
          e->averageTimePerStep = (totalTimeInMilliseconds / timeStep);
          e->episodeR = episodeR;
          e->episodeZ = episodeZ;
          e->update();
        }
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
        if (verbose)
          std::cout << "## nbRun=" << run << std::endl;
        if (enableStatistics)
          statistics.clear();
        nbEpisodeDone = 0;
        // For each run
        if (!evaluate)
          agent->reset();
        runEpisodes();
        if (enableStatistics)
          benchmark();

        if (enableTestEpisodesAfterEachRun)
        {
          // test run
          std::cout << "## enableTestEpisodesAfterEachRun=" << enableTestEpisodesAfterEachRun
              << std::endl;
          Simulator<T>* testSimulator = new Simulator<T>(agent, environment, maxEpisodeTimeSteps,
              maxTestEpisodesAfterEachRun, 1);
          testSimulator->setEvaluate(true);
          testSimulator->run();
          delete testSimulator;
        }
        Probabilistic::srand(0); //<< fixMe
      }

    }

    bool isBeginingOfEpisode() const
    {
      return a_t == 0;
    }

    bool isEndingOfEpisode() const
    {
      return endingOfEpisode;
    }

    const Environment<T>* getEnvironment() const
    {
      return environment;
    }

    int getMaxEpisodeTimeSteps() const
    {
      return maxEpisodeTimeSteps;
    }

    void computeValueFunction(const char* outFile = "visualization/valueFunction.txt") const
    {
      if (environment->dimension() == 2) // only for two state variables
      {
        std::ofstream out(outFile);
        PVector<T> x_t(2);
        for (float x = 0; x <= 10; x += 0.1)
        {
          for (float y = 0; y <= 10; y += 0.1)
          {
            x_t.at(0) = x;
            x_t.at(1) = y;
            out << agent->computeValueFunction(&x_t) << " ";
          }
          out << std::endl;
        }
        out.close();
      }

      // draw
      environment->draw();
    }
};

}  // namespace RLLib

#endif /* SIMULATOR_H_ */
