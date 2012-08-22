/*
 * Simulator.h
 *
 *  Created on: Jun 29, 2012
 *      Author: sam
 */

#ifndef SIMULATOR_H_
#define SIMULATOR_H_

#include <iostream>
#include <cmath>
#include <numeric>
#include <typeinfo>

#include "../src/Control.h"
#include "Env.h"

#include <iostream>
#include <vector>

template<class T, class O>
class Simulator
{
  protected:

    Control<T, O>* agent;
    Env<O>* env;

    const Action* action;
    int maxTestRuns;
    DenseVector<O> xt_tmp;

    std::vector<double> xTest;

  public:
    Simulator(Control<T, O>* agent, Env<O>* env) :
        agent(agent), env(env), action(0), maxTestRuns(20)
    {
    }
    void run(int maxRuns, int maxSteps, int maxEpisodes)
    {
      std::cout << "## ControlLearner=" << typeid(*agent).name() << std::endl;
      xTest.clear();

      for (int run = 0; run < maxRuns; run++)
      {
        std::cout << "## run=" << run << std::endl;
        agent->reset();

        for (int episode = 0; episode < maxEpisodes; episode++)
        {
          env->initialize();
          action = &agent->initialize(env->getVars());
          int steps = 0;
          do
          {
            env->step(*action);
            ++steps;
            action = &agent->step(xt_tmp, *action, env->getVars(), env->r(),
                env->z());
          } while (!env->endOfEpisode() && steps < maxSteps);

          std::cout << steps << " ";
          std::cout.flush();

        }
        std::cout << std::endl;

        std::cout << "## test" << std::endl;
        test(maxTestRuns, maxSteps);
      }

      double xbar = std::accumulate(xTest.begin(), xTest.end(), 0.0)
          / (double(xTest.size()));
      std::cout << "## avg length=" << xbar << std::endl;
      double sigmabar = 0;
      for (std::vector<double>::const_iterator x = xTest.begin();
          x != xTest.end(); ++x)
        sigmabar += pow((*x - xbar), 2);
      sigmabar = sqrt(sigmabar) / double(xTest.size());
      double se/*standard error*/= sigmabar / sqrt(double(xTest.size()));
      std::cout << "## (+- 95%) =" << (se * 2) << std::endl;

    }

    void test(int maxRuns, int maxSteps)
    {
      for (int run = 0; run < maxRuns; run++)
      {
        env->initialize();
        action = &agent->proposeAction(env->getVars());
        int steps = 0;
        do
        {
          env->step(*action);
          ++steps;
          action = &agent->proposeAction(env->getVars());
        } while (!env->endOfEpisode() && steps < maxSteps);

        xTest.push_back(steps);
        std::cout << steps << " ";
        std::cout.flush();
      }
      std::cout << std::endl;
    }
};

#endif /* SIMULATOR_H_ */
