/*
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

#include "../src/Control.h"
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
    Simulator(Control<T, O>* agent, Env<O>* env, int maxTestRuns = 20) :
        maxTestRuns(maxTestRuns), agent(agent), env(env), action(0),
            x_t(new DenseVector<O>(env->getVars().dimension())),
            x_tp1(new DenseVector<O>(env->getVars().dimension()))
    {
    }

    ~Simulator()
    {
      delete x_t;
      delete x_tp1;
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
          env->setOn(false);
          env->initialize();
          x_t->set(env->getVars());
          action = &agent->initialize(*x_t);
          int steps = 0;
          do
          {
            env->step(*action);
            x_tp1->set(env->getVars());
            ++steps;
            action = &agent->step(*x_t, *action, *x_tp1, env->r(), env->z());
            x_t->set(*x_tp1);
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
        env->setOn(true);
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

    void computeValueFunction() const
    {
      if (env->getVars().dimension() == 2) // only for two state variables
      {
        std::ofstream out("valueFunction.txt");
        DenseVector<float> x_t(2);
        for (float x = -10; x <= 10; x += 0.1)
        {
          for (float y = -5; y <= 5; y += 0.1)
          {
            x_t[0] = x;
            x_t[1] = y;
            out << agent->computeValueFunction(x_t) << " ";
          }
          out << std::endl;
        }
        out.close();
      }
    }
};

#endif /* SIMULATOR_H_ */
