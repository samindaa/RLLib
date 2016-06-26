/*
 * OpenAiGymRLProblem.h
 *
 *  Created on: Jun 25, 2016
 *      Author: sabeyruw
 */

#ifndef OPENAI_GYM_OPENAIGYMRLPROBLEM_H_
#define OPENAI_GYM_OPENAIGYMRLPROBLEM_H_

#include <vector>
#include <iostream>
//
#include "RL.h"

class OpenAiGymTRStep
{
  public:
    std::vector<double> observation_tp1;
    double reward_tp1;
    int episode_state_tp1;

    OpenAiGymTRStep() :
        reward_tp1(0), episode_state_tp1(0)
    {
    }
};

class OpenAiGymRLProblem: public RLLib::RLProblem<double>
{
  public:
    OpenAiGymTRStep* step_tp1;

  public:
    OpenAiGymRLProblem(int nbVars, int nbDiscreteActions, int nbContinuousActions) :
        RLLib::RLProblem<double>(NULL, nbVars, nbDiscreteActions, nbContinuousActions), //
        step_tp1(new OpenAiGymTRStep())
    {
    }
    virtual ~OpenAiGymRLProblem()
    {
      delete step_tp1;
    }

    void initialize()
    {
    }

    void step(const RLLib::Action<double>* a)
    {
    }

};

#endif /* OPENAI_GYM_OPENAIGYMRLPROBLEM_H_ */
