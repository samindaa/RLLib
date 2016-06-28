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
#include "ControlAlgorithm.h"

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

    bool endOfEpisode() const
    {
      return (step_tp1->episode_state_tp1 == 3);
    }

    double r() const
    {
      return step_tp1->reward_tp1;
    }

    double z() const
    {
      return 0.0f;
    }

};

#endif /* OPENAI_GYM_OPENAIGYMRLPROBLEM_H_ */
