/*
 * RLLibOpenAiGymAgent.h
 *
 *  Created on: Jun 25, 2016
 *      Author: sabeyruw
 */

#ifndef OPENAI_GYM_RLLIBOPENAIGYMAGENT_H_
#define OPENAI_GYM_RLLIBOPENAIGYMAGENT_H_

#include "OpenAiGymRLProblem.h"

class RLLibOpenAiGymAgent
{
  public:
    OpenAiGymRLProblem* problem; //<< interface between OpenAi and RLLib

    RLLibOpenAiGymAgent() :
        problem(nullptr)
    {
    }

    virtual ~RLLibOpenAiGymAgent()
    {
    }

    virtual const RLLib::Action<double>* step() =0;
};

#endif /* OPENAI_GYM_RLLIBOPENAIGYMAGENT_H_ */
