/*
 * RLLibOpenAiGymAgentRegistry.h
 *
 *  Created on: Jun 25, 2016
 *      Author: sabeyruw
 */

#ifndef OPENAI_GYM_RLLIBOPENAIGYMAGENTREGISTRY_H_
#define OPENAI_GYM_RLLIBOPENAIGYMAGENTREGISTRY_H_

#include "RLLibOpenAiGymAgent.h"

class RLLibOpenAiGymAgentRegistry
{
  public:
    static RLLibOpenAiGymAgent* make(const std::string& name);
};

#endif /* OPENAI_GYM_RLLIBOPENAIGYMAGENTREGISTRY_H_ */
