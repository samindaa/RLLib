/*
 * RLLibOpenAiGymAgentMacro.h
 *
 *  Created on: Jul 21, 2016
 *      Author: sabeyruw
 */

#ifndef OPENAI_GYM_RLLIBOPENAIGYMAGENTMACRO_H_
#define OPENAI_GYM_RLLIBOPENAIGYMAGENTMACRO_H_

#include "OpenAiGymRLProblem.h"
#include "RLLibOpenAiGymAgent.h"
#include "RLLibOpenAiGymAgentLoader.h"
#include "RLLibOpenAiGymAgentFactory.h"
#include "RLLibOpenAiGymAgentRegistry.h"

#define OPENAI_AGENT(NAME, ENV)                                                 \
class NAME;                                                                     \
class NAME##Base : public RLLibOpenAiGymAgent                                   \
{                                                                               \
    public: NAME##Base() : RLLibOpenAiGymAgent() {}                             \
    public: ~NAME##Base() {}                                                    \
};                                                                              \
class NAME##Factory : public RLLibOpenAiGymAgentFactoryTemplate<NAME>           \
{                                                                               \
    public: NAME##Factory() { RLLibOpenAiGymAgentRegistry::getInstance().registerInstance(getName(), getEnv(), this); } \
    public: const char* getName() const { return #NAME; }                       \
    public: const char* getEnv()  const { return #ENV;  }                       \
};                                                                              \

#define OPENAI_AGENT_MAKE(NAME)                                                 \
RLLibOpenAiGymAgentLoader<NAME##Factory> __the##NAME##Factory;                  \


#endif /* OPENAI_GYM_RLLIBOPENAIGYMAGENTMACRO_H_ */
