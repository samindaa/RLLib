/*
 * RLLibOpenAiGymAgentRegistry.h
 *
 *  Created on: Jun 25, 2016
 *      Author: sabeyruw
 */

#ifndef OPENAI_GYM_RLLIBOPENAIGYMAGENTREGISTRY_H_
#define OPENAI_GYM_RLLIBOPENAIGYMAGENTREGISTRY_H_

#include "RLLibOpenAiGymAgent.h"
#include "RLLibOpenAiGymAgentFactory.h"

#include <iostream>
#include <unordered_map>

class RLLibOpenAiGymAgentRegistry
{
  protected:
    class Entry
    {
      public:
        std::string name;
        std::string env;
        RLLibOpenAiGymAgentFactory* factory;

        Entry(const std::string& name, const std::string& env, RLLibOpenAiGymAgentFactory* factory) :
            name(name), env(env), factory(factory)
        {
        }

    };
    std::unordered_map<std::string, Entry*> registry; // env, entry

  public:

    static RLLibOpenAiGymAgentRegistry& getInstance();

    RLLibOpenAiGymAgent* make(const std::string& env);
    void registerInstance(const std::string& name, const std::string& env,
        RLLibOpenAiGymAgentFactory* factory);

  private:
    RLLibOpenAiGymAgentRegistry();
    ~RLLibOpenAiGymAgentRegistry();
    RLLibOpenAiGymAgentRegistry(RLLibOpenAiGymAgentRegistry const&) = delete;
    RLLibOpenAiGymAgentRegistry& operator=(RLLibOpenAiGymAgentRegistry const&) = delete;
};

#endif /* OPENAI_GYM_RLLIBOPENAIGYMAGENTREGISTRY_H_ */
