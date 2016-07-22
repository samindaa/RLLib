/*
 * RLLibOpenAiGymAgentRegistry.cpp
 *
 *  Created on: Jun 25, 2016
 *      Author: sabeyruw
 */

#include "RLLibOpenAiGymAgentRegistry.h"

RLLibOpenAiGymAgentRegistry::RLLibOpenAiGymAgentRegistry()
{
}

RLLibOpenAiGymAgentRegistry::~RLLibOpenAiGymAgentRegistry()
{
  for (auto iter = registry.begin(); iter != registry.end(); ++iter)
  {
    delete iter->second;
  }
}

RLLibOpenAiGymAgentRegistry& RLLibOpenAiGymAgentRegistry::getInstance()
{
  static RLLibOpenAiGymAgentRegistry theInstance;
  return theInstance;
}

RLLibOpenAiGymAgent* RLLibOpenAiGymAgentRegistry::make(const std::string& env)
{
  // Register OpenAI Gym agents here
  std::cout << "env: [" << env << "]" << std::endl;

  std::unordered_map<std::string, Entry*>::iterator iter = registry.find(env);
  if (iter != RLLibOpenAiGymAgentRegistry::getInstance().registry.end())
  {
    return iter->second->factory->make();
  }
  else
  {
    return nullptr;
  }

}

void RLLibOpenAiGymAgentRegistry::registerInstance(const std::string& name, const std::string& env,
    RLLibOpenAiGymAgentFactory* factory)
{
  std::cout << "registering: name: " << name << " env: " << env << std::endl;
  registry.emplace(env, new Entry(name, env, factory));
}
