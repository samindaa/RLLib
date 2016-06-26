/*
 * RLLibOpenAiGymAgentRegistry.cpp
 *
 *  Created on: Jun 25, 2016
 *      Author: sabeyruw
 */

#include "RLLibOpenAiGymAgentRegistry.h"
//
#include "MountainCarAgent.h"

RLLibOpenAiGymAgent* RLLibOpenAiGymAgentRegistry::make(const std::string& name)
{ // TODO
  std::cout << "name: [" << name << "]" << std::endl;
  if (name == "MountainCar-v0")
  {
    return new MountainCarAgent();
  }
  else
  {
    return NULL;
  }
}
