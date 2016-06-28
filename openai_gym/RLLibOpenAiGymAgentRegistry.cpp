/*
 * RLLibOpenAiGymAgentRegistry.cpp
 *
 *  Created on: Jun 25, 2016
 *      Author: sabeyruw
 */

#include "RLLibOpenAiGymAgentRegistry.h"
//
#include "AcrobotAgent.h"
#include "CartPoleAgent.h"
#include "PendulumAgent.h"
#include "MountainCarAgent.h"

RLLibOpenAiGymAgent* RLLibOpenAiGymAgentRegistry::make(const std::string& name)
{
  // Register OpenAI Gym agents here
  std::cout << "name: [" << name << "]" << std::endl;
  if (name == "MountainCar-v0")
  {
    return new MountainCarAgent();
  }
  else if (name == "Pendulum-v0")
  {
    return new PendulumAgent();
  }
  else if (name == "Acrobot-v0")
  {
    return new AcrobotAgent();
  }
  else if (name == "CartPole-v0")
  {
    return new CartPoleAgent();
  }
  else
  {
    return NULL;
  }
}
