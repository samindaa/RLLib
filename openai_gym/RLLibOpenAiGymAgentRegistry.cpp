/*
 * RLLibOpenAiGymAgentRegistry.cpp
 *
 *  Created on: Jun 25, 2016
 *      Author: sabeyruw
 */

#include "RLLibOpenAiGymAgentRegistry.h"

#include "AcrobotAgentv0_v0.h"
#include "CartPoleAgentv0_v0.h"
#include "MountainCarv0Agent_v0.h"
#include "Pendulumv0Agent_v0.h"

RLLibOpenAiGymAgent* RLLibOpenAiGymAgentRegistry::make(const std::string& name)
{
  // Register OpenAI Gym agents here
  std::cout << "name: [" << name << "]" << std::endl;
  if (name == "MountainCar-v0")
  {
    return new MountainCarAgent_v0();
  }
  else if (name == "Pendulum-v0")
  {
    return new PendulumAgent_v0();
  }
  else if (name == "Acrobot-v0")
  {
    return new AcrobotAgent_v0();
  }
  else if (name == "CartPole-v0")
  {
    return new CartPoleAgent_v0();
  }
  else
  {
    return NULL;
  }
}
