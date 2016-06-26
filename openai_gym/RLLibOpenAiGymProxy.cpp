/*
 * RLLibOpenAiGymProxy.cpp
 *
 *  Created on: Jun 25, 2016
 *      Author: sabeyruw
 */

#include "RLLibOpenAiGymProxy.h"
#include <sstream>
#include <iostream>
#include <iterator>
#include <algorithm>

RLLibOpenAiGymProxy::RLLibOpenAiGymProxy() :
    SyncTcpServer(2345), agent(NULL)
{
}

RLLibOpenAiGymProxy::~RLLibOpenAiGymProxy()
{
  if (!agent)
  {
    delete agent;
  }
}

std::string RLLibOpenAiGymProxy::toRLLib(const std::string& str)
{
  //std::cout << "recv: " << str << std::endl;

  if (!agent)
  {
    size_t idx = str.find("__ENV__");
    if (idx != std::string::npos)
    {
      if (agent)
      {
        delete agent;
      }
      agent = RLLibOpenAiGymAgentRegistry::make(str.substr(idx + 8));

      return agent ? "OK" : "NOT_OK";
    }
  }

  std::stringstream ss(str);
  std::vector<std::string> tokens;
  std::copy(std::istream_iterator<std::string>(ss), std::istream_iterator<std::string>(),
      std::back_inserter(tokens));

  agent->problem->step_tp1->observation_tp1.clear();
  std::stringstream ssEpisodeState(tokens[tokens.size() - 1]);
  ssEpisodeState >> agent->problem->step_tp1->episode_state_tp1;


  std::stringstream ssEpisodeReward(tokens[tokens.size() - 2]);
  ssEpisodeReward >> agent->problem->step_tp1->reward_tp1;

  agent->problem->step_tp1->observation_tp1.resize(tokens.size() - 2);
  for (size_t i = 0; i < tokens.size() - 2; ++i)
  {
    std::stringstream ssStateVar(tokens[i]);
    ssStateVar >> agent->problem->step_tp1->observation_tp1[i];
  }

  const RLLib::Action<double>* action_tp1 = agent->toRLLibStep();

  std::stringstream ssAction_tp1;
  ssAction_tp1 << (action_tp1 ? action_tp1->id() : -1);
  return ssAction_tp1.str();

}

