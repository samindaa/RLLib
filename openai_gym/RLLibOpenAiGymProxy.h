/*
 * RLLibOpenAiGymProxy.h
 *
 *  Created on: Jun 25, 2016
 *      Author: sabeyruw
 */

#ifndef OPENAI_GYM_RLLIBOPENAIGYMPROXY_H_
#define OPENAI_GYM_RLLIBOPENAIGYMPROXY_H_

#include <vector>
#include <iostream>
//
#include "RLLibOpenAiGymAgentRegistry.h"

class RLLibOpenAiGymProxy
{
  private:
    RLLibOpenAiGymAgent* agent;

  public:
    RLLibOpenAiGymProxy();
    virtual ~RLLibOpenAiGymProxy();

    std::string toRLLib(const std::string& str);
};

#endif /* OPENAI_GYM_RLLIBOPENAIGYMPROXY_H_ */
