/*
 * RLLibOpenAiGymAgentFactory.h
 *
 *  Created on: Jul 21, 2016
 *      Author: sabeyruw
 */

#ifndef OPENAI_GYM_RLLIBOPENAIGYMAGENTFACTORY_H_
#define OPENAI_GYM_RLLIBOPENAIGYMAGENTFACTORY_H_

#include "RLLibOpenAiGymAgent.h"

class RLLibOpenAiGymAgentFactory
{
  public:
    virtual ~RLLibOpenAiGymAgentFactory()
    {
    }
    virtual RLLibOpenAiGymAgent* make() =0;
};

template<typename T>
class RLLibOpenAiGymAgentFactoryTemplate: public RLLibOpenAiGymAgentFactory
{
  public:
    virtual ~RLLibOpenAiGymAgentFactoryTemplate()
    {
    }

    RLLibOpenAiGymAgent* make()
    {
      return new T();
    }

    virtual const char* getName() const =0;
    virtual const char* getEnv() const =0;

};

#endif /* OPENAI_GYM_RLLIBOPENAIGYMAGENTFACTORY_H_ */
