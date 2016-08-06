/*
 * CartPoleAgent.h
 *
 *  Created on: Jun 27, 2016
 *      Author: sabeyruw
 */

#ifndef OPENAI_GYM_CARTPOLEAGENT_V0_H_
#define OPENAI_GYM_CARTPOLEAGENT_V0_H_

#include "RLLibOpenAiGymAgentMacro.h"

//Env
class CartPole_v0: public OpenAiGymRLProblem
{
  protected:
    // Global variables:
    RLLib::Range<double>* xRange;
    RLLib::Range<double>* xDotRange;
    RLLib::Range<double>* thetaRange;
    RLLib::Range<double>* thetaDotRange;

  public:
    CartPole_v0() :
        OpenAiGymRLProblem(4, 2, 1),  //
        xRange(new RLLib::Range<double>(-2.4, 2.4)), // m
        xDotRange(new RLLib::Range<double>(-2.0f, 2.0f)), // m/s
        thetaRange(new RLLib::Range<double>(-M_PI / 15.0, M_PI / 15.0)), // rad
        thetaDotRange(new RLLib::Range<double>(-M_PI, M_PI)) // rad/s
    {

      for (int i = 0; i < discreteActions->dimension(); ++i)
      {
        discreteActions->push_back(i, i);
      }

      // subject to change
      continuousActions->push_back(0, 0.0);

      observationRanges->push_back(xRange);
      observationRanges->push_back(xDotRange);
      observationRanges->push_back(thetaRange);
      observationRanges->push_back(thetaDotRange);
    }

    virtual ~CartPole_v0()
    {
      delete xRange;
      delete xDotRange;
      delete thetaRange;
      delete thetaDotRange;
    }

};

OPENAI_AGENT(CartPoleAgent_v0, CartPole-v0)
class CartPoleAgent_v0: public CartPoleAgent_v0Base
{
  private:
    // RLLib
    RLLib::Random<double>* random;
    int order;
    RLLib::Projector<double>* projector;
    RLLib::StateToStateAction<double>* toStateAction;
    RLLib::Trace<double>* e;
    double alpha;
    double gamma;
    double lambda;
    RLLib::Sarsa<double>* sarsa;
    double epsilon;
    RLLib::Policy<double>* acting;
    RLLib::OnPolicyControlLearner<double>* control;
    RLLib::RLAgent<double>* agent;
    RLLib::RLRunner<double>* simulator;

  public:
    CartPoleAgent_v0();
    virtual ~CartPoleAgent_v0();
    const RLLib::Action<double>* step();
};

#endif /* OPENAI_GYM_CARTPOLEAGENT_V0_H_ */
