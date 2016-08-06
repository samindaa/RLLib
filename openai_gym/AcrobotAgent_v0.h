/*
 * AcrobotAgent.h
 *
 *  Created on: Jun 27, 2016
 *      Author: sabeyruw
 */

#ifndef OPENAI_GYM_ACROBOTAGENT_V0_H_
#define OPENAI_GYM_ACROBOTAGENT_V0_H_

#include "RLLibOpenAiGymAgentMacro.h"

//Env
class Acrobot_v0: public OpenAiGymRLProblem
{
  protected:
    // Global variables:
    RLLib::Range<double>* thetaRange;
    RLLib::Range<double>* theta1DotRange;
    RLLib::Range<double>* theta2DotRange;

  public:
    Acrobot_v0() :
        OpenAiGymRLProblem(4, 3, 1),  //
        thetaRange(new RLLib::Range<double>(-M_PI, M_PI)), //
        theta1DotRange(new RLLib::Range<double>(-4.0 * M_PI, 4.0 * M_PI)), //
        theta2DotRange(new RLLib::Range<double>(-9.0 * M_PI, 9.0 * M_PI))
    {

      for (int i = 0; i < discreteActions->dimension(); ++i)
      {
        discreteActions->push_back(i, i);
      }

      // subject to change
      continuousActions->push_back(0, 0.0);

      observationRanges->push_back(thetaRange);
      observationRanges->push_back(thetaRange);
      observationRanges->push_back(theta1DotRange);
      observationRanges->push_back(theta2DotRange);
    }

    virtual ~Acrobot_v0()
    {
      delete thetaRange;
      delete theta1DotRange;
      delete theta2DotRange;
    }
};

OPENAI_AGENT(AcrobotAgent_v0, Acrobot-v0)
class AcrobotAgent_v0: public AcrobotAgent_v0Base
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
    AcrobotAgent_v0();
    virtual ~AcrobotAgent_v0();
    const RLLib::Action<double>* step();
};

#endif /* OPENAI_GYM_ACROBOTAGENT_V0_H_ */
