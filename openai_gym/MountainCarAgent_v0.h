/*
 * MountainCarAgent.h
 *
 *  Created on: Jun 25, 2016
 *      Author: sabeyruw
 */

#ifndef OPENAI_GYM_MOUNTAINCARAGENT_V0_H_
#define OPENAI_GYM_MOUNTAINCARAGENT_V0_H_

#include "RLLibOpenAiGymAgentMacro.h"

// Env
class MountainCar_v0: public OpenAiGymRLProblem
{
  protected:
    // Global variables:
    RLLib::Range<double>* positionRange;
    RLLib::Range<double>* velocityRange;

  public:
    MountainCar_v0() :
        OpenAiGymRLProblem(2, 3, 1),  //
        positionRange(new RLLib::Range<double>(-1.2, 0.6)), //
        velocityRange(new RLLib::Range<double>(-0.07, 0.07))
    {
      for (int i = 0; i < discreteActions->dimension(); ++i)
      {
        discreteActions->push_back(i, i);
      }

      // subject to change
      continuousActions->push_back(0, 0.0);

      observationRanges->push_back(positionRange);
      observationRanges->push_back(velocityRange);
    }

    virtual ~MountainCar_v0()
    {
      delete positionRange;
      delete velocityRange;
    }

};

OPENAI_AGENT(MountainCarAgent_v0, MountainCar-v0)
class MountainCarAgent_v0: public MountainCarAgent_v0Base
{
  private:
    // An algorithm
    RLLib::Random<double>* random;
    RLLib::Hashing<double>* hashing;
    RLLib::Projector<double>* projector;
    RLLib::StateToStateAction<double>* toStateAction;
    RLLib::Trace<double>* e;
    double alpha_v;
    double gamma;
    double lambda;
    RLLib::Sarsa<double>* sarsa;
    double epsilon;
    RLLib::Policy<double>* acting;
    RLLib::OnPolicyControlLearner<double>* control;
    RLLib::RLAgent<double>* agent;
    RLLib::RLRunner<double>* simulator;

  public:
    MountainCarAgent_v0();
    virtual ~MountainCarAgent_v0();
    const RLLib::Action<double>* step();
};

#endif /* OPENAI_GYM_MOUNTAINCARAGENT_V0_H_ */
