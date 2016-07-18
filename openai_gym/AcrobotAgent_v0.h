/*
 * AcrobotAgent.h
 *
 *  Created on: Jun 27, 2016
 *      Author: sabeyruw
 */

#ifndef OPENAI_GYM_ACROBOTAGENT_V0_H_
#define OPENAI_GYM_ACROBOTAGENT_V0_H_

#include "RLLibOpenAiGymAgent.h"

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
      observationRanges->push_back(theta1DotRange);
      observationRanges->push_back(theta2DotRange);
    }

    virtual ~Acrobot_v0()
    {
      delete thetaRange;
      delete theta1DotRange;
      delete theta2DotRange;
    }

    void updateTRStep()
    {
      output->o_tp1->setEntry(0, thetaRange->toUnit(step_tp1->observation_tp1.at(0)));
      output->o_tp1->setEntry(1, thetaRange->toUnit(step_tp1->observation_tp1.at(1)));
      output->o_tp1->setEntry(2, theta1DotRange->toUnit(step_tp1->observation_tp1.at(2)));
      output->o_tp1->setEntry(3, theta2DotRange->toUnit(step_tp1->observation_tp1.at(3)));

      output->observation_tp1->setEntry(0, step_tp1->observation_tp1.at(0));
      output->observation_tp1->setEntry(1, step_tp1->observation_tp1.at(1));
      output->observation_tp1->setEntry(2, theta1DotRange->bound(step_tp1->observation_tp1.at(2)));
      output->observation_tp1->setEntry(1, theta2DotRange->bound(step_tp1->observation_tp1.at(3)));

    }
};

class AcrobotAgent_v0: public RLLibOpenAiGymAgent
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
    const RLLib::Action<double>* toRLLibStep();
};

#endif /* OPENAI_GYM_ACROBOTAGENT_V0_H_ */
