/*
 * MountainCarAgent.h
 *
 *  Created on: Jun 25, 2016
 *      Author: sabeyruw
 */

#ifndef OPENAI_GYM_MOUNTAINCARAGENT_H_
#define OPENAI_GYM_MOUNTAINCARAGENT_H_

#include "ControlAlgorithm.h"
#include "RLLibOpenAiGymAgent.h"

// Env
class MountainCar: public OpenAiGymRLProblem
{
  protected:
    // Global variables:
    RLLib::Range<double>* positionRange;
    RLLib::Range<double>* velocityRange;

  public:
    MountainCar() :
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

    virtual ~MountainCar()
    {
      delete positionRange;
      delete velocityRange;
    }

    void updateTRStep()
    {
      output->o_tp1->setEntry(0, positionRange->toUnit(step_tp1->observation_tp1.at(0)));
      output->o_tp1->setEntry(1, velocityRange->toUnit(step_tp1->observation_tp1.at(1)));

      output->observation_tp1->setEntry(0, step_tp1->observation_tp1.at(0));
      output->observation_tp1->setEntry(1, step_tp1->observation_tp1.at(1));

    }

    bool endOfEpisode() const
    {
      return (step_tp1->episode_state_tp1 == 3);
    }

    double r() const
    {
      return step_tp1->reward_tp1;
    }

    double z() const
    {
      return 0.0f;
    }

};

class MountainCarAgent: public RLLibOpenAiGymAgent
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
    MountainCarAgent();
    virtual ~MountainCarAgent();
    const RLLib::Action<double>* toRLLibStep();
};

#endif /* OPENAI_GYM_MOUNTAINCARAGENT_H_ */
