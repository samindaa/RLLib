/*
 * PendulumOnPolicyLearning.h
 *
 *  Created on: Mar 11, 2013
 *      Author: sam
 */

#ifndef PENDULUMONPOLICYLEARNING_H_
#define PENDULUMONPOLICYLEARNING_H_

#include "ControlAlgorithm.h"
#include "SwingPendulum.h"
#include "Simulator.h"
#include "HeaderTest.h"

using namespace RLLib;

class ActorCriticOnPolicyControlLearnerPendulumTest: public TestBase
{
  protected:
    Env<float>* problem;
    Projector<double, float>* projector;
    StateToStateAction<double, float>* toStateAction;

    double alpha_v;
    double alpha_u;
    double alpha_r;
    double gamma;
    double lambda;

    Trace<double>* criticE;
    OnPolicyTD<double>* critic;

    PolicyDistribution<double>* policyDistribution;

    Trace<double>* actorMuE;
    Trace<double>* actorSigmaE;
    Traces<double>* actorTraces;
    ActorOnPolicy<double, float>* actor;

    OnPolicyControlLearner<double, float>* control;
    Simulator<double, float>* sim;

  public:
    ActorCriticOnPolicyControlLearnerPendulumTest();
    ~ActorCriticOnPolicyControlLearnerPendulumTest();

    void run();

  private:
    void testRandom();
    void testActorCritic();
    void testActorCriticWithEligiblity();

    void evaluate();
    void deleteObjects();
};

#endif /* PENDULUMONPOLICYLEARNING_H_ */
