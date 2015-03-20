/*
 * Copyright 2015 Saminda Abeyruwan (saminda@cs.miami.edu)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * PendulumOnPolicyLearning.h
 *
 *  Created on: Mar 11, 2013
 *      Author: sam
 */

#ifndef PENDULUMONPOLICYLEARNING_H_
#define PENDULUMONPOLICYLEARNING_H_

#include "ControlAlgorithm.h"
#include "SwingPendulum.h"
#include "Test.h"

RLLIB_TEST(ActorCriticOnPolicyControlLearnerPendulumTest)

class ActorCriticOnPolicyControlLearnerPendulumTest: public ActorCriticOnPolicyControlLearnerPendulumTestBase
{
  protected:
    Random<double>* random;
    RLProblem<double>* problem;
    Hashing<double>* hashing;
    Projector<double>* projector;
    StateToStateAction<double>* toStateAction;

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
    ActorOnPolicy<double>* actor;

    OnPolicyControlLearner<double>* control;
    RLAgent<double>* agent;
    RLRunner<double>* sim;

  public:
    ActorCriticOnPolicyControlLearnerPendulumTest();
    ~ActorCriticOnPolicyControlLearnerPendulumTest();

    void run();

  private:
    void testRandom();
    void testActorCritic();
    void testActorCriticWithEligiblity();

    double evaluate();
    void deleteObjects();
};

#endif /* PENDULUMONPOLICYLEARNING_H_ */
