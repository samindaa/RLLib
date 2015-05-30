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
 * MountainCarTest.h
 *
 *  Created on: May 15, 2013
 *      Author: sam
 */

#ifndef MOUNTAINCARTEST_H_
#define MOUNTAINCARTEST_H_

#include "Test.h"

RLLIB_TEST(MountainCarTest)

class MountainCarTest: public MountainCarTestBase
{
  public:
    MountainCarTest()
    {
    }

    virtual ~MountainCarTest()
    {
    }
    void run();

  private:
    void testSarsaTabularActionMountainCar();
    void testOnPolicyBoltzmannRTraceTabularActionCar();
    void testSarsaMountainCar();
    void testSarsaTrueMountainCar();
    void testSarsaAdaptiveMountainCar();
    void testSarsaAdaptiveMountainCar2();
    void testExpectedSarsaMountainCar();
    void testQMountainCar();
    void testGreedyGQOnPolicyMountainCar();
    void testGreedyGQMountainCar();
    void testSoftmaxGQOnMountainCar();
    void testOffPACMountainCar();
    void testOffPACOnPolicyMountainCar();

    void testOnPolicyContinousActionCar(const int& nbMemory, const double& lambda,
        const double& gamma, double alpha_v, double alpha_u);
    void testOnPolicyBoltzmannATraceCar();
    void testOnPolicyBoltzmannRTraceCar();
    void testOnPolicyContinousActionCar();

    void testOnPolicyBoltzmannATraceNaturalActorCriticCar();
};

#endif /* MOUNTAINCARTEST_H_ */
