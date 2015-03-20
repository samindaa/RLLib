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
 * LearningAlgorithmTest.h
 *
 *  Created on: May 12, 2013
 *      Author: sam
 */

#ifndef LEARNINGALGORITHMTEST_H_
#define LEARNINGALGORITHMTEST_H_

#include "Test.h"
#include "Projector.h"
#include "SupervisedAlgorithm.h"

RLLIB_TEST(SupervisedAlgorithmTest)

class SupervisedAlgorithmTest: public SupervisedAlgorithmTestBase
{
  private:
    Random<double>* random;

  public:
    SupervisedAlgorithmTest();
    virtual ~SupervisedAlgorithmTest();
    void run();

    void linearRegressionWithTileFeatures();
    void logisticRegressionWithTileFeatures();
    void linearRegressionWithRegularFeatures();
    void logisticRegressionWithRegularFeatures();
};

#endif /* LEARNINGALGORITHMTEST_H_ */
