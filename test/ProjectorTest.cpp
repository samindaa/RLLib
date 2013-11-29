/*
 * Copyright 2013 Saminda Abeyruwan (saminda@cs.miami.edu)
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
 * ProjectorTest.cpp
 *
 *  Created on: May 15, 2013
 *      Author: sam
 */

#include "ProjectorTest.h"

RLLIB_TEST_MAKE(ProjectorTest)

void ProjectorTest::testProjector()
{
  Probabilistic::srand(0);
  int numObservations = 2;
  int memorySize = 512;
  int numTiling = 32;
  bool bias = true;
  SVector<double> w(memorySize + bias);
  for (int t = 0; t < 50; t++)
    w.insertEntry(Probabilistic::nextInt(memorySize), Probabilistic::nextDouble());
  UNH hashing(memorySize);
  TileCoderHashing<double> coder(&hashing, numTiling, bias);
  PVector<double> x(numObservations);
  for (int p = 0; p < 5; p++)
  {
    for (int o = 0; o < numObservations; o++)
      x[o] = Probabilistic::nextDouble() / 0.25;
    const Vector<double>* vect = coder.project(&x);
    cout << w << endl;
    cout << *(const SVector<double>*) vect << endl;
    cout << w.dimension() << " " << vect->dimension() << endl;
    cout << w.dot(vect) << endl;
    cout << "---------" << endl;
  }
}

void ProjectorTest::run()
{
  testProjector();
}

