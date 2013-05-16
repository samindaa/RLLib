/*
 * ProjectorTest.cpp
 *
 *  Created on: May 15, 2013
 *      Author: sam
 */

#include "ProjectorTest.h"

RLLIB_TEST_MAKE(ProjectorTest)

void ProjectorTest::testProjector()
{
  int numObservations = 2;
  int memorySize = 512;
  int numTiling = 32;
  bool bias = true;
  SparseVector<double> w(memorySize + bias);
  for (int t = 0; t < 50; t++)
    w.insertEntry(rand() % memorySize, Random::nextDouble());
  TileCoderHashing<double, float> coder(memorySize, numTiling, bias);
  DenseVector<float> x(numObservations);
  for (int p = 0; p < 5; p++)
  {
    for (int o = 0; o < numObservations; o++)
      x[o] = Random::nextDouble() / 0.25;
    const SparseVector<double>& vect = coder.project(x);
    cout << w << endl;
    cout << vect << endl;
    cout << w.dimension() << " " << vect.dimension() << endl;
    cout << w.dot(vect) << endl;
    cout << "---------" << endl;
  }
}

void ProjectorTest::run()
{
  testProjector();
}

