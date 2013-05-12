/*
 * Test.cpp
 *
 * Created on: May 12, 2013
 *     Author: sam
 *
 * Runs the test suit.
 *
 */

#include "VectorTest.h"
#include "TraceTest.h"
#include "LearningAlgorithmTest.h"

int main(int argc, char** argv)
{
  cout << "*** VectorTest starts ... " << endl;
  SparseVectorTest sparseVectorTest;
  sparseVectorTest.run();
  cout << "*** VectorTest ends ... " << endl;

  cout << "*** TraceTest starts ... " << endl;
  TraceTest traceTest;
  traceTest.run();
  cout << "*** TraceTest ends ... " << endl;

  cout << "*** LearningAlgorithmTest starts " << endl;
  SupervisedAlgorithmTest supervisedAlgorithmTest;
  supervisedAlgorithmTest.run();
  cout << "*** LearningAlgorithmTest ends " << endl;

  return EXIT_SUCCESS;
}

