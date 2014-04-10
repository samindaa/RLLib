/*
 * IDBDTest.cpp
 *
 *  Created on: Nov 18, 2013
 *      Author: sam
 */

#include "IDBDTest.h"

void IDBDTest::run()
{
  testIDBD();
  testK1();
  testAutostep();
}

void IDBDTest::testIDBD()
{
  {
    NoisyInputSumEvaluation noisyInputSumEvaluation;
    IDBD<double> idbd(noisyInputSumEvaluation.nbInputs, 0.001);
    double error = noisyInputSumEvaluation.evaluateLearner(&idbd);
    std::cout << error << std::endl;
    Assert::assertObjectEquals(2.0, error, 0.1);
  }

  {
    NoisyInputSumEvaluation noisyInputSumEvaluation;
    IDBD<double> idbd(noisyInputSumEvaluation.nbInputs, 0.01);
    double error = noisyInputSumEvaluation.evaluateLearner(&idbd);
    std::cout << error << std::endl;
    Assert::assertObjectEquals(1.5, error, 0.1);
  }
}

void IDBDTest::testK1()
{
  {
    NoisyInputSumEvaluation noisyInputSumEvaluation;
    K1<double> k1(noisyInputSumEvaluation.nbInputs, 0.001);
    double error = noisyInputSumEvaluation.evaluateLearner(&k1);
    std::cout << error << std::endl;
    Assert::assertObjectEquals(1.5, error, 0.1);
  }

  {
    NoisyInputSumEvaluation noisyInputSumEvaluation;
    K1<double> k1(noisyInputSumEvaluation.nbInputs, 0.01);
    double error = noisyInputSumEvaluation.evaluateLearner(&k1);
    std::cout << error << std::endl;
    Assert::assertObjectEquals(1.0, error, 0.1);
  }
}

void IDBDTest::testAutostep()
{
  NoisyInputSumEvaluation noisyInputSumEvaluation;
  Autostep<double> autostep(noisyInputSumEvaluation.nbInputs);
  double error = noisyInputSumEvaluation.evaluateLearner(&autostep);
  std::cout << error << std::endl;
  Assert::assertObjectEquals(2.4, error, 0.1);
}

RLLIB_TEST_MAKE(IDBDTest)
