/*
 * AdalineTest.cpp
 *
 *  Created on: Nov 19, 2013
 *      Author: sam
 */

#include "AdalineTest.h"

RLLIB_TEST_MAKE(AdalineTest)
AdalineTest::AdalineTest() :
    random(new Random<double>)
{
}

AdalineTest::~AdalineTest()
{
  delete random;
}

void AdalineTest::run()
{
  testAdaline();
  testAdalineOnTracking();
}

void AdalineTest::testAdaline()
{
  random->reseed(0);
  PVector<double> targetWeights(2);
  targetWeights[0] = 1.0;
  targetWeights[1] = 2.0;
  Adaline<double> adaline(2, 0.05);
  learnTarget(&targetWeights, &adaline);
  std::cout << adaline.weights()->getEntry(0) << " " << adaline.weights()->getEntry(1) << std::endl;
  Assert::assertObjectEquals(1.0, adaline.weights()->getEntry(0), 1e-2);
  Assert::assertObjectEquals(2.0, adaline.weights()->getEntry(1), 1e-2);
}

void AdalineTest::testAdalineOnTracking()
{
  {
    random->reseed(0);
    NoisyInputSumEvaluation noisyInputSumEvaluation;
    Adaline<double> adaline(noisyInputSumEvaluation.nbInputs, 0.0);
    double error = noisyInputSumEvaluation.evaluateLearner(&adaline);
    std::cout << error << std::endl;
    Assert::assertObjectEquals(noisyInputSumEvaluation.nbNonZeroWeights, error, 0.2);
  }
  {
    random->reseed(0);
    NoisyInputSumEvaluation noisyInputSumEvaluation;
    Adaline<double> adaline(noisyInputSumEvaluation.nbInputs, 0.03);
    double error = noisyInputSumEvaluation.evaluateLearner(&adaline);
    std::cout << error << std::endl;
    Assert::assertObjectEquals(3.4, error, 0.1);
  }
}

void AdalineTest::updateFeatures(Vector<double>* features)
{
  for (int i = 0; i < features->dimension(); i++)
    features->setEntry(i, random->nextReal());
}

void AdalineTest::learnTarget(const Vector<double>* targetWeights, Adaline<double>* learner)
{
  int nbUpdate = 0;
  double threshold = 1e-3;
  History<double, 5> history;
  history.fill(threshold);
  PVector<double> features(targetWeights->dimension());
  double target = 0.0f;
  while (history.getSum() > threshold)
  {
    updateFeatures(&features);
    target = targetWeights->dot(&features);
    double error = learner->predict(&features) - target;
    ASSERT(Boundedness::checkValue(error));
    history.add(std::fabs(error));
    learner->learn(&features, target);
    ++nbUpdate;
    Assert::assertPasses(nbUpdate < 100000);
  }
  Assert::assertPasses(nbUpdate > 30);
  Assert::assertObjectEquals(target, learner->predict(&features), threshold * 10);
}

