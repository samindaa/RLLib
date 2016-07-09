/*
 * FuncApproxTest.cpp
 *
 *  Created on: Jun 28, 2016
 *      Author: sabeyruw
 */

#include "FuncApproxTest.h"

RLLIB_TEST_MAKE(FuncApproxTest)

FuncApproxTest::FuncApproxTest()
{
}

FuncApproxTest::~FuncApproxTest()
{
}

double fsin(const double& x)
{
  return std::sin(8.0f * x);
}

double fsinc(const double& x)
{
  return std::sin(8.0f * x) / (8.0f * x);
}

double ff(const double& x)
{
  return x / M_PI_4;
}

void FuncApproxTest::run()
{

  std::vector<double (*)(const double&)> funcVec;
  funcVec.push_back(&fsin);
  funcVec.push_back(&fsinc);
  funcVec.push_back(&ff);

  RLLib::Random<double> random;
  RLLib::Range<double> domain(0, 1.0f);

  const size_t tExamples = 50;
  // Lets create some inputs
  std::vector<double> X;
  std::vector<std::vector<double>> T(funcVec.size(), std::vector<double>());

  for (size_t i = 0; i < tExamples; ++i)
  {
    const double x = domain.choose(&random);
    X.push_back(x);
    for (size_t j = 0; j < funcVec.size(); ++j)
    {
      T[j].push_back(funcVec[j](x));
    }
  }

  // To files
  std::ofstream ofsT("funcApproxT.txt");
  for (size_t i = 0; i < tExamples; ++i)
  {
    ofsT << i << " " << X[i] << " ";
    for (size_t j = 0; j < funcVec.size(); ++j)
    {
      ofsT << T[j][i] << " ";
    }
    ofsT << std::endl;
  }
  ofsT.close();

  // Learn LMS (say)
  int memory = 128;
  int nbTilings = 4;
  double alpha = 0.1;
  int nbInputs = 1;
  double gridResolution = 4.0f;

  Hashing<double>* hashing = new MurmurHashing<double>(&random, memory);
  Projector<double>* projector = new TileCoderHashing<double>(hashing, nbInputs, gridResolution,
      nbTilings, false);

  LearningAlgorithm<double>* predictor = new Adaline<double>(hashing->getMemorySize(),
      alpha / projector->vectorNorm());

  // One round of learning

  Vector<double>* x_t = new PVector<double>(1);

  for (size_t r = 0; r < 5; ++r)
  {
    for (size_t i = 0; i < tExamples; ++i)
    {
      x_t->setEntry(0, X[i]);
      for (size_t j = 0; j < funcVec.size(); ++j)
      {
        predictor->learn(projector->project(x_t, j), T[j][i]);
      }
    }
  }

  std::ofstream ofsP("funcApproxP.txt");

  for (double x = 0.01; x < 1.0f; x += 0.01)
  {
    ofsP << x << " ";
    x_t->setEntry(0, x);
    for (size_t j = 0; j < funcVec.size(); ++j)
    {
      ofsP << funcVec[j](x) << " " << predictor->predict(projector->project(x_t, j)) << " ";
    }
    ofsP << std::endl;
  }

  delete hashing;
  delete projector;
  delete predictor;
  delete x_t;

}
