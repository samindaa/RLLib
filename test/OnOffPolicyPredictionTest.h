/*
 * OnOffPolicyPredictionTest.h
 *
 *  Created on: Oct 26, 2013
 *      Author: sam
 */

#ifndef ONOFFPOLICYPREDICTIONTEST_H_
#define ONOFFPOLICYPREDICTIONTEST_H_

#include "Test.h"

class OnPolicyTDFactory
{
  protected:
    std::vector<OnPolicyTD<double>*> newOnPolicyTDs;
    std::vector<Trace<double>*> newTraces;
    Vector<double>* lambdaVector;

  public:
    OnPolicyTDFactory() :
        lambdaVector(0)
    {
    }
    virtual ~OnPolicyTDFactory()
    {
      for (std::vector<OnPolicyTD<double>*>::iterator iter = newOnPolicyTDs.begin();
          iter != newOnPolicyTDs.end(); ++iter)
        delete *iter;
      for (std::vector<Trace<double>*>::iterator iter = newTraces.begin(); iter != newTraces.end();
          ++iter)
        delete *iter;
      if (lambdaVector)
        delete lambdaVector;
    }
    virtual OnPolicyTD<double>* create(const double& gamma, const double& lambda,
        const double& vectorNorm, const int& vectorSize)=0;
    virtual const Vector<double>* lambdaValues() =0;
    virtual double precision() =0;

  protected:
    Vector<double>* getLambdaVector(const int& vectorSize)
    {
      if (!lambdaVector)
        lambdaVector = new PVector<double>(vectorSize);
      return lambdaVector;
    }
};

class OffPolicyTDFactory: public OnPolicyTDFactory
{
  protected:
    std::vector<OffPolicyTD<double>*> newOffPolicyTDs;

  public:
    OffPolicyTDFactory() :
        OnPolicyTDFactory()
    {
    }

    virtual ~OffPolicyTDFactory()
    {
      for (std::vector<OffPolicyTD<double>*>::iterator iter = newOffPolicyTDs.begin();
          iter != newOffPolicyTDs.end(); ++iter)
        delete *iter;
    }

    virtual OffPolicyTD<double>* newTD(const double& gamma, const double& lambda,
        const double& vectorNorm, const int& vectorSize) =0;

};

RLLIB_TEST(OnOffPolicyPredictionTest)
class OnOffPolicyPredictionTest: public OnOffPolicyPredictionTestBase
{
  protected:
    std::vector<OnPolicyTDFactory*> onPolicyTDFactoryVector;
    std::vector<OffPolicyTDFactory*> offPolicyTDFactoryVector;
    LineProblem* lineProblem;
    RandomWalk* randomWalkProblem;

  public:
    OnOffPolicyPredictionTest();
    ~OnOffPolicyPredictionTest();

  protected:
    void testTD(FiniteStateGraph* graph, OnPolicyTDFactory* factory, const double& lambda,
        const int& nbEpisodeMax);

    void testOffPolicyGTD(RandomWalk* problem, OffPolicyTDFactory* factory, const double& lambda,
        const int& nbEpisodeMax, const double& targetLeftProbability,
        const double& behaviourLeftProbability);

    void registerTDFactories();
    void clearTDFactories();
    void testOnLineProblem();
    void testOnLineProblemWithLambda();
    void testOnRandomWalkProblem();
    void testOnRandomWalkProblemWithLambda();

    void testOffPolicy();
    void testOffPolicyWithLambda();
    int nbEpisodeMax() const;

  public:
    void run();
};

class TDTest: public OnPolicyTDFactory
{
  public:
    OnPolicyTD<double>* create(const double& gamma, const double& lambda, const double& vectorNorm,
        const int& vectorSize)
    {
      OnPolicyTD<double>* newOnPolicyTD = new TD<double>(0.01f / vectorNorm, gamma, vectorSize);
      newOnPolicyTDs.push_back(newOnPolicyTD);
      return newOnPolicyTD;
    }

    const Vector<double>* lambdaValues()
    {
      return getLambdaVector(0);
    }

    double precision()
    {
      return 0.01;
    }
};

class TDLambdaTest: public OnPolicyTDFactory
{
  public:
    OnPolicyTD<double>* create(const double& gamma, const double& lambda, const double& vectorNorm,
        const int& vectorSize)
    {
      Trace<double>* newTrace = new ATrace<double>(vectorSize);
      OnPolicyTD<double>* newOnPolicyTD = new TDLambda<double>(0.01 / vectorNorm, gamma, lambda,
          newTrace);
      newTraces.push_back(newTrace);
      newOnPolicyTDs.push_back(newOnPolicyTD);
      return newOnPolicyTD;
    }

    const Vector<double>* lambdaValues()
    {
      Vector<double>* vec = getLambdaVector(2);
      vec->setEntry(0, 0.5);
      vec->setEntry(1, 1.0);
      return vec;
    }

    double precision()
    {
      return 0.01;
    }
};

class TDLambdaAlphaBoundTest: public OnPolicyTDFactory
{
  public:
    OnPolicyTD<double>* create(const double& gamma, const double& lambda, const double& vectorNorm,
        const int& vectorSize)
    {
      Trace<double>* newTrace = new ATrace<double>(vectorSize);
      OnPolicyTD<double>* newOnPolicyTD = new TDLambdaAlphaBound<double>(0.4 / vectorNorm, gamma,
          lambda, newTrace);
      newTraces.push_back(newTrace);
      newOnPolicyTDs.push_back(newOnPolicyTD);
      return newOnPolicyTD;
    }

    const Vector<double>* lambdaValues()
    {
      Vector<double>* vec = getLambdaVector(2);
      vec->setEntry(0, 0.5);
      vec->setEntry(1, 1.0);
      return vec;
    }

    double precision()
    {
      return 0.01;
    }
};

class TDLambdaTrueTest: public OnPolicyTDFactory
{
  public:
    OnPolicyTD<double>* create(const double& gamma, const double& lambda, const double& vectorNorm,
        const int& vectorSize)
    {
      Trace<double>* newTrace = new ATrace<double>(vectorSize);
      OnPolicyTD<double>* newOnPolicyTD = new TDLambdaTrue<double>(0.01f / vectorNorm, gamma,
          lambda, newTrace);
      newTraces.push_back(newTrace);
      newOnPolicyTDs.push_back(newOnPolicyTD);
      return newOnPolicyTD;
    }

    const Vector<double>* lambdaValues()
    {
      Vector<double>* vec = getLambdaVector(2);
      vec->setEntry(0, 0.5);
      vec->setEntry(1, 1.0);
      return vec;
    }

    double precision()
    {
      return 0.01;
    }
};

class GTDLambdaTest: public OffPolicyTDFactory
{
  public:
    OnPolicyTD<double>* create(const double& gamma, const double& lambda, const double& vectorNorm,
        const int& vectorSize)
    {
      Trace<double>* newTrace = new AMaxTrace<double>(vectorSize);
      OnPolicyTD<double>* newOnPolicyTD = new GTDLambda<double>(0.01 / vectorNorm, 0.5 / vectorNorm,
          gamma, lambda, newTrace);
      newTraces.push_back(newTrace);
      newOnPolicyTDs.push_back(newOnPolicyTD);
      return newOnPolicyTD;
    }

    OffPolicyTD<double>* newTD(const double& gamma, const double& lambda, const double& vectorNorm,
        const int& vectorSize)
    {
      Trace<double>* newTrace = new AMaxTrace<double>(vectorSize);
      OffPolicyTD<double>* newOffPolicyTD = new GTDLambda<double>(0.01 / vectorNorm,
          0.5 / vectorNorm, gamma, lambda, newTrace);
      newTraces.push_back(newTrace);
      newOffPolicyTDs.push_back(newOffPolicyTD);
      return newOffPolicyTD;
    }

    const Vector<double>* lambdaValues()
    {
      Vector<double>* vec = getLambdaVector(2);
      vec->setEntry(0, 0.1);
      vec->setEntry(1, 0.2);
      return vec;
    }

    double precision()
    {
      return 0.05;
    }
};

#endif /* ONOFFPOLICYPREDICTIONTEST_H_ */
