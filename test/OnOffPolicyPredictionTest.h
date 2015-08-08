/*
 * OnOffPolicyPredictionTest.h
 *
 *  Created on: Oct 26, 2013
 *      Author: sam
 */

#ifndef ONOFFPOLICYPREDICTIONTEST_H_
#define ONOFFPOLICYPREDICTIONTEST_H_

#include "Test.h"
//
#include "StateGraph.h"

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
    virtual const Vector<double>* getLambdaVector() =0;
    virtual double precision() =0;

  protected:
    Vector<double>* createLambdaVector(const int& vectorSize)
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
    Random<double>* random;
    LineProblem* lineProblem;
    RandomWalk* randomWalkProblem;
    RandomWalk2* randomWalk2Problem;

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
    void testOnRandomWalk2Problem();
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

    const Vector<double>* getLambdaVector()
    {
      return createLambdaVector(0);
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

    const Vector<double>* getLambdaVector()
    {
      Vector<double>* vec = createLambdaVector(2);
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

    const Vector<double>* getLambdaVector()
    {
      Vector<double>* vec = createLambdaVector(2);
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

    const Vector<double>* getLambdaVector()
    {
      Vector<double>* vec = createLambdaVector(2);
      vec->setEntry(0, 0.5);
      vec->setEntry(1, 1.0);
      return vec;
    }

    double precision()
    {
      return 0.01;
    }
};

class TDLambdaTrueTest2: public OnPolicyTDFactory
{
  private:
    double alpha;

  public:
    TDLambdaTrueTest2(const double& alpha) :
        alpha(alpha)
    {
    }

    OnPolicyTD<double>* create(const double& gamma, const double& lambda, const double& vectorNorm,
        const int& vectorSize)
    {
      Trace<double>* newTrace = new ATrace<double>(vectorSize);
      OnPolicyTD<double>* newOnPolicyTD = new TDLambdaTrue<double>(alpha, gamma, lambda, newTrace);
      newTraces.push_back(newTrace);
      newOnPolicyTDs.push_back(newOnPolicyTD);
      return newOnPolicyTD;
    }

    const Vector<double>* getLambdaVector()
    {
      double labmbda_range[] = { 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.925, 0.95,
          0.9725, 1 };

      int size = sizeof(labmbda_range) / sizeof(labmbda_range[0]);
      Vector<double>* vec = createLambdaVector(size);
      for (int i = 0; i < size; ++i)
        vec->setEntry(i, labmbda_range[i]);
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

    const Vector<double>* getLambdaVector()
    {
      Vector<double>* vec = createLambdaVector(2);
      vec->setEntry(0, 0.1);
      vec->setEntry(1, 0.2);
      return vec;
    }

    double precision()
    {
      return 0.05;
    }
};

class GTDLambdaTrueTest: public OffPolicyTDFactory
{
  public:
    OnPolicyTD<double>* create(const double& gamma, const double& lambda, const double& vectorNorm,
        const int& vectorSize)
    {
      Trace<double>* e = new ATrace<double>(vectorSize);
      Trace<double>* e_d = new ATrace<double>(vectorSize);
      Trace<double>* e_w = new ATrace<double>(vectorSize);
      OnPolicyTD<double>* newOnPolicyTD = new GTDLambdaTrue<double>(0.01 / vectorNorm,
          0.5 / vectorNorm, gamma, lambda, e, e_d, e_w);
      newTraces.push_back(e);
      newTraces.push_back(e_d);
      newTraces.push_back(e_w);
      newOnPolicyTDs.push_back(newOnPolicyTD);
      return newOnPolicyTD;
    }

    OffPolicyTD<double>* newTD(const double& gamma, const double& lambda, const double& vectorNorm,
        const int& vectorSize)
    {
      Trace<double>* e = new ATrace<double>(vectorSize);
      Trace<double>* e_d = new ATrace<double>(vectorSize);
      Trace<double>* e_w = new ATrace<double>(vectorSize);
      OffPolicyTD<double>* newOffPolicyTD = new GTDLambdaTrue<double>(0.05 / vectorNorm,
          0.1 / vectorNorm, gamma, lambda, e, e_d, e_w);
      newTraces.push_back(e);
      newTraces.push_back(e_d);
      newTraces.push_back(e_w);
      newOffPolicyTDs.push_back(newOffPolicyTD);
      return newOffPolicyTD;
    }

    const Vector<double>* getLambdaVector()
    {
      const double labmbda_range[] = { 0, //
          (1.0f - std::pow(2, -1)), (1.0f - std::pow(2, -2)), (1.0f - std::pow(2, -3)), //
          (1.0f - std::pow(2, -4)), (1.0f - std::pow(2, -5)), (1.0f - std::pow(2, -6)), //
          (1.0f - std::pow(2, -7)), (1.0f - std::pow(2, -8)), (1.0f - std::pow(2, -9)), //
          (1.0f - std::pow(2, -10)), //
          1.0f };
      int size = sizeof(labmbda_range) / sizeof(labmbda_range[0]);
      Vector<double>* vec = createLambdaVector(size);
      for (int i = 0; i < size; ++i)
        vec->setEntry(i, labmbda_range[i]);

      return vec;
    }

    double precision()
    {
      return 0.01;
    }
};

#endif /* ONOFFPOLICYPREDICTIONTEST_H_ */
