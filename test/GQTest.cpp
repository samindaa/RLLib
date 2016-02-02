/*
 * GQTest.cpp
 *
 *  Created on: Nov 6, 2013
 *      Author: sam
 */

#include "GQTest.h"

GreedyGQFactory::GreedyGQFactory(const double& gamma, const double& alpha_theta,
    const double& alpha_w, const double& lambda) :
    gamma(gamma), alpha_theta(alpha_theta), alpha_w(alpha_w), lambda(lambda)
{
}

GreedyGQFactory::~GreedyGQFactory()
{
  for (std::vector<OffPolicyControlLearner<double>*>::iterator iter =
      offPolicyControlLearners.begin(); iter != offPolicyControlLearners.end(); ++iter)
    delete *iter;
  for (std::vector<Predictor<double>*>::iterator iter = predictors.begin();
      iter != predictors.end(); ++iter)
    delete *iter;
  for (std::vector<Trace<double>*>::iterator iter = traces.begin(); iter != traces.end(); ++iter)
    delete *iter;
}

OffPolicyControlLearner<double>* GreedyGQFactory::createLearner(Actions<double>* actions,
    StateToStateAction<double>* toStateAction, Policy<double>* target, Policy<double>* behavior)
{
  Trace<double>* e = new ATrace<double>(toStateAction->dimension() * actions->dimension());
  traces.push_back(e);
  GQ<double>* gq = new GQ<double>(alpha_theta, alpha_w, gamma, lambda, e);
  predictors.push_back(gq);
  OffPolicyControlLearner<double>* controlGQ = new GreedyGQ<double>(target, behavior, actions,
      toStateAction, gq);
  offPolicyControlLearners.push_back(controlGQ);
  return controlGQ;
}

double GreedyGQFactory::getGamma() const
{
  return gamma;
}

double GreedyGQFactory::getLambda() const
{
  return lambda;
}

void GQTest::testOnPolicyGQ()
{
  testGQOnRandomWalk(0.5, 0.5, new GreedyGQFactory(1.0, 0.01, 0.0, 0.0));
  testGQOnRandomWalk(0.5, 0.5, new GreedyGQFactory(0.9, 0.01, 0.0, 0.0));
  testGQOnRandomWalk(0.5, 0.5, new GreedyGQFactory(0.9, 0.01, 0.0, 0.1));
  testGQOnRandomWalk(0.5, 0.5, new GreedyGQFactory(0.9, 0.01, 0.5, 0.1));
}

void GQTest::testOffPolicyGQ()
{
  testGQOnRandomWalk(0.3, 0.5, new GreedyGQFactory(0.9, 0.01, 0.0, 0.0));
  testGQOnRandomWalk(0.3, 0.5, new GreedyGQFactory(0.9, 0.01, 0.0, 0.1));
  testGQOnRandomWalk(0.3, 0.5, new GreedyGQFactory(0.9, 0.01, 0.5, 0.1));
}

void GQTest::testGQOnRandomWalk(const double& targetLeftProbability,
    const double& behaviourLeftProbability, OffPolicyLearnerFactory* learnerFactory)
{
  Timer timer;
  timer.start();
  Random<double>* random = new Random<double>;
  RandomWalk* problem = new RandomWalk(random);

  Policy<double>* behaviorPolicy = RandomWalk::newPolicy(random, problem->getActions(),
      behaviourLeftProbability);
  Policy<double>* targetPolicy = RandomWalk::newPolicy(random, problem->getActions(),
      targetLeftProbability);
  problem->setPolicy(behaviorPolicy);
  FSGAgentState* agentState = new FSGAgentState(problem);
  OffPolicyControlLearner<double>* learner = learnerFactory->createLearner(problem->getActions(),
      agentState, targetPolicy, behaviorPolicy);
  Vector<double>* vFun = new PVector<double>(agentState->dimension());

  int nbEpisode = 0;
  const PVector<double> solution = agentState->computeSolution(targetPolicy,
      learnerFactory->getGamma(), learnerFactory->getLambda());
  while (FiniteStateGraph::distanceToSolution(&solution, vFun) > 0.05)
  {
    FiniteStateGraph::StepData stepData = agentState->step();
    if (stepData.v_t()->empty())
      learner->initialize(stepData.v_tp1());
    else
      learner->learn(stepData.v_t(), stepData.a_t, stepData.v_tp1(), stepData.r_tp1, 0);
    if (stepData.s_tp1->v()->empty())
    {
      ++nbEpisode;
      //double error = distanceToSolution(solution, td->weights());
      //std::cout << "nbEpisode=" << nbEpisode << " error=" << error << std::endl;
      Assert::assertPasses(nbEpisode < 100000);
    }
    computeValueFunction(learner, agentState, vFun);
  }
  timer.stop();
  Assert::assertPasses(nbEpisode > 100);

  const Predictor<double>* predictor = learner->predictor();
  const Vector<double>* v = predictor->weights();
  Assert::checkValues(v);

  double error = FiniteStateGraph::distanceToSolution(&solution, vFun);
  std::cout << "## nbEpisode=" << nbEpisode << " ||v||=" << v->dimension() << " error=" << error
      << " elapsedTime(ms)=" << timer.getElapsedTimeInMilliSec() << std::endl;
  std::cout << vFun << std::endl;
  std::cout << v << std::endl;

  delete random;
  delete targetPolicy;
  delete agentState;
  delete problem;
  delete learnerFactory;
  delete vFun;
}

void GQTest::computeValueFunction(const OffPolicyControlLearner<double>* learner,
    const FSGAgentState* agentState, Vector<double>* vFun)
{
  const std::map<GraphState*, int>* stateIndexes = agentState->getStateIndexes();
  for (std::map<GraphState*, int>::const_iterator iter = stateIndexes->begin();
      iter != stateIndexes->end(); ++iter)
    vFun->setEntry(iter->second, learner->computeValueFunction(iter->first->v()));
}

void GQTest::run()
{
  testOnPolicyGQ();
  testOffPolicyGQ();
}

RLLIB_TEST_MAKE(GQTest)

