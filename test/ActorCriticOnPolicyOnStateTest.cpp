/*
 * ActorCriticOnPolicyOnStateTest.cpp
 *
 *  Created on: Oct 28, 2013
 *      Author: sam
 */

#include "ActorCriticOnPolicyOnStateTest.h"

ActorCriticOnPolicyOnStateTest::ActorCriticOnPolicyOnStateTest() :
    gamma(0.9), rewardRequired(0.6), mu(0.2), sigma(0.5)
{
  random = new Random<double>();
  problem = new NoStateProblem(random, mu, sigma);
  projector = new NoStateProblemProjector();
  toStateAction = new NoStateProblemStateToStateAction(projector, problem->getContinuousActions());

  alpha_v = alpha_u = alpha_r = lambda = 0;

  criticE = 0;
  critic = 0;
  policyDistribution = 0;
  actorMuE = 0;
  actorSigmaE = 0;
  actorTraces = 0;
  actor = 0;
  control = 0;
  agent = 0;
  sim = 0;
}

ActorCriticOnPolicyOnStateTest::~ActorCriticOnPolicyOnStateTest()
{
  delete random;
  delete problem;
  delete projector;
  delete toStateAction;
  if (criticE)
    delete criticE;
  if (policyDistribution)
    delete policyDistribution;
  if (actorMuE)
    delete actorMuE;
  if (actorSigmaE)
    delete actorSigmaE;
  if (actorTraces)
    delete actorTraces;
  deleteObjects();
}

void ActorCriticOnPolicyOnStateTest::deleteObjects()
{
  if (policyDistribution)
    delete policyDistribution;
  policyDistribution = 0;
  if (critic)
    delete critic;
  critic = 0;
  if (actor)
    delete actor;
  actor = 0;
  if (control)
    delete control;
  control = 0;
  if (agent)
    delete agent;
  agent = 0;
  if (sim)
    delete sim;
  sim = 0;
}

void ActorCriticOnPolicyOnStateTest::checkDistribution(
    PolicyDistribution<double>* policyDistribution)
{
  alpha_v = 0.1 / projector->vectorNorm();
  alpha_u = 0.01 / projector->vectorNorm();

  critic = new TD<double>(alpha_v, gamma, projector->dimension());
  actor = new Actor<double>(alpha_u, policyDistribution);
  control = new ActorCritic<double>(critic, actor, projector, toStateAction);

  agent = new LearnerAgent<double>(control);
  sim = new RLRunner<double>(agent, problem, 10000, 1, 1);
  sim->run();
  double discReward = sim->episodeR / sim->timeStep;
  std::cout << discReward << std::endl;
  ASSERT(discReward > rewardRequired);
  deleteObjects();
}

void ActorCriticOnPolicyOnStateTest::testNormalDistribution()
{
  policyDistribution = new NormalDistribution<double>(random, problem->getContinuousActions(), 0.5,
      1.0, projector->dimension());
  checkDistribution(policyDistribution);
}

void ActorCriticOnPolicyOnStateTest::testNormalDistributionMeanAdjusted()
{
  policyDistribution = new NormalDistributionSkewed<double>(random, problem->getContinuousActions(),
      0.5, 1.0, projector->dimension());
  checkDistribution(policyDistribution);
}

void ActorCriticOnPolicyOnStateTest::testNormalDistributionWithEligibility()
{
  srand(0); // Consistent
  lambda = 0.2;
  criticE = new ATrace<double>(projector->dimension());
  alpha_v = 0.5 / projector->vectorNorm();
  critic = new TDLambda<double>(alpha_v, gamma, lambda, criticE);

  actorMuE = new ATrace<double>(projector->dimension());
  actorSigmaE = new ATrace<double>(projector->dimension());
  actorTraces = new Traces<double>();
  actorTraces->push_back(actorMuE);
  actorTraces->push_back(actorSigmaE);

  alpha_u = 0.1 / projector->vectorNorm();

  policyDistribution = new NormalDistribution<double>(random, problem->getContinuousActions(), 0.5,
      1.0, projector->dimension());

  actor = new ActorLambda<double>(alpha_u, gamma, lambda, policyDistribution, actorTraces);
  control = new ActorCritic<double>(critic, actor, projector, toStateAction);

  agent = new LearnerAgent<double>(control);
  sim = new RLRunner<double>(agent, problem, 1000, 1, 1);
  sim->run();
  double discReward = sim->episodeR / sim->timeStep;
  std::cout << discReward << std::endl;
  ASSERT(discReward > rewardRequired);
  deleteObjects();
}

void ActorCriticOnPolicyOnStateTest::run()
{
  testNormalDistribution();
  testNormalDistributionMeanAdjusted();
  testNormalDistributionWithEligibility();
}

RLLIB_TEST_MAKE(ActorCriticOnPolicyOnStateTest)
