/*
 * SwingPendulumTest.cpp
 *
 *  Created on: May 15, 2013
 *      Author: sam
 */

#include "SwingPendulumTest.h"

RLLIB_TEST_MAKE(SwingPendulumTest)

void SwingPendulumTest::testOffPACSwingPendulum()
{
  srand(time(0));
  Env<float>* problem = new SwingPendulum;
  Projector<double, float>* projector = new TileCoderHashing<double, float>(1000000, 10, true);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<double, float>(
      projector, &problem->getDiscreteActionList());

  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_w = .0001 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.4;

  Trace<double>* critice = new ATrace<double>(projector->dimension());
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma, lambda, critice);
  double alpha_u = 0.5 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(projector->dimension(),
      &problem->getDiscreteActionList());

  Trace<double>* actore = new ATrace<double>(projector->dimension());
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore);
  ActorOffPolicy<double, float>* actor = new ActorLambdaOffPolicy<double, float>(alpha_u, gamma,
      lambda, target, actoreTraces);

  Policy<double>* behavior = new RandomPolicy<double>(&problem->getDiscreteActionList());
  /*Policy<double>* behavior = new RandomBiasPolicy<double>(
   &problem->getDiscreteActionList());*/
  OffPolicyControlLearner<double, float>* control = new OffPAC<double, float>(behavior, critic,
      actor, toStateAction, projector, gamma);

  Simulator<double, float>* sim = new Simulator<double, float>(control, problem);
  sim->run(1, 5000, 200);
  sim->computeValueFunction();

  delete problem;
  delete projector;
  delete toStateAction;
  delete critice;
  delete critic;
  delete actore;
  delete actoreTraces;
  delete actor;
  delete behavior;
  delete target;
  delete control;
  delete sim;
}

void SwingPendulumTest::testOnPolicySwingPendulum()
{
  srand(time(0));
  Env<float>* problem = new SwingPendulum;
  Projector<double, float>* projector = new TileCoderHashing<double, float>(1000, 10, false);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<double, float>(
      projector, &problem->getContinuousActionList());

  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_u = 0.001 / projector->vectorNorm();
  double alpha_r = .0001;
  double gamma = 1.0;
  double lambda = 0.5;

  Trace<double>* critice = new ATrace<double>(projector->dimension());
  TDLambda<double>* critic = new TDLambda<double>(alpha_v, gamma, lambda, critice);

  PolicyDistribution<double>* policyDistribution = new NormalDistributionScaled<double>(0, 1.0,
      projector->dimension(), &problem->getContinuousActionList());
  Range<double> policyRange(-2.0, 2.0);
  Range<double> problemRange(-2.0, 2.0);
  PolicyDistribution<double>* acting = new ScaledPolicyDistribution<double>(
      &problem->getContinuousActionList(), policyDistribution, &policyRange, &problemRange);

  Trace<double>* actore1 = new ATrace<double>(projector->dimension());
  Trace<double>* actore2 = new ATrace<double>(projector->dimension());
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore1);
  actoreTraces->push_back(actore2);
  ActorOnPolicy<double, float>* actor = new ActorLambda<double, float>(alpha_u, gamma, lambda,
      acting, actoreTraces);

  OnPolicyControlLearner<double, float>* control = new AverageRewardActorCritic<double, float>(
      critic, actor, projector, toStateAction, alpha_r);

  Simulator<double, float>* sim = new Simulator<double, float>(control, problem);
  sim->run(1, 5000, 100, false);
  sim->test(1, 1000);
  sim->computeValueFunction();

  delete problem;
  delete projector;
  delete toStateAction;
  delete critice;
  delete critic;
  delete actore1;
  delete actore2;
  delete actoreTraces;
  delete actor;
  delete policyDistribution;
  delete acting;
  delete control;
  delete sim;
}

void SwingPendulumTest::testOffPACSwingPendulum2()
{
  srand(time(0));
  Env<float>* problem = new SwingPendulum;
  Projector<double, float>* projector = new TileCoderHashing<double, float>(1000000, 10, true);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<double, float>(
      projector, &problem->getDiscreteActionList());

  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_w = .005 / projector->vectorNorm();
  double gamma = 0.99;
  Trace<double>* critice = new AMaxTrace<double>(projector->dimension());
  Trace<double>* criticeML = new MaxLengthTrace<double>(critice, 1000);
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma, 0.4, criticeML);
  double alpha_u = 0.5 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(projector->dimension(),
      &problem->getDiscreteActionList());

  Trace<double>* actore = new AMaxTrace<double>(projector->dimension());
  Trace<double>* actoreML = new MaxLengthTrace<double>(actore, 1000);
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actoreML);
  ActorOffPolicy<double, float>* actor = new ActorLambdaOffPolicy<double, float>(alpha_u, gamma,
      0.4, target, actoreTraces);

  /*Policy<double>* behavior = new RandomPolicy<double>(
   &problem->getActionList());*/
  Policy<double>* behavior = new BoltzmannDistribution<double>(projector->dimension(),
      &problem->getDiscreteActionList());
  OffPolicyControlLearner<double, float>* control = new OffPAC<double, float>(behavior, critic,
      actor, toStateAction, projector, gamma);

  Simulator<double, float>* sim = new Simulator<double, float>(control, problem);
  sim->run(1, 5000, 200);

  delete problem;
  delete projector;
  delete toStateAction;
  delete critice;
  delete criticeML;
  delete critic;
  delete actore;
  delete actoreML;
  delete actoreTraces;
  delete actor;
  delete behavior;
  delete target;
  delete control;
  delete sim;
}

void SwingPendulumTest::run()
{
  testOffPACSwingPendulum();
  testOnPolicySwingPendulum();
  //testOffPACSwingPendulum2();
}

