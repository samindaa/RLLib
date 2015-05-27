/*
 * RoboCupTest.cpp
 *
 *  Created on: Mar 12, 2015
 *      Author: sam
 */

#include "RoboCupTest.h"

RLLIB_TEST_MAKE(RoboCupTest)

RoboCupTest::RoboCupTest()
{
}

RoboCupTest::~RoboCupTest()
{
}

void RoboCupTest::run()
{
  //std::cout << std::endl << std::endl << " *** mc reg    *** " << std::endl;
  //testSarsaMountainCar();
  //std::cout << std::endl << std::endl << " *** mc true   *** " << std::endl;
  //testSarsaTrueMountainCar();
  //std::cout << std::endl << std::endl << " *** mc offpac *** " << std::endl;
  //testOffPACMountainCar();
  //std::cout << std::endl << std::endl << " *** mc3 gq *** " << std::endl;
  //testGreedyGQMountainCar3D();
  //std::cout << std::endl << std::endl << " *** sw avg    *** " << std::endl;
  //testSwingPendulumActorCriticWithEligiblity();
  //std::cout << std::endl << std::endl << " *** gw offpac *** " << std::endl;
  //testOffPACContinuousGridworld();
  testTemperature();
}

// Test cases
void RoboCupTest::testSarsaMountainCar()
{
  Random<double>* random = new Random<double>;
  RLProblem<double>* problem = new MountainCar<double>;
  Hashing<double>* hashing = new MurmurHashing<double>(random, 10000);
  Projector<double>* projector = new TileCoderHashing<double>(hashing, problem->dimension(), 10, 10,
      true);
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());
  Trace<double>* e = new RTrace<double>(projector->dimension());
  double alpha = 0.15 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.3;
  Sarsa<double>* sarsa = new Sarsa<double>(alpha, gamma, lambda, e);
  double epsilon = 0.01;
  Policy<double>* acting = new EpsilonGreedy<double>(random, problem->getDiscreteActions(), sarsa,
      epsilon);
  OnPolicyControlLearner<double>* control = new SarsaControl<double>(acting, toStateAction, sarsa);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  RLRunner<double>* sim = new RLRunner<double>(agent, problem, 5000, 50, 1);
  sim->run();
  //sim->computeValueFunction();

  delete random;
  delete problem;
  delete hashing;
  delete projector;
  delete toStateAction;
  delete e;
  delete sarsa;
  delete acting;
  delete control;
  delete agent;
  delete sim;
}

void RoboCupTest::testSarsaTrueMountainCar()
{
  Random<double>* random = new Random<double>;
  RLProblem<double>* problem = new MountainCar<double>;
  Hashing<double>* hashing = new MurmurHashing<double>(random, 10000);
  Projector<double>* projector = new TileCoderHashing<double>(hashing, problem->dimension(), 10, 10,
      true);
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());
  Trace<double>* e = new ATrace<double>(projector->dimension());
  double alpha = 1.0 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.9;
  Sarsa<double>* sarsa = new SarsaTrue<double>(alpha, gamma, lambda, e);
  double epsilon = 0.01;
  Policy<double>* acting = new EpsilonGreedy<double>(random, problem->getDiscreteActions(), sarsa,
      epsilon);
  OnPolicyControlLearner<double>* control = new SarsaControl<double>(acting, toStateAction, sarsa);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  RLRunner<double>* sim = new RLRunner<double>(agent, problem, 5000, 50, 1);
  sim->run();

  delete random;
  delete problem;
  delete hashing;
  delete projector;
  delete toStateAction;
  delete e;
  delete sarsa;
  delete acting;
  delete control;
  delete agent;
  delete sim;
}

void RoboCupTest::testOffPACMountainCar()
{
  Random<float>* random = new Random<float>;
  RLProblem<float>* problem = new MountainCar<float>;
  Hashing<float>* hashing = new MurmurHashing<float>(random, 1000000);
  Projector<float>* projector = new TileCoderHashing<float>(hashing, problem->dimension(), 10, 10,
      true);
  StateToStateAction<float>* toStateAction = new StateActionTilings<float>(projector,
      problem->getDiscreteActions());

  double alpha_v = 0.05 / projector->vectorNorm();
  double alpha_w = 0.0001 / projector->vectorNorm();
  double lambda = 0.0;  //0.4;
  double gamma = 0.99;
  Trace<float>* critice = new ATrace<float>(projector->dimension());
  OffPolicyTD<float>* critic = new GTDLambda<float>(alpha_v, alpha_w, gamma, lambda, critice);
  double alpha_u = 1.0 / projector->vectorNorm();
  PolicyDistribution<float>* target = new BoltzmannDistribution<float>(random,
      problem->getDiscreteActions(), projector->dimension());

  Trace<float>* actore = new ATrace<float>(projector->dimension());
  Traces<float>* actoreTraces = new Traces<float>();
  actoreTraces->push_back(actore);
  ActorOffPolicy<float>* actor = new ActorLambdaOffPolicy<float>(alpha_u, gamma, lambda, target,
      actoreTraces);

  Policy<float>* behavior = new RandomPolicy<float>(random, problem->getDiscreteActions());

  OffPolicyControlLearner<float>* control = new OffPAC<float>(behavior, critic, actor,
      toStateAction, projector);

  RLAgent<float>* agent = new LearnerAgent<float>(control);
  RLRunner<float>* sim = new RLRunner<float>(agent, problem, 5000, 50, 1);
  //sim->setVerbose(false);
  sim->run();
  //sim->computeValueFunction();
  //control->persist("visualization/mcar_offpac.data");

  //control->reset();
  //control->resurrect("visualization/mcar_offpac.data");
  //sim->runEvaluate(10, 10);

  delete random;
  delete problem;
  delete hashing;
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
  delete agent;
  delete sim;
}

void RoboCupTest::testGreedyGQMountainCar3D()
{
  Random<double>* random = new Random<double>;
  RLProblem<double>* problem = new MountainCar3D<double>(random);
  Projector<double>* projector = new MountainCar3DTilesProjector<double>(random);
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());
  Trace<double>* e = new ATrace<double>(projector->dimension(), 0.001);
  Trace<double>* eML = new MaxLengthTrace<double>(e, 2000);
  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_w = .0001 / projector->vectorNorm();
  double gamma_tp1 = 0.99;
  double lambda_t = 0.8;
  GQ<double>* gq = new GQ<double>(alpha_v, alpha_w, gamma_tp1, lambda_t, eML);
  //double epsilon = 0.01;
  Policy<double>* behavior = new EpsilonGreedy<double>(random, problem->getDiscreteActions(), gq,
      0.1);
  /*Policy<double>* behavior = new RandomPolicy<double>(
   &problem->getDiscreteActions());*/
  Policy<double>* target = new Greedy<double>(problem->getDiscreteActions(), gq);
  OffPolicyControlLearner<double>* control = new GreedyGQ<double>(target, behavior,
      problem->getDiscreteActions(), toStateAction, gq);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  RLRunner<double>* sim = new RLRunner<double>(agent, problem, 5000, 50, 1);
  sim->run();

  delete random;
  delete problem;
  delete projector;
  delete toStateAction;
  delete e;
  delete eML;
  delete gq;
  delete behavior;
  delete target;
  delete control;
  delete agent;
  delete sim;
}

void RoboCupTest::testSwingPendulumActorCriticWithEligiblity()
{
  Random<double>* random;
  RLProblem<double>* problem;
  Hashing<double>* hashing;
  Projector<double>* projector;
  StateToStateAction<double>* toStateAction;

  double alpha_v;
  double alpha_u;
  double alpha_r;
  double gamma;
  double lambda;

  Trace<double>* criticE;
  OnPolicyTD<double>* critic;

  PolicyDistribution<double>* policyDistribution;

  Trace<double>* actorMuE;
  Trace<double>* actorSigmaE;
  Traces<double>* actorTraces;
  ActorOnPolicy<double>* actor;

  OnPolicyControlLearner<double>* control;
  RLAgent<double>* agent;
  RLRunner<double>* sim;

  random = new Random<double>();
  problem = new SwingPendulum<double>;
  hashing = new UNH<double>(random, 1000);
  projector = new TileCoderHashing<double>(hashing, problem->dimension(), 10, 10, true);
  toStateAction = new StateActionTilings<double>(projector, problem->getContinuousActions());

  alpha_v = alpha_u = alpha_r = gamma = lambda = 0;

  criticE = new ATrace<double>(projector->dimension());
  critic = 0;

  policyDistribution = new NormalDistributionScaled<double>(random, problem->getContinuousActions(),
      0, 1.0, projector->dimension());

  actorMuE = new ATrace<double>(projector->dimension());
  actorSigmaE = new ATrace<double>(projector->dimension());
  actorTraces = new Traces<double>();
  actorTraces->push_back(actorMuE);
  actorTraces->push_back(actorSigmaE);

  gamma = 1.0;
  alpha_v = 0.5 / projector->vectorNorm();
  critic = new TD<double>(alpha_v, gamma, projector->dimension());
  alpha_u = 0.05 / projector->vectorNorm();
  actor = new Actor<double>(alpha_u, policyDistribution);
  control = new AverageRewardActorCritic<double>(critic, actor, projector, toStateAction, 0.01);
  agent = new LearnerAgent<double>(control);
  sim = new RLRunner<double>(agent, problem, 5000, 50, 1);
  sim->run();

  delete random;
  delete problem;
  delete hashing;
  delete projector;
  delete toStateAction;
  delete criticE;
  delete policyDistribution;
  delete actorMuE;
  delete actorSigmaE;
  delete actorTraces;
  delete critic;
  delete actor;
  delete control;
  delete agent;
  delete sim;
}

void RoboCupTest::testOffPACContinuousGridworld()
{
  Random<double>* random = new Random<double>;
  RLProblem<double>* problem = new ContinuousGridworld<double>(random);
  Hashing<double>* hashing = new MurmurHashing<double>(random, 1000000);
  Projector<double>* projector = new TileCoderHashing<double>(hashing, problem->dimension(), 10, 10,
      true);
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());

  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_w = 0.0001 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.4;
  Trace<double>* critice = new ATrace<double>(projector->dimension());
  GTDLambda<double>* critic = new GTDLambda<double>(alpha_v, alpha_w, gamma, lambda, critice);
  double alpha_u = 0.001 / projector->vectorNorm();
  PolicyDistribution<double>* target = new BoltzmannDistribution<double>(random,
      problem->getDiscreteActions(), projector->dimension());

  Trace<double>* actore = new ATrace<double>(projector->dimension());
  Traces<double>* actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore);
  ActorOffPolicy<double>* actor = new ActorLambdaOffPolicy<double>(alpha_u, gamma, lambda, target,
      actoreTraces);

  Policy<double>* behavior = new RandomPolicy<double>(random, problem->getDiscreteActions());
  OffPolicyControlLearner<double>* control = new OffPAC<double>(behavior, critic, actor,
      toStateAction, projector);

  RLAgent<double>* agent = new LearnerAgent<double>(control);
  RLRunner<double>* sim = new RLRunner<double>(agent, problem, 5000, 50, 1);
  //sim->setTestEpisodesAfterEachRun(true);
  //sim->setVerbose(false);
  sim->run();
  //sim->computeValueFunction();

  //control->persist("visualization/cgw_offpac.data");

  //control->reset();
  //control->resurrect("visualization/cgw_offpac.data");
  //sim->runEvaluate(100);

  delete random;
  delete problem;
  delete hashing;
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
  delete agent;
  delete sim;
}

void RoboCupTest::testTemperature()
{
  Random<double>* random = new Random<double>;
  RLRProblem<double>* problem = new RLRProblem<double>(random);
  Hashing<double>* hashing = new MurmurHashing<double>(random, 512);
  Projector<double>* projector = new TileCoderHashing<double>(hashing, problem->dimension(), //
      28, 1, true);
  StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
      problem->getDiscreteActions());
  Trace<double>* e = new ATrace<double>(projector->dimension());
  double alpha_v = 0.2 / projector->vectorNorm();
  double alpha_w = 0.000001 / projector->vectorNorm();
  double lambda_t = 0.4;
  GQ<double>* gq = new GQ<double>(alpha_v, alpha_w, 1.0f, lambda_t, e);
  Vector<double>* targetDistribution = new PVector<double>(
      problem->getDiscreteActions()->dimension());
  targetDistribution->setEntry(0, 1.0f);
  Policy<double>* target = new ConstantPolicy<double>(random, problem->getDiscreteActions(),
      targetDistribution);
  OffPolicyControlLearner<double>* control = new GreedyGQ<double>(target, target,
      problem->getDiscreteActions(), toStateAction, gq);

  RLAgent<double>* agent = new RLRAgent<double>(gq, control);
  RLRunner<double>* sim = new RLRunner<double>(agent, problem, -1, 200, 1);
  sim->run();

  // prediction test
  const std::vector<double>& xvec = problem->getLKneePitchVec();
  const std::vector<double>& tvec = problem->getRemainingTimeVec();

  assert(xvec.size() == tvec.size());

  const std::string pdata =
      "/home/sam/School/conf_papers/papers/RC15-Symposium/RLLibPaper/prediction.txt";
  std::ofstream pout(pdata.c_str());
  Vector<double>* inx = new PVector<double>(problem->dimension());
  for (size_t i = 0; i < xvec.size(); ++i)
  {
    double xx = xvec.at(i);
    double tt = tvec.at(i);
    if (xx > 80.0f)
      tt = 0.01f;
    inx->setEntry(0, problem->getTemperatureRange()->toUnit(xx));
    double pp = gq->predict(
        toStateAction->stateActions(inx)->at(*problem->getDiscreteActions()->begin()));
    pout << xx << " " << tt << " " << pp << std::endl;
  }
  pout.close();
  delete inx;

  delete random;
  delete problem;
  delete hashing;
  delete projector;
  delete toStateAction;
  delete e;
  delete gq;
  delete targetDistribution;
  delete target;
  delete control;
  delete agent;
  delete sim;

}
