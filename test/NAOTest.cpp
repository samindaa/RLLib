/*
 * NAOTest.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: sam
 */

#include "NAOTest.h"

#define NAO_TRAIN "train"
#define NAO_TEST  "test"

RLLIB_TEST_MAKE(NAOTest)
NAOTest::NAOTest()
{
}

NAOTest::~NAOTest()
{
}

void NAOTest::run()
{
  if (argv[1] == NAO_TRAIN)
  {
    testTrain();
  }
  else if (argv[1] == NAO_TEST)
  {
    testEvaluate();
  }
  else
  {
    std::cout << "Nothing ..." << std::endl;
    RLLib::PVector<double> w;
    w.resurrect("/home/sam/Projects/saminda/RC14/patt_rec/bin_backup/optTheta_audio_4ch.bin");
  }
}

void NAOTest::testTrain()
{
  // OffLine
  {
    Random<float>* random = new Random<float>;
    RLProblem<float>* problem = new MountainCar<float>(random);
    Hashing<float>* hashing = new MurmurHashing<float>(random, 1000000);
    Projector<float>* projector = new TileCoderHashing<float>(hashing, problem->dimension(), 10,
        10);
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
    RLRunner<float>* sim = new RLRunner<float>(agent, problem, 5000, 100, 1);
    //sim->setVerbose(false);
    sim->run();
    control->persist("NAOTest_x32_M.bin");

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
  // OnLine
  {
    Random<double>* random = new Random<double>;
    RLProblem<double>* problem = new SwingPendulum<double>(random);
    Hashing<double>* hashing = new MurmurHashing<double>(random, 1000);
    Projector<double>* projector = new TileCoderHashing<double>(hashing, problem->dimension(), 10,
        10, false);
    StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
        problem->getContinuousActions());

    double alpha_v = 0.1 / projector->vectorNorm();
    double alpha_u = 0.001 / projector->vectorNorm();
    double alpha_r = .0001;
    double gamma = 1.0;
    double lambda = 0.5;

    Trace<double>* critice = new ATrace<double>(projector->dimension());
    TDLambda<double>* critic = new TDLambda<double>(alpha_v, gamma, lambda, critice);

    PolicyDistribution<double>* policyDistribution = new NormalDistributionScaled<double>(random,
        problem->getContinuousActions(), 0, 1.0, projector->dimension());
    Range<double> policyRange(-2.0, 2.0);
    Range<double> problemRange(-2.0, 2.0);
    PolicyDistribution<double>* acting = new ScaledPolicyDistribution<double>(
        problem->getContinuousActions(), policyDistribution, &policyRange, &problemRange);

    Trace<double>* actore1 = new ATrace<double>(projector->dimension());
    Trace<double>* actore2 = new ATrace<double>(projector->dimension());
    Traces<double>* actoreTraces = new Traces<double>();
    actoreTraces->push_back(actore1);
    actoreTraces->push_back(actore2);
    ActorOnPolicy<double>* actor = new ActorLambda<double>(alpha_u, gamma, lambda, acting,
        actoreTraces);

    OnPolicyControlLearner<double>* control = new AverageRewardActorCritic<double>(critic, actor,
        projector, toStateAction, alpha_r);

    RLAgent<double>* agent = new LearnerAgent<double>(control);
    RLRunner<double>* sim = new RLRunner<double>(agent, problem, 5000, 100, 1);
    sim->run();
    control->persist("NAOTest_x32_S.bin");

    delete random;
    delete problem;
    delete hashing;
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
    delete agent;
    delete sim;
  }
}

void NAOTest::testEvaluate()
{
  {
    Random<float>* random = new Random<float>;
    RLProblem<float>* problem = new MountainCar<float>(random);
    Hashing<float>* hashing = new MurmurHashing<float>(random, 1000000);
    Projector<float>* projector = new TileCoderHashing<float>(hashing, problem->dimension(), 10, 10,
        true);
    StateToStateAction<float>* toStateAction = new StateActionTilings<float>(projector,
        problem->getDiscreteActions());

    Trace<float>* critice = new ATrace<float>(projector->dimension());
    OffPolicyTD<float>* critic = new GTDLambda<float>(0, 0, 0, 0, critice);
    PolicyDistribution<float>* target = new BoltzmannDistribution<float>(random,
        problem->getDiscreteActions(), projector->dimension());

    Trace<float>* actore = new ATrace<float>(projector->dimension());
    Traces<float>* actoreTraces = new Traces<float>();
    actoreTraces->push_back(actore);
    ActorOffPolicy<float>* actor = new ActorLambdaOffPolicy<float>(0, 0, 0, target, actoreTraces);

    Policy<float>* behavior = new RandomPolicy<float>(random, problem->getDiscreteActions());

    OffPolicyControlLearner<float>* control = new OffPAC<float>(behavior, critic, actor,
        toStateAction, projector);

    RLAgent<float>* agent = new ControlAgent<float>(control);
    RLRunner<float>* sim = new RLRunner<float>(agent, problem, 5000, 10, 10);

    control->reset();
    control->resurrect("NAOTest_x32_M.bin");
    sim->runEvaluate(10, 10);

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
  // OnLine
  {
    Random<double>* random = new Random<double>;
    RLProblem<double>* problem = new SwingPendulum<double>(random);
    Hashing<double>* hashing = new MurmurHashing<double>(random, 1000);
    Projector<double>* projector = new TileCoderHashing<double>(hashing, problem->dimension(), 10,
        10, false);
    StateToStateAction<double>* toStateAction = new StateActionTilings<double>(projector,
        problem->getContinuousActions());

    Trace<double>* critice = new ATrace<double>(projector->dimension());
    TDLambda<double>* critic = new TDLambda<double>(0, 0, 0, critice);

    PolicyDistribution<double>* policyDistribution = new NormalDistributionScaled<double>(random,
        problem->getContinuousActions(), 0, 1.0, projector->dimension());
    Range<double> policyRange(-2.0, 2.0);
    Range<double> problemRange(-2.0, 2.0);
    PolicyDistribution<double>* acting = new ScaledPolicyDistribution<double>(
        problem->getContinuousActions(), policyDistribution, &policyRange, &problemRange);

    Trace<double>* actore1 = new ATrace<double>(projector->dimension());
    Trace<double>* actore2 = new ATrace<double>(projector->dimension());
    Traces<double>* actoreTraces = new Traces<double>();
    actoreTraces->push_back(actore1);
    actoreTraces->push_back(actore2);
    ActorOnPolicy<double>* actor = new ActorLambda<double>(0, 0, 0, acting, actoreTraces);

    OnPolicyControlLearner<double>* control = new AverageRewardActorCritic<double>(critic, actor,
        projector, toStateAction, 0);

    RLAgent<double>* agent = new ControlAgent<double>(control);
    RLRunner<double>* sim = new RLRunner<double>(agent, problem, 5000, 10, 10);

    control->reset();
    control->resurrect("NAOTest_x32_S.bin");
    sim->run();

    delete random;
    delete problem;
    delete hashing;
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
    delete agent;
    delete sim;
  }
}
