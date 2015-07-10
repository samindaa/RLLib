/*
 * MountainCarModel.cpp
 *
 *  Created on: Oct 17, 2013
 *      Author: sam
 */

#include "MountainCarModel.h"

using namespace RLLibViz;

MountainCarModel::MountainCarModel()
{
  // RLLib:
  random = new Random<double>;
  behaviourEnvironment = new MountainCar<double>;
  evaluationEnvironment = new MountainCar<double>;
  hashing = new MurmurHashing<double>(random, 1000000);
  projector = new TileCoderHashing<double>(hashing, behaviourEnvironment->dimension(), 10, 10,
      true);
  toStateAction = new StateActionTilings<double>(projector,
      behaviourEnvironment->getDiscreteActions());

  alpha_v = 0.05 / projector->vectorNorm();
  alpha_w = 0.0001 / projector->vectorNorm();
  gamma = 0.99;
  lambda = 0.0; //0.4;
  critice = new ATrace<double>(projector->dimension());
  critic = new GTDLambda<double>(alpha_v, alpha_w, gamma, lambda, critice);

  alpha_u = 1.0 / projector->vectorNorm();

  target = new BoltzmannDistribution<double>(random, behaviourEnvironment->getDiscreteActions(),
      projector->dimension());

  actore = new ATrace<double>(projector->dimension());
  actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore);
  actor = new ActorLambdaOffPolicy<double>(alpha_u, gamma, lambda, target, actoreTraces);

  behavior = new RandomPolicy<double>(random, behaviourEnvironment->getDiscreteActions());
  control = new OffPAC<double>(behavior, critic, actor, toStateAction, projector);

  learningAgent = new LearnerAgent<double>(control);
  evaluationAgent = new ControlAgent<double>(control);

  learningRunner = new RLRunner<double>(learningAgent, behaviourEnvironment, 5000);
  evaluationRunner = new RLRunner<double>(evaluationAgent, evaluationEnvironment, 5000);
  learningRunner->setVerbose(false);
  evaluationRunner->setVerbose(false);

  simulators.insert(std::make_pair(simulators.size(), learningRunner));
  simulators.insert(std::make_pair(simulators.size(), evaluationRunner));

}

MountainCarModel::~MountainCarModel()
{
  delete random;
  delete behaviourEnvironment;
  delete evaluationEnvironment;
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
  delete learningAgent;
  delete evaluationAgent;
  delete learningRunner;
  delete evaluationRunner;
}

void MountainCarModel::doLearning(Window* window)
{
  for (Simulators::iterator i = simulators.begin(); i != simulators.end(); ++i)
    i->second->step();

  for (Simulators::iterator i = simulators.begin(); i != simulators.end(); ++i)
  {
    if (i->second->isEndingOfEpisode())
    {
      emit signal_add(window->plotVector[i->first], Vec(i->second->timeStep, 0),
          Vec(i->second->episodeR, 0));
      emit signal_draw(window->plotVector[i->first]);
    }

    emit signal_add(window->problemVector[i->first],
        Vec(i->second->getRLProblem()->getTRStep()->observation_tp1->getEntry(0),
            i->second->getRLProblem()->getTRStep()->observation_tp1->getEntry(1)),
        Vec(0.0, 0.0, 0.0, 1.0));
    emit signal_draw(window->problemVector[i->first]);
  }

  updateValueFunction(window, control, evaluationEnvironment->getTRStep(),
      evaluationEnvironment->getObservationRanges(), evaluationRunner->isEndingOfEpisode(), 1);
}

void MountainCarModel::doEvaluation(Window* window)
{ // FixMe: Sam
}
