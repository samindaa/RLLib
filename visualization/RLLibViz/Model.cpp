#include "Model.h"
#include <iostream>

Model::Model(QObject *parent) :
  QObject(parent), window(0)
{
  // RLLib:
  behaviourEnvironment = new ContinuousGridworld;
  evaluationEnvironment = new ContinuousGridworld;
  projector = new TileCoderHashing<double, float>(1000000, 10, true);
  toStateAction = new StateActionTilings<double, float>(projector, &behaviourEnvironment->getDiscreteActionList());

  alpha_v = 0.1 / projector->vectorNorm();
  alpha_w = 0.0001 / projector->vectorNorm();
  gamma = 0.99;
  lambda = 0.4;
  critice = new ATrace<double>(projector->dimension());
  critic = new GTDLambda<double>(alpha_v, alpha_w, gamma, lambda, critice);

  alpha_u = 0.001 / projector->vectorNorm();

  target = new BoltzmannDistribution<double>(projector->dimension(), &behaviourEnvironment->getDiscreteActionList());

  actore = new ATrace<double>(projector->dimension());
  actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore);
  actor = new ActorLambdaOffPolicy<double, float>(alpha_u, gamma, lambda, target, actoreTraces);

  behavior = new RandomPolicy<double>(&behaviourEnvironment->getDiscreteActionList());
  control = new OffPAC<double, float>(behavior, critic, actor, toStateAction, projector, gamma);

  learningRunner = new Simulator<double, float>(control, behaviourEnvironment, 1, 5000, 3000);
  evaluationRunner = new Simulator<double, float>(control, evaluationEnvironment, 1, 5000, 3000);
  evaluationRunner->setEvaluate(true);

  // Timer
  startTimer(5);
}

Model::~Model()
{
  delete behaviourEnvironment;
  delete evaluationEnvironment;
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
  delete learningRunner;
  delete evaluationRunner;
}

void Model::timerEvent(QTimerEvent*)
{
  learningRunner->step();
  evaluationRunner->step();

  std::vector<Simulator<double, float>* > simulators;
  simulators.push_back(learningRunner);
  simulators.push_back(evaluationRunner);

  for (unsigned int i = 0; i < window->renders.size(); i++)
  {
    if (simulators[i]->isBeginingOfEpisode())
      window->renders[i]->poses.clear();
    else
      window->renders[i]->poses.push_back(
            std::make_pair(simulators[i]->getEnvironment()->getTRStep().o_tp1->at(0),
                           simulators[i]->getEnvironment()->getTRStep().o_tp1->at(1)));
    window->renders[i]->update();
  }

  //std::cout << "Model::timerEvent" << std::endl;
}

void Model::setWindow(Window* window)
{
  this->window = window;
}
