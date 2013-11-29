/*
 * SwingPendulumModel2.cpp
 *
 *  Created on: Nov 20, 2013
 *      Author: sam
 */

#include "SwingPendulumModel2.h"

using namespace RLLibViz;

SwingPendulumModel2::SwingPendulumModel2(QObject *parent) :
    ModelBase(parent)
{
  // RLLib:
  behaviourEnvironment = new SwingPendulum<double>(true);
  evaluationEnvironment = new SwingPendulum<double>;
  hashing = new MurmurHashing(1000000);
  projector = new TileCoderHashing<double>(hashing, 10, true);
  toStateAction = new StateActionTilings<double>(projector,
      behaviourEnvironment->getDiscreteActions());

  alpha_v = 0.1 / projector->vectorNorm();
  alpha_w = .0001 / projector->vectorNorm();
  gamma = 0.99;
  lambda = 0.0;
  alpha_u = 0.5 / projector->vectorNorm();

  critice = new ATrace<double>(projector->dimension());
  critic = new GTDLambda<double>(alpha_v, alpha_w, gamma, lambda, critice);

  target = new BoltzmannDistribution<double>(projector->dimension(),
      behaviourEnvironment->getDiscreteActions());

  actore = new ATrace<double>(projector->dimension());
  actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore);
  actor = new ActorLambdaOffPolicy<double>(alpha_u, gamma, lambda, target, actoreTraces);

  behavior = new RandomPolicy<double>(behaviourEnvironment->getDiscreteActions());
  control = new OffPAC<double>(behavior, critic, actor, toStateAction, projector);

  learningAgent = new LearnerAgent<double>(control);
  evaluationAgent = new ControlAgent<double>(control);

  learningRunner = new Simulator<double>(learningAgent, behaviourEnvironment, 1000);
  evaluationRunner = new Simulator<double>(evaluationAgent, evaluationEnvironment, 5000);
  learningRunner->setVerbose(false);
  evaluationRunner->setVerbose(false);

  simulators.insert(std::make_pair(simulators.size(), learningRunner));
  simulators.insert(std::make_pair(simulators.size(), evaluationRunner));

  valueFunction = new Matrix(101, 101); // << Fixed for 0:0.1:10

}

SwingPendulumModel2::~SwingPendulumModel2()
{
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
  delete valueFunction;
}

void SwingPendulumModel2::initialize()
{
  ModelBase::initialize();
}

void SwingPendulumModel2::doWork()
{
  for (std::tr1::unordered_map<int, Simulator<double>*>::iterator i = simulators.begin();
      i != simulators.end(); ++i)
    i->second->step();

  for (std::tr1::unordered_map<int, Simulator<double>*>::iterator i = simulators.begin();
      i != simulators.end(); ++i)
  {
    if (i->second->isEndingOfEpisode())
    {
      emit signal_draw(window->views[i->first]);
      emit signal_add(window->plots[i->first], Vec(i->second->timeStep, 0),
          Vec(i->second->episodeR, 0));
      emit signal_draw(window->plots[i->first]);
    }
    else
    {
      emit signal_add(window->views[i->first],
          Vec(i->second->getRLProblem()->getObservations()->at(0),
              i->second->getRLProblem()->getObservations()->at(1)), Vec(0.0, 0.0, 0.0, 1.0));
      emit signal_draw(window->views[i->first]);
    }
  }

  // Value function
  if (evaluationRunner->isEndingOfEpisode() && window->vfuns.size() > 1)
  {
    RLLib::PVector<double> x_t(2);
    double maxValue = 0, minValue = 0;
    float y = 0;
    for (int i = 0; i < valueFunction->rows(); i++)
    {
      float x = 0;
      for (int j = 0; j < valueFunction->cols(); j++)
      {
        x_t[0] = y;
        x_t[1] = x;
        double v = control->computeValueFunction(&x_t);
        valueFunction->at(i, j) = v;
        if (v > maxValue)
          maxValue = v;
        if (v < minValue)
          minValue = v;
        x += 0.1;
      }
      y += 0.1;
    }
    //out.close();
    emit signal_add(window->vfuns[1], valueFunction, minValue, maxValue);
    emit signal_draw(window->vfuns[1]);
  }
}

