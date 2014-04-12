/*
 * SwingPendulumModel.cpp
 *
 *  Created on: Oct 14, 2013
 *      Author: sam
 */

#include "SwingPendulumModel.h"

using namespace RLLibViz;

SwingPendulumModel::SwingPendulumModel(QObject *parent) :
    ModelBase(parent)
{
  random = new Random<double>;
  problem = new SwingPendulum<double>;
  hashing = new MurmurHashing<double>(random, 1000);
  projector = new TileCoderHashing<double>(hashing, problem->dimension(), 10, 10, false);
  toStateAction = new StateActionTilings<double>(projector, problem->getContinuousActions());

  alpha_v = 0.1 / projector->vectorNorm();
  alpha_u = 0.001 / projector->vectorNorm();
  alpha_r = .0001;
  gamma = 1.0;
  lambda = 0.5;

  critice = new ATrace<double>(projector->dimension());
  critic = new TDLambda<double>(alpha_v, gamma, lambda, critice);

  policyDistribution = new NormalDistributionScaled<double>(random, problem->getContinuousActions(),
      0, 1.0, projector->dimension());
  policyRange = new Range<double>(-2.0, 2.0);
  problemRange = new Range<double>(-2.0, 2.0);
  acting = new ScaledPolicyDistribution<double>(problem->getContinuousActions(), policyDistribution,
      policyRange, problemRange);

  actore1 = new ATrace<double>(projector->dimension());
  actore2 = new ATrace<double>(projector->dimension());
  actoreTraces = new Traces<double>();
  actoreTraces->push_back(actore1);
  actoreTraces->push_back(actore2);
  actor = new ActorLambda<double>(alpha_u, gamma, lambda, acting, actoreTraces);

  control = new AverageRewardActorCritic<double>(critic, actor, projector, toStateAction, alpha_r);
  agent = new LearnerAgent<double>(control);
  simulator = new Simulator<double>(agent, problem, 5000);
  simulator->setVerbose(false);
  valueFunction = new Matrix(100, 100);

}

SwingPendulumModel::~SwingPendulumModel()
{
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
  delete policyRange;
  delete problemRange;
  delete acting;
  delete control;
  delete agent;
  delete simulator;
  delete valueFunction;
}

void SwingPendulumModel::initialize()
{
  ModelBase::initialize();
}

void SwingPendulumModel::doWork()
{
  simulator->step();
  if (simulator->isEndingOfEpisode())
  {
    emit signal_add(window->plots[0], Vec(simulator->timeStep, 0), Vec(simulator->episodeR, 0));
    emit signal_draw(window->plots[0]);
  }

  emit signal_add(window->views[0],
      Vec(simulator->getRLProblem()->getObservations()->at(0),
          simulator->getRLProblem()->getObservations()->at(1)), Vec(0.0, 0.0, 0.0, 1.0));
  emit signal_draw(window->views[0]);

  // Value function
  if (simulator->isEndingOfEpisode() && window->vfuns.size() > 0)
  {
    RLLib::PVector<double> x_t(2);
    double maxValue = 0, minValue = 0;
    const Range<double>* thetaRange = problem->getObservationRanges()->at(0);
    const Range<double>* velocityRange = problem->getObservationRanges()->at(1);

    for (int theta = 0; theta < valueFunction->rows(); theta++)
    {
      for (int velocity = 0; velocity < valueFunction->cols(); velocity++)
      {
        x_t[0] = thetaRange->toUnit(
            thetaRange->length() * theta / valueFunction->cols() + thetaRange->min());
        x_t[1] = velocityRange->toUnit(
            velocityRange->length() * velocity / valueFunction->rows() + velocityRange->min());
        double v = control->computeValueFunction(&x_t);
        valueFunction->at(theta, velocity) = v;
        if (v > maxValue)
          maxValue = v;
        if (v < minValue)
          minValue = v;
      }
    }
    //out.close();
    emit signal_add(window->vfuns[0], valueFunction, minValue, maxValue);
    emit signal_draw(window->vfuns[0]);
  }
}
