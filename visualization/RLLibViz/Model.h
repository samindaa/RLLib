#ifndef MODEL_H
#define MODEL_H

#include <QObject>
#include <vector>
#include <QTimerEvent>

#include "Window.h"

// From the RLLib
#include "Vector.h"
#include "Trace.h"
#include "Projector.h"
#include "ControlAlgorithm.h"
#include "Representation.h"

// From the simulation
#include "ContinuousGridworld.h"
#include "Simulator.h"

using namespace RLLib;

class Model : public QObject
{
  Q_OBJECT

private:
  // Window
  Window* window;

  // RLLib
  Environment<float>* behaviourEnvironment;
  Environment<float>* evaluationEnvironment;
  Projector<double, float>* projector;
  StateToStateAction<double, float>* toStateAction;

  double alpha_v;
  double alpha_w;
  double gamma;
  double lambda;

  Trace<double>* critice;
  GTDLambda<double>* critic;

  double alpha_u;

  PolicyDistribution<double>* target;

  Trace<double>* actore;
  Traces<double>* actoreTraces;
  ActorOffPolicy<double, float>* actor;

  Policy<double>* behavior;
  OffPolicyControlLearner<double, float>* control;

  Simulator<double, float>* learningRunner;
  Simulator<double, float>* evaluationRunner;

public:
  explicit Model(QObject *parent = 0);
  virtual ~Model();

  void timerEvent(QTimerEvent *);
  void setWindow(Window* window);

};

#endif // MODEL_H
