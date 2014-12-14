/*
 * AcrobotModel.h
 *
 *  Created on: Dec 6, 2014
 *      Author: sam
 */

#ifndef ACROBOTMODEL_H_
#define ACROBOTMODEL_H_

#include "ModelBase.h"

// From the simulation
#include "Acrobot.h"

namespace RLLibViz
{

class AcrobotModel: public ModelBase
{
  Q_OBJECT

  protected:
    // RLLib
    Random<double>* random;
    RLProblem<double>* problem;
    int order;
    Projector<double>* projector;
    StateToStateAction<double>* toStateAction;
    Trace<double>* e;
    double alpha;
    double gamma;
    double lambda;
    Sarsa<double>* sarsaAdaptive;
    double epsilon;
    Policy<double>* acting;
    OnPolicyControlLearner<double>* control;
    RLAgent<double>* agent;
    RLRunner<double>* simulator;

  public:
    AcrobotModel();
    virtual ~AcrobotModel();

  protected:
    void doLearning(Window* window);
    void doEvaluation(Window* window);
};

}  // namespace RLLibViz

#endif /* ACROBOTMODEL_H_ */
