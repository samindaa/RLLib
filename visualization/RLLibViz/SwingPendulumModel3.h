/*
 * SwingPendulumModel3.h
 *
 *  Created on: Dec 4, 2014
 *      Author: sam
 */

#ifndef SWINGPENDULUMMODEL3_H_
#define SWINGPENDULUMMODEL3_H_

#include "ModelBase.h"

// From the simulation
#include "SwingPendulum.h"

namespace RLLibViz
{

class SwingPendulumModel3: public ModelBase
{
  Q_OBJECT

  protected:
    // RLLib
    Random<double>* random;
    RLProblem<double>* problem;
    Hashing<double>* hashing;
    Projector<double>* projector;
    StateToStateAction<double>* toStateAction;
    Trace<double>* e;
    double alpha;
    double gamma;
    double lambda;
    Sarsa<double>* sarsaTrue;
    double epsilon;
    Policy<double>* acting;
    OnPolicyControlLearner<double>* control;
    RLAgent<double>* agent;
    RLRunner<double>* simulator;

  public:
    SwingPendulumModel3();
    virtual ~SwingPendulumModel3();

  protected:
    void doLearning(Window* window);
    void doEvaluation(Window* window);
};

}  // namespace RLLibViz

#endif /* SWINGPENDULUMMODEL3_H_ */
