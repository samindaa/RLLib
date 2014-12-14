/*
 * SwingPendulumModel4.h
 *
 *  Created on: Dec 5, 2014
 *      Author: sam
 */

#ifndef SWINGPENDULUMMODEL4_H_
#define SWINGPENDULUMMODEL4_H_

#include "ModelBase.h"

// From the simulation
#include "SwingPendulum.h"

namespace RLLibViz
{

class SwingPendulumModel4: public ModelBase
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
    SwingPendulumModel4();
    virtual ~SwingPendulumModel4();

  protected:
    void doLearning(Window* window);
    void doEvaluation(Window* window);
};

}  // namespace RLLibViz



#endif /* SWINGPENDULUMMODEL4_H_ */
