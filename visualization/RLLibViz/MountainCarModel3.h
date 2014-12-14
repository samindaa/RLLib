/*
 * MountainCarModel3.h
 *
 *  Created on: Dec 5, 2014
 *      Author: sam
 */

#ifndef MOUNTAINCARMODEL3_H_
#define MOUNTAINCARMODEL3_H_


#include "ModelBase.h"

// From the simulation
#include "MountainCar.h"

namespace RLLibViz
{

class MountainCarModel3: public ModelBase
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
    double gamma;
    double lambda;
    Sarsa<double>* sarsaAdaptive;
    double epsilon;
    Policy<double>* acting;
    OnPolicyControlLearner<double>* control;
    RLAgent<double>* agent;
    RLRunner<double>* simulator;


  public:
    MountainCarModel3();
    virtual ~MountainCarModel3();

  protected:
    void doLearning(Window* window);
    void doEvaluation(Window* window);
};

}  // namespace RLLibViz



#endif /* MOUNTAINCARMODEL3_H_ */
