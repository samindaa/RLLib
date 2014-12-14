/*
 * MountainCarModel2.h
 *
 *  Created on: Dec 4, 2014
 *      Author: sam
 */

#ifndef MOUNTAINCARMODEL2_H_
#define MOUNTAINCARMODEL2_H_

#include "ModelBase.h"

// From the simulation
#include "MountainCar.h"

namespace RLLibViz
{

class MountainCarModel2: public ModelBase
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
    double gamma;
    double lambda;
    Sarsa<double>* sarsaAdaptive;
    double epsilon;
    Policy<double>* acting;
    OnPolicyControlLearner<double>* control;
    RLAgent<double>* agent;
    RLRunner<double>* simulator;


  public:
    MountainCarModel2();
    virtual ~MountainCarModel2();

  protected:
    void doLearning(Window* window);
    void doEvaluation(Window* window);
};

}  // namespace RLLibViz

#endif /* MOUNTAINCARMODEL2_H_ */
