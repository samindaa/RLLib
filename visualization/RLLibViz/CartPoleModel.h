/*
 * CartPoleModel.h
 *
 *  Created on: Dec 8, 2014
 *      Author: sam
 */

#ifndef CARTPOLEMODEL_H_
#define CARTPOLEMODEL_H_

#include "ModelBase.h"

// From the simulation
#include "CartPole.h"

namespace RLLibViz
{

class CartPoleModel: public ModelBase
{
  Q_OBJECT

  protected:
    // RLLib
    Random<double>* random;
    RLProblem<double>* problem;
    Hashing<double>* hashing;
    int order;
    Vector<double>* gridResolutions;
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
    CartPoleModel();
    virtual ~CartPoleModel();

  protected:
    void doLearning(Window* window);
    void doEvaluation(Window* window);
};

}  // namespace RLLibViz

#endif /* CARTPOLEMODEL_H_ */
