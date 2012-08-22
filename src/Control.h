/*
 * Control.h
 *
 *  Created on: Aug 19, 2012
 *      Author: sam
 */

#ifndef CONTROL_H_
#define CONTROL_H_

#include "Action.h"
#include "Vector.h"

template<class T, class O>
class Control
{
  public:
    virtual ~Control()
    {
    }
    virtual void reset() =0;
    virtual const Action& initialize(const DenseVector<O>& x_0) =0;
    virtual const Action& proposeAction(const DenseVector<O>& x) =0;
    virtual const Action& step(const DenseVector<O>& x_t, const Action& a_t,
        const DenseVector<O>& x_tp1, const double& r_tp1,
        const double& z_tp1) =0;
};

template<class T, class O>
class OnPolicyControlLearner: public Control<T, O>
{
  public:
    virtual ~OnPolicyControlLearner()
    {
    }
};

template<class T, class O>
class OffPolicyControlLearner: public Control<T, O>
{
  public:
    virtual ~OffPolicyControlLearner()
    {
    }
};

#endif /* CONTROL_H_ */
