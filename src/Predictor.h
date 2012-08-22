/*
 * Predictor.h
 *
 *  Created on: Aug 19, 2012
 *      Author: sam
 */

#ifndef PREDICTOR_H_
#define PREDICTOR_H_

#include "Vector.h"

template<class T>
class Predictor
{
  public:
    virtual ~Predictor()
    {
    }
    virtual int dimension() const=0;
    virtual double predict(const SparseVector<T>& x) const =0;
    virtual void reset() =0;
};

template<class T>
class OffPolicyTD: public Predictor<T>
{
  public:
    virtual ~OffPolicyTD()
    {
    }
    //@@>>TODO: which method to use
};

#endif /* PREDICTOR_H_ */
