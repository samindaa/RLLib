/*
 * Predictor.h
 *
 *  Created on: Aug 19, 2012
 *      Author: sam
 */

#ifndef PREDICTOR_H_
#define PREDICTOR_H_

#include "Vector.h"

namespace RLLib
{

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

    virtual void persist(const std::string& f) const =0;
    virtual void resurrect(const std::string& f) =0;
};

} // namespace RLLib

#endif /* PREDICTOR_H_ */
