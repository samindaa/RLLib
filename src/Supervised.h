/*
 * Supervised.h
 *
 *  Created on: Sep 7, 2012
 *      Author: sam
 */

#ifndef SUPERVISED_H_
#define SUPERVISED_H_

#include "Predictor.h"

namespace RLLib
{

template<class T>
class LearningAlgorithm: public Predictor<T>
{
  public:
    virtual ~LearningAlgorithm()
    {
    }
    virtual void learn(const SparseVector<T>& x, const T& y) = 0;
};

} // namespace RLLib

#endif /* SUPERVISED_H_ */
