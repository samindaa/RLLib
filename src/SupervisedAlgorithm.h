/*
 * SupervisedAlgorithm.h
 *
 *  Created on: Sep 7, 2012
 *      Author: sam
 */

#ifndef SUPERVISEDALGORITHM_H_
#define SUPERVISEDALGORITHM_H_

#include "Supervised.h"

template<class T>
class Adaline: public LearningAlgorithm<T>
{
  protected:
    SparseVector<T>* w;
    double alpha;
  public:
    Adaline(const int& size, const double& alpha) :
        w(new SparseVector<T>(size)), alpha(alpha)
    {
    }
    virtual ~Adaline()
    {
      delete w;
    }

    int dimension() const
    {
      return w->dimension();
    }
    double predict(const SparseVector<T>& x) const
    {
      return w->dot(x);
    }
    void reset()
    {
      w->clear();
    }

    void learn(const SparseVector<T>& x, const T& y)
    {
      w->addToSelf(alpha * (y - predict(x)), x);
    }
};

template<class T>
class Autostep
{ // @@TODO
};

#endif /* SUPERVISEDALGORITHM_H_ */
