/*
 * SupervisedAlgorithm.h
 *
 *  Created on: Sep 7, 2012
 *      Author: sam
 */

#ifndef SUPERVISEDALGORITHM_H_
#define SUPERVISEDALGORITHM_H_

#include "Supervised.h"

namespace RLLib
{

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

    void persist(const std::string& f) const
    {
      w->persist(f);
    }
    void resurrect(const std::string& f)
    {
      w->resurrect(f);
    }
};

template<class T>
class Autostep: public LearningAlgorithm<T>
{
  protected:
    SparseVector<T>* w;
    SparseVector<T>* alpha;
    SparseVector<T>* h;
    SparseVector<T>* v;
    SparseVector<T>* x2;
    SparseVector<T>* deltaX;
    SparseVector<T>* deltaXh;
    SparseVector<T>* absDeltaXh;
    SparseVector<T>* sparseV;
    double tau, delta;

  public:
    Autostep(const int& nbFeatures) :
        w(new SparseVector<T>(nbFeatures)), alpha(
            new SparseVector<T>(w->dimension())), h(
            new SparseVector<T>(w->dimension())), v(
            new SparseVector<T>(w->dimension())), x2(
            new SparseVector<T>(w->dimension())), deltaX(
            new SparseVector<T>(w->dimension())), deltaXh(
            new SparseVector<T>(w->dimension())), absDeltaXh(
            new SparseVector<T>(w->dimension())), sparseV(
            new SparseVector<T>(w->dimension())), tau(1000.0), delta(0)
    {
      alpha->set(1.0 / nbFeatures);
      v->set(1.0);
    }

    virtual ~Autostep()
    {
      delete w;
      delete alpha;
      delete h;
      delete v;
      delete x2;
      delete deltaX;
      delete deltaXh;
      delete absDeltaXh;
      delete sparseV;
    }
  private:
    void updateAlpha(const SparseVector<T>& x, const SparseVector<T>& x2,
        const SparseVector<T>& deltaX)
    {
      deltaXh->set(deltaX).ebeMultiplyToSelf(h);
      absDeltaXh->set(deltaX).absToSelf();
      // TODO:
    }

  public:
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
      delta = y - predict(x);
      deltaX->set(x).multiplyToSelf(delta);
      x2->set(x).ebeMultiplyToSelf(x);
      updateAlpha(x, x2, *deltaX);
      SparseVector<T>& alphaDeltaX = deltaX->ebeMultiplyToSelf(alpha);
      w->addToSelf(alphaDeltaX);
      SparseVector<T>& minusX2AlphaH =
          x2->ebeMultiplyToSelf(alpha).ebeMultiplyToSelf(h).multiplyToSelf(-1);
      h->addToSelf(minusX2AlphaH).addToSelf(alphaDeltaX);
    }

    void persist(const std::string& f) const
    {
      w->persist(f);
    }
    void resurrect(const std::string& f)
    {
      w->resurrect(f);
    }
};

} // namespace RLLib

#endif /* SUPERVISEDALGORITHM_H_ */
