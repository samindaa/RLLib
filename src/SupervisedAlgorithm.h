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

    double initialize()
    {
      return 0.0;
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
class IDBD: public LearningAlgorithm<T>
{
  protected:
    SparseVector<T>* w;
    SparseVector<T>* alpha;
    SparseVector<T>* h;
    double theta, minimumStepSize;
  public:
    IDBD(const int& size, const double& theta) :
        w(new SparseVector<T>(size)), alpha(new SparseVector<T>(w->dimension())), h(
            new SparseVector<T>(w->dimension())), theta(theta), minimumStepSize(10e-7)
    {
      alpha->set(1.0 / w->dimension());
    }
    virtual ~IDBD()
    {
      delete w;
      delete alpha;
      delete h;
    }

    double initialize()
    {
      return 0.0;
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
      alpha->set(1.0 / w->dimension());
      h->clear();
    }

    void learn(const SparseVector<T>& x, const T& y)
    {
      double delta = y - predict(x);
      SparseVector<T> deltaX(x);
      deltaX.multiplyToSelf(delta);
      SparseVector<T> deltaXh(deltaX);
      deltaXh.ebeMultiplyToSelf(*h);
      SparseVector<T>::multiplySelfByExponential(*alpha, theta, deltaXh, minimumStepSize);
      SparseVector<T>& alphaDeltaX = deltaX.ebeMultiplyToSelf(*alpha);
      w->addToSelf(alphaDeltaX);
      SparseVector<T> alphaX2(x);
      alphaX2.ebeMultiplyToSelf(x).ebeMultiplyToSelf(*alpha).ebeMultiplyToSelf(*h);
      h->addToSelf(-1.0, alphaX2);
      h->addToSelf(alphaDeltaX);
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
    double tau, minimumStepSize;

  public:
    Autostep(const int& nbFeatures) :
        w(new SparseVector<T>(nbFeatures)), alpha(new SparseVector<T>(w->dimension())), h(
            new SparseVector<T>(w->dimension())), v(new SparseVector<T>(w->dimension())), tau(
            1000.0), minimumStepSize(10e-7)
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
    }
  private:
    void updateAlpha(const SparseVector<T>& x, const SparseVector<T>& x2,
        const SparseVector<T>& deltaX)
    {
      SparseVector<T> deltaXh(deltaX);
      deltaXh.ebeMultiplyToSelf(*h);
      SparseVector<T> absDeltaXh(deltaXh);
      SparseVector<T>::absToSelf(absDeltaXh);
      SparseVector<T> vUpdate(absDeltaXh);
      vUpdate.substractToSelf(*v).ebeMultiplyToSelf(x2).ebeMultiplyToSelf(*alpha);
      v->addToSelf(1.0 / tau, vUpdate);
      SparseVector<T>::positiveMaxToSelf(*v, absDeltaXh);
      SparseVector<T>::multiplySelfByExponential(*alpha, 0.01, deltaXh.ebeDivideToSelf(*v),
          minimumStepSize);
      SparseVector<T> x2ByAlpha(x2);
      x2ByAlpha.ebeMultiplyToSelf(*alpha);
      double sum = std::max(x2ByAlpha.sum(), 1.0);
      if (sum > 1.0)
        alpha->multiplyToSelf(1.0 / sum);
    }

  public:
    double initialize()
    {
      return 0.0;
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
      double delta = y - predict(x);
      SparseVector<T> deltaX(x);
      deltaX.multiplyToSelf(delta);
      SparseVector<T> x2(x);
      x2.ebeMultiplyToSelf(x);
      updateAlpha(x, x2, deltaX);
      SparseVector<T>& alphaDeltaX = deltaX.ebeMultiplyToSelf(*alpha);
      w->addToSelf(alphaDeltaX);
      SparseVector<T>& minusX2AlphaH = x2.ebeMultiplyToSelf(*alpha).ebeMultiplyToSelf(*h);
      h->addToSelf(minusX2AlphaH.multiplyToSelf(-1.0)).addToSelf(alphaDeltaX);
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
