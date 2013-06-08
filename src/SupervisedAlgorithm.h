/*
 * Copyright 2013 Saminda Abeyruwan (saminda@cs.miami.edu)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
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

    // Auxiliary variables
    SparseVector<T>* deltaX;
    SparseVector<T>* deltaXh;
    SparseVector<T>* alphaX2;

  public:
    IDBD(const int& size, const double& theta) :
        w(new SparseVector<T>(size)), alpha(new SparseVector<T>(w->dimension())), h(
            new SparseVector<T>(w->dimension())), theta(theta), minimumStepSize(10e-7), deltaX(
            new SparseVector<T>(w->dimension())), deltaXh(new SparseVector<T>(w->dimension())), alphaX2(
            new SparseVector<T>(w->dimension()))
    {
      alpha->set(1.0 / w->dimension());
    }
    virtual ~IDBD()
    {
      delete w;
      delete alpha;
      delete h;

      delete deltaX;
      delete deltaXh;
      delete alphaX2;
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
      deltaX->set(x);
      deltaX->multiplyToSelf(delta);
      deltaXh->set(*deltaX);
      deltaXh->ebeMultiplyToSelf(*h);
      SparseVector<T>::multiplySelfByExponential(*alpha, theta, *deltaXh, minimumStepSize);
      SparseVector<T>& alphaDeltaX = deltaX->ebeMultiplyToSelf(*alpha);
      w->addToSelf(alphaDeltaX);
      alphaX2->set(x);
      alphaX2->ebeMultiplyToSelf(x).ebeMultiplyToSelf(*alpha).ebeMultiplyToSelf(*h);
      h->addToSelf(-1.0, *alphaX2);
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
class SemiLinerIDBD: public LearningAlgorithm<T>
{
  protected:
    SparseVector<T>* w;
    SparseVector<T>* alpha;
    SparseVector<T>* h;
    double theta, minimumStepSize;

    // Auxiliary variables
    SparseVector<T>* deltaX;
    SparseVector<T>* deltaXh;
    SparseVector<T>* alphaX2YMinusOneMinusY;

  public:
    SemiLinerIDBD(const int& size, const double& theta) :
        w(new SparseVector<T>(size)), alpha(new SparseVector<T>(w->dimension())), h(
            new SparseVector<T>(w->dimension())), theta(theta), minimumStepSize(10e-7), deltaX(
            new SparseVector<T>(w->dimension())), deltaXh(new SparseVector<T>(w->dimension())), alphaX2YMinusOneMinusY(
            new SparseVector<T>(w->dimension()))
    {
      alpha->set(1.0 / w->dimension());
    }
    virtual ~SemiLinerIDBD()
    {
      delete w;
      delete alpha;
      delete h;

      delete deltaX;
      delete deltaXh;
      delete alphaX2YMinusOneMinusY;
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
      return 1.0 / (1.0 + exp(-w->dot(x)));
    }
    void reset()
    {
      w->clear();
      alpha->set(1.0 / w->dimension());
      h->clear();
    }

    void learn(const SparseVector<T>& x, const T& z)
    {
      double y = predict(x);
      double delta = z - y;
      deltaX->set(x);
      deltaX->multiplyToSelf(delta);
      deltaXh->set(*deltaX);
      deltaXh->ebeMultiplyToSelf(*h);
      SparseVector<T>::multiplySelfByExponential(*alpha, theta, *deltaXh, minimumStepSize);
      SparseVector<T>& alphaDeltaX = deltaX->ebeMultiplyToSelf(*alpha);
      w->addToSelf(alphaDeltaX);
      alphaX2YMinusOneMinusY->set(x);
      alphaX2YMinusOneMinusY->ebeMultiplyToSelf(x).ebeMultiplyToSelf(*alpha).ebeMultiplyToSelf(*h).multiplyToSelf(
          y).multiplyToSelf(1.0 - y);
      h->addToSelf(-1.0, *alphaX2YMinusOneMinusY);
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

    // Auxiliary variables
    SparseVector<T>* deltaXh;
    SparseVector<T>* absDeltaXh;
    SparseVector<T>* vUpdate;
    SparseVector<T>* x2ByAlpha;

    SparseVector<T>* deltaX;
    SparseVector<T>* x2;

  public:
    Autostep(const int& nbFeatures) :
        w(new SparseVector<T>(nbFeatures)), alpha(new SparseVector<T>(w->dimension())), h(
            new SparseVector<T>(w->dimension())), v(new SparseVector<T>(w->dimension())), tau(
            1000.0), minimumStepSize(10e-7), deltaXh(new SparseVector<T>(w->dimension())), absDeltaXh(
            new SparseVector<T>(w->dimension())), vUpdate(new SparseVector<T>(w->dimension())), x2ByAlpha(
            new SparseVector<T>(w->dimension())), deltaX(new SparseVector<T>(w->dimension())), x2(
            new SparseVector<T>(w->dimension()))
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

      delete deltaXh;
      delete absDeltaXh;
      delete vUpdate;
      delete x2ByAlpha;

      delete deltaX;
      delete x2;
    }
  private:
    void updateAlpha(const SparseVector<T>& x, const SparseVector<T>& x2,
        const SparseVector<T>& deltaX)
    {
      deltaXh->set(deltaX);
      deltaXh->ebeMultiplyToSelf(*h);
      absDeltaXh->set(*deltaXh);
      SparseVector<T>::absToSelf(*absDeltaXh);
      vUpdate->set(*absDeltaXh);
      vUpdate->substractToSelf(*v).ebeMultiplyToSelf(x2).ebeMultiplyToSelf(*alpha);
      v->addToSelf(1.0 / tau, *vUpdate);
      SparseVector<T>::positiveMaxToSelf(*v, *absDeltaXh);
      SparseVector<T>::multiplySelfByExponential(*alpha, 0.01, deltaXh->ebeDivideToSelf(*v),
          minimumStepSize);
      x2ByAlpha->set(x2);
      x2ByAlpha->ebeMultiplyToSelf(*alpha);
      double sum = std::max(x2ByAlpha->sum(), 1.0);
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
      deltaX->set(x);
      deltaX->multiplyToSelf(delta);
      x2->set(x);
      x2->ebeMultiplyToSelf(x);
      updateAlpha(x, *x2, *deltaX);
      SparseVector<T>& alphaDeltaX = deltaX->ebeMultiplyToSelf(*alpha);
      w->addToSelf(alphaDeltaX);
      SparseVector<T>& minusX2AlphaH = x2->ebeMultiplyToSelf(*alpha).ebeMultiplyToSelf(*h);
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
