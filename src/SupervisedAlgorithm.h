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
class Adaline: public LearningAlgorithm<T>, public LinearLearner<T>
{
  protected:
    SparseVector<T>* w;
    double alpha;
  public:
    Adaline(const int& size, const double& alpha) :
        w(new SVector<T>(size)), alpha(alpha)
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

    double predict(const Vector<T>* x) const
    {
      return w->dot(x);
    }

    void reset()
    {
      w->clear();
    }

    void learn(const Vector<T>* x, const T& y)
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

    const Vector<T>* weights() const
    {
      return w;
    }
};

template<class T>
class IDBD: public LearningAlgorithm<T>, public LinearLearner<T>
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
        w(new SVector<T>(size)), alpha(new SVector<T>(w->dimension())), h(
            new SVector<T>(w->dimension())), theta(theta), minimumStepSize(10e-7), deltaX(
            new SVector<T>(w->dimension())), deltaXh(new SVector<T>(w->dimension())), alphaX2(
            new SVector<T>(w->dimension()))
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

    double predict(const Vector<T>* x) const
    {
      return w->dot(x);
    }

    void reset()
    {
      w->clear();
      alpha->set(1.0 / w->dimension());
      h->clear();
    }

    void learn(const Vector<T>* x, const T& y)
    {
      double delta = y - predict(x);
      deltaX->set(x);
      deltaX->mapMultiplyToSelf(delta);
      deltaXh->set(deltaX);
      deltaXh->ebeMultiplyToSelf(h);
      Vectors<T>::multiplySelfByExponential(alpha, theta, deltaXh, minimumStepSize);
      Vector<T>* alphaDeltaX = deltaX->ebeMultiplyToSelf(alpha);
      w->addToSelf(alphaDeltaX);
      alphaX2->set(x);
      alphaX2->ebeMultiplyToSelf(x)->ebeMultiplyToSelf(alpha)->ebeMultiplyToSelf(h);
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

    const Vector<T>* weights() const
    {
      return w;
    }
};

template<class T>
class SemiLinearIDBD: public LearningAlgorithm<T>, public LinearLearner<T>
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
    SemiLinearIDBD(const int& size, const double& theta) :
        w(new SVector<T>(size)), alpha(new SVector<T>(w->dimension())), h(
            new SVector<T>(w->dimension())), theta(theta), minimumStepSize(10e-7), deltaX(
            new SVector<T>(w->dimension())), deltaXh(new SVector<T>(w->dimension())), alphaX2YMinusOneMinusY(
            new SVector<T>(w->dimension()))
    {
      alpha->set(1.0 / w->dimension());
    }
    virtual ~SemiLinearIDBD()
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

    double predict(const Vector<T>* x) const
    {
      return 1.0 / (1.0 + exp(-w->dot(x)));
    }

    void reset()
    {
      w->clear();
      alpha->set(1.0 / w->dimension());
      h->clear();
    }

    void learn(const Vector<T>* x, const T& z)
    {
      double y = predict(x);
      double delta = z - y;
      deltaX->set(x);
      deltaX->mapMultiplyToSelf(delta);
      deltaXh->set(deltaX);
      deltaXh->ebeMultiplyToSelf(h);
      Vectors<T>::multiplySelfByExponential(alpha, theta, deltaXh, minimumStepSize);
      Vector<T>* alphaDeltaX = deltaX->ebeMultiplyToSelf(alpha);
      w->addToSelf(alphaDeltaX);
      alphaX2YMinusOneMinusY->set(x);
      alphaX2YMinusOneMinusY->ebeMultiplyToSelf(x)->ebeMultiplyToSelf(alpha)->ebeMultiplyToSelf(h)->mapMultiplyToSelf(
          y)->mapMultiplyToSelf(1.0 - y);
      h->addToSelf(-1.0, alphaX2YMinusOneMinusY);
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

    const Vector<T>* weights() const
    {
      return w;
    }
};

template<class T>
class Autostep: public LearningAlgorithm<T>, public LinearLearner<T>
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
        w(new SVector<T>(nbFeatures)), alpha(new SVector<T>(w->dimension())), h(
            new SVector<T>(w->dimension())), v(new SVector<T>(w->dimension())), tau(1000.0), minimumStepSize(
            10e-7), deltaXh(new SVector<T>(w->dimension())), absDeltaXh(
            new SVector<T>(w->dimension())), vUpdate(new SVector<T>(w->dimension())), x2ByAlpha(
            new SVector<T>(w->dimension())), deltaX(new SVector<T>(w->dimension())), x2(
            new SVector<T>(w->dimension()))
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
    void updateAlpha(const SparseVector<T>* x, const SparseVector<T>* x2,
        const SparseVector<T>* deltaX)
    {
      deltaXh->set(deltaX);
      deltaXh->ebeMultiplyToSelf(h);
      absDeltaXh->set(deltaXh);
      Vectors<T>::absToSelf(absDeltaXh);
      vUpdate->set(absDeltaXh);
      vUpdate->subtractToSelf(v)->ebeMultiplyToSelf(x2)->ebeMultiplyToSelf(alpha);
      v->addToSelf(1.0 / tau, vUpdate);
      Vectors<T>::positiveMaxToSelf(v, absDeltaXh);
      Vectors<T>::multiplySelfByExponential(alpha, 0.01,
          (const SparseVector<T>*) deltaXh->ebeDivideToSelf(v), minimumStepSize);
      x2ByAlpha->set(x2);
      x2ByAlpha->ebeMultiplyToSelf(alpha);
      double sum = std::max(x2ByAlpha->sum(), 1.0);
      if (sum > 1.0)
        alpha->mapMultiplyToSelf(1.0 / sum);
    }

  public:
    double initialize()
    {
      return 0.0;
    }

    double predict(const Vector<T>* x) const
    {
      return w->dot(x);
    }
    void reset()
    {
      w->clear();
    }

    void learn(const Vector<T>* x, const T& y)
    {
      double delta = y - predict(x);
      deltaX->set(x);
      deltaX->mapMultiplyToSelf(delta);
      x2->set(x);
      x2->ebeMultiplyToSelf(x);
      updateAlpha((const SparseVector<T>*) x, x2, deltaX); // fixMe
      Vector<T>* alphaDeltaX = deltaX->ebeMultiplyToSelf(alpha);
      w->addToSelf(alphaDeltaX);
      Vector<T>* minusX2AlphaH = x2->ebeMultiplyToSelf(alpha)->ebeMultiplyToSelf(h);
      h->addToSelf(minusX2AlphaH->mapMultiplyToSelf(-1.0))->addToSelf(alphaDeltaX);
    }

    void persist(const std::string& f) const
    {
      w->persist(f);
    }

    void resurrect(const std::string& f)
    {
      w->resurrect(f);
    }

    const Vector<T>* weights() const
    {
      return w;
    }
};

} // namespace RLLib

#endif /* SUPERVISEDALGORITHM_H_ */
