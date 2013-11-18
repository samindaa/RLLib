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
    Vector<T>* w;
    double alpha;
  public:
    Adaline(const int& size, const double& alpha) :
        w(new PVector<T>(size)), alpha(alpha)
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

    double learn(const Vector<T>* x, const T& y)
    {
      double delta = y - predict(x);
      w->addToSelf(alpha * delta, x);
      return delta;
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
    Vector<T>* w;
    Vector<T>* alphas;
    Vector<T>* hs;
    double theta, minimumStepSize;

    // Auxiliary variables
    Vector<T>* deltaX;
    Vector<T>* deltaXh;
    Vector<T>* alphaX2;

  public:
    IDBD(const int& size, const double& theta) :
        w(new PVector<T>(size)), alphas(new PVector<T>(w->dimension())), hs(
            new PVector<T>(w->dimension())), theta(theta), minimumStepSize(1e-6), deltaX(0), deltaXh(
            0), alphaX2(0)
    {
      alphas->set(1.0 / w->dimension());
    }
    virtual ~IDBD()
    {
      delete w;
      delete alphas;
      delete hs;

      if (deltaX)
        delete deltaX;
      if (deltaXh)
        delete deltaXh;
      if (alphaX2)
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
      alphas->set(1.0 / w->dimension());
      hs->clear();
    }

    double learn(const Vector<T>* x, const T& y)
    {
      double delta = y - predict(x);
      Vectors<T>::bufferedCopy(x, deltaX)->mapMultiplyToSelf(delta);
      Vectors<T>::bufferedCopy(deltaX, deltaXh)->ebeMultiplyToSelf(hs);
      Vectors<T>::multiplySelfByExponential(alphas, theta, deltaXh, minimumStepSize);
      Vector<T>* alphaDeltaX = deltaX->ebeMultiplyToSelf(alphas);
      w->addToSelf(alphaDeltaX);
      Vectors<T>::bufferedCopy(x, alphaX2)->ebeMultiplyToSelf(x)->ebeMultiplyToSelf(alphas)->ebeMultiplyToSelf(
          hs);
      hs->addToSelf(-1.0f, alphaX2);
      hs->addToSelf(alphaDeltaX);
      return delta;
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
    Vector<T>* w;
    Vector<T>* alphas;
    Vector<T>* hs;
    double theta, minimumStepSize;

    // Auxiliary variables
    Vector<T>* deltaX;
    Vector<T>* deltaXh;
    Vector<T>* alphaX2YMinusOneMinusY;

  public:
    SemiLinearIDBD(const int& size, const double& theta) :
        w(new PVector<T>(size)), alphas(new PVector<T>(w->dimension())), hs(
            new PVector<T>(w->dimension())), theta(theta), minimumStepSize(1e-6), deltaX(0), deltaXh(
            0), alphaX2YMinusOneMinusY(0)
    {
      alphas->set(1.0 / w->dimension());
    }
    virtual ~SemiLinearIDBD()
    {
      delete w;
      delete alphas;
      delete hs;

      if (deltaX)
        delete deltaX;
      if (deltaXh)
        delete deltaXh;
      if (alphaX2YMinusOneMinusY)
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
      alphas->set(1.0 / w->dimension());
      hs->clear();
    }

    double learn(const Vector<T>* x, const T& z)
    {
      double y = predict(x);
      double delta = z - y;
      Vectors<T>::bufferedCopy(x, deltaX)->mapMultiplyToSelf(delta);
      Vectors<T>::bufferedCopy(deltaX, deltaXh)->ebeMultiplyToSelf(hs);
      Vectors<T>::multiplySelfByExponential(alphas, theta, deltaXh, minimumStepSize);
      Vector<T>* alphaDeltaX = deltaX->ebeMultiplyToSelf(alphas);
      w->addToSelf(alphaDeltaX);
      Vectors<T>::bufferedCopy(x, alphaX2YMinusOneMinusY)->ebeMultiplyToSelf(x)->ebeMultiplyToSelf(
          alphas)->ebeMultiplyToSelf(hs)->mapMultiplyToSelf(y)->mapMultiplyToSelf(1.0 - y);
      hs->addToSelf(-1.0f, alphaX2YMinusOneMinusY);
      hs->addToSelf(alphaDeltaX);
      return delta;
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
    Vector<T>* w;
    Vector<T>* alphas;
    Vector<T>* h;
    Vector<T>* v;
    double tau, minimumStepsize, kappa;

    // Auxiliary variables
    Vector<T>* deltaXh;
    Vector<T>* absDeltaXh;
    Vector<T>* sparseV;
    Vector<T>* vUpdate;
    Vector<T>* x2ByAlphas;

    Vector<T>* deltaX;
    Vector<T>* x2;

  public:
    Autostep(const int& nbFeatures) :
        w(new PVector<T>(nbFeatures)), alphas(new PVector<T>(w->dimension())), h(
            new PVector<T>(w->dimension())), v(new PVector<T>(w->dimension())), tau(10000), minimumStepsize(
            1e-6), kappa(0.01f), deltaXh(0), absDeltaXh(0), sparseV(0), vUpdate(0), x2ByAlphas(0), deltaX(
            0), x2(0)
    {
      alphas->set(1.0);
      v->set(1.0);
    }

    virtual ~Autostep()
    {
      delete w;
      delete alphas;
      delete h;
      delete v;

      if (deltaXh)
        delete deltaXh;
      if (absDeltaXh)
        delete absDeltaXh;
      if (sparseV)
        delete sparseV;
      if (vUpdate)
        delete vUpdate;
      if (x2ByAlphas)
        delete x2ByAlphas;
      if (deltaX)
        delete deltaX;
      if (x2)
        delete x2;
    }
  private:
    void updateAlphas(const Vector<T>* x, const Vector<T>* x2, const Vector<T>* deltaX)
    {
      Vectors<T>::bufferedCopy(deltaX, deltaXh)->ebeMultiplyToSelf(h);
      Vectors<T>::absToSelf(Vectors<T>::bufferedCopy(deltaXh, absDeltaXh));
      if (!sparseV)
        sparseV = deltaX->copy();
      Vectors<T>::toBinary(sparseV, deltaX)->ebeMultiplyToSelf(v);
      Vectors<T>::bufferedCopy(absDeltaXh, vUpdate)->subtractToSelf(sparseV)->ebeMultiplyToSelf(x2)->ebeMultiplyToSelf(
          alphas);
      v->addToSelf(1.0f / tau, vUpdate);
      Vectors<T>::positiveMaxToSelf(v, absDeltaXh);
      Vectors<T>::multiplySelfByExponential(dynamic_cast<DenseVector<T>*>(alphas), kappa,
          deltaXh->ebeDivideToSelf(v), minimumStepsize);
      Vectors<T>::bufferedCopy(x2, x2ByAlphas)->ebeMultiplyToSelf(alphas);
      double sum = x2ByAlphas->sum();
      if (sum > 1.0f)
        Filters<T>::mapMultiplyToSelf(alphas, 1.0f / sum, x);
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

    double learn(const Vector<T>* x, const T& y)
    {
      double delta = y - predict(x);
      Vectors<T>::bufferedCopy(x, deltaX)->mapMultiplyToSelf(delta);
      Vectors<T>::bufferedCopy(x, x2)->ebeMultiplyToSelf(x);
      updateAlphas(x, x2, deltaX);
      Vector<T>* alphasDeltaX = deltaX->ebeMultiplyToSelf(alphas);
      w->addToSelf(alphasDeltaX);
      Vector<T>* x2AlphasH = x2->ebeMultiplyToSelf(alphas)->ebeMultiplyToSelf(h);
      h->addToSelf(-1.0, x2AlphasH)->addToSelf(alphasDeltaX);
      return delta;
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
