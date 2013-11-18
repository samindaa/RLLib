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
    VectorPool<T>* pool;
    double theta, minimumStepSize;

  public:
    IDBD(const int& size, const double& theta) :
        w(new PVector<T>(size)), alphas(new PVector<T>(w->dimension())), hs(
            new PVector<T>(w->dimension())), pool(new VectorPool<T>(size)), theta(theta), minimumStepSize(
            1e-6)
    {
      alphas->set(1.0 / w->dimension());
    }

    virtual ~IDBD()
    {
      delete w;
      delete alphas;
      delete hs;
      delete pool;
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

    double learn(const Vector<T>* x_t, const T& y_tp1)
    {
      double delta = y_tp1 - predict(x_t);
      Vector<T>* deltaX = pool->newVector(x_t)->mapMultiplyToSelf(delta);
      Vector<T>* deltaXH = pool->newVector(deltaX)->ebeMultiplyToSelf(hs);
      Vectors<T>::multiplySelfByExponential(alphas, theta, deltaXH, minimumStepSize);
      Vector<T>* alphasDeltaX = deltaX->ebeMultiplyToSelf(alphas);
      w->addToSelf(alphasDeltaX);
      Vector<T>* alphasX2 =
          pool->newVector(x_t)->ebeMultiplyToSelf(x_t)->ebeMultiplyToSelf(alphas)->ebeMultiplyToSelf(
              hs);
      hs->addToSelf(-1.0f, alphasX2);
      hs->addToSelf(alphasDeltaX);
      pool->releaseAll();
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
    VectorPool<T>* pool;
    double theta, minimumStepSize;

  public:
    SemiLinearIDBD(const int& size, const double& theta) :
        w(new PVector<T>(size)), alphas(new PVector<T>(w->dimension())), hs(
            new PVector<T>(w->dimension())), pool(new VectorPool<T>(size)), theta(theta), minimumStepSize(
            1e-6)
    {
      alphas->set(1.0 / w->dimension());
    }

    virtual ~SemiLinearIDBD()
    {
      delete w;
      delete alphas;
      delete hs;
      delete pool;
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

    double learn(const Vector<T>* x_t, const T& z_tp1)
    {
      double y_tp1 = predict(x_t);
      double delta = z_tp1 - y_tp1;
      Vector<T>* deltaX = pool->newVector(x_t)->mapMultiplyToSelf(delta);
      Vector<T>* deltaXH = pool->newVector(deltaX)->ebeMultiplyToSelf(hs);
      Vectors<T>::multiplySelfByExponential(alphas, theta, deltaXH, minimumStepSize);
      Vector<T>* alphasDeltaX = deltaX->ebeMultiplyToSelf(alphas);
      w->addToSelf(alphasDeltaX);
      Vector<T>* alphasX2YMinusOneMinusY =
          pool->newVector(x_t)->ebeMultiplyToSelf(x_t)->ebeMultiplyToSelf(alphas)->ebeMultiplyToSelf(
              hs)->mapMultiplyToSelf(y_tp1)->mapMultiplyToSelf(1.0 - y_tp1);
      hs->addToSelf(-1.0f, alphasX2YMinusOneMinusY);
      hs->addToSelf(alphasDeltaX);
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
    VectorPool<T>* pool;
    double tau, minimumStepsize, kappa;

  public:
    Autostep(const int& nbFeatures) :
        w(new PVector<T>(nbFeatures)), alphas(new PVector<T>(w->dimension())), h(
            new PVector<T>(w->dimension())), v(new PVector<T>(w->dimension())), pool(
            new VectorPool<T>(nbFeatures)), tau(10000), minimumStepsize(1e-6), kappa(0.01f)
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
      delete pool;
    }

  private:
    void updateAlphas(const Vector<T>* x, const Vector<T>* x2, const Vector<T>* deltaX)
    {
      Vector<T>* deltaXH = pool->newVector(deltaX)->ebeMultiplyToSelf(h);
      Vector<T>* absDeltaXH = Vectors<T>::absToSelf(pool->newVector(deltaXH));
      Vector<T>* sparseV = pool->newVector(x);
      sparseV->clear();
      Vectors<T>::toBinary(sparseV, deltaX)->ebeMultiplyToSelf(v);
      Vector<T>* vUpdate = pool->newVector(absDeltaXH)->subtractToSelf(sparseV)->ebeMultiplyToSelf(
          x2)->ebeMultiplyToSelf(alphas);
      v->addToSelf(1.0f / tau, vUpdate);
      Vectors<T>::positiveMaxToSelf(v, absDeltaXH);
      Vectors<T>::multiplySelfByExponential(dynamic_cast<DenseVector<T>*>(alphas), kappa,
          deltaXH->ebeDivideToSelf(v), minimumStepsize);
      Vector<T>* x2ByAlphas = pool->newVector(x2)->ebeMultiplyToSelf(alphas);
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

    double learn(const Vector<T>* x_t, const T& y_tp1)
    {
      double delta = y_tp1 - predict(x_t);
      Vector<T>* deltaX = pool->newVector(x_t)->mapMultiplyToSelf(delta);
      Vector<T>* x2 = pool->newVector(x_t)->ebeMultiplyToSelf(x_t);
      updateAlphas(x_t, x2, deltaX);
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
