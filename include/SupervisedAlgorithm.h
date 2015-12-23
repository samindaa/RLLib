/*
 * Copyright 2015 Saminda Abeyruwan (saminda@cs.miami.edu)
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

  template<typename T>
  class Adaline: public LearningAlgorithm<T>, public LinearLearner<T>
  {
    protected:
      Vector<T>* w;
      T alpha;
    public:
      Adaline(const int& size, const T& alpha) :
          w(new PVector<T>(size)), alpha(alpha)
      {
      }
      virtual ~Adaline()
      {
        delete w;
      }

      T initialize()
      {
        return T(0);
      }

      T predict(const Vector<T>* x) const
      {
        return w->dot(x);
      }

      void reset()
      {
        w->clear();
      }

      T learn(const Vector<T>* x_t, const T& y_tp1)
      {
        T delta = y_tp1 - predict(x_t);
        w->addToSelf(alpha * delta, x_t);
        return delta;
      }

      void persist(const char* f) const
      {
        w->persist(f);
      }

      void resurrect(const char* f)
      {
        w->resurrect(f);
      }

      Vector<T>* weights() const
      {
        return w;
      }
  };

  template<typename T>
  class IDBD: public LearningAlgorithm<T>, public LinearLearner<T>
  {
    protected:
      Vector<T>* w;
      Vector<T>* alphas;
      Vector<T>* hs;
      VectorPool<T>* pool;
      T theta, minimumStepSize;

    public:
      IDBD(const int& size, const T& theta) :
          w(new PVector<T>(size)), alphas(new PVector<T>(w->dimension())), //
          hs(new PVector<T>(w->dimension())), pool(new VectorPool<T>(size)), theta(theta), //
          minimumStepSize(1e-6)
      {
        alphas->set(1.0f / w->dimension());
      }

      virtual ~IDBD()
      {
        delete w;
        delete alphas;
        delete hs;
        delete pool;
      }

      T initialize()
      {
        return T(0);
      }

      T predict(const Vector<T>* x) const
      {
        return w->dot(x);
      }

      void reset()
      {
        w->clear();
        alphas->set(1.0f / w->dimension());
        hs->clear();
      }

      T learn(const Vector<T>* x_t, const T& y_tp1)
      {
        T delta = y_tp1 - predict(x_t);
        Vector<T>* deltaX = pool->newVector(x_t)->mapMultiplyToSelf(delta);
        Vector<T>* deltaXH = pool->newVector(deltaX)->ebeMultiplyToSelf(hs);
        Vectors<T>::multiplySelfByExponential(alphas, theta, deltaXH, minimumStepSize);
        Vector<T>* alphasDeltaX = deltaX->ebeMultiplyToSelf(alphas);
        w->addToSelf(alphasDeltaX);
        Vector<T>* alphasX2 = pool->newVector(x_t)->ebeMultiplyToSelf(x_t)->ebeMultiplyToSelf(
            alphas)->ebeMultiplyToSelf(hs);
        hs->addToSelf(-1.0f, alphasX2);
        hs->addToSelf(alphasDeltaX);
        pool->releaseAll();
        return delta;
      }

      void persist(const char* f) const
      {
        w->persist(f);
      }

      void resurrect(const char* f)
      {
        w->resurrect(f);
      }

      Vector<T>* weights() const
      {
        return w;
      }
  };

  template<typename T>
  class SemiLinearIDBD: public LearningAlgorithm<T>, public LinearLearner<T>
  {
    protected:
      Vector<T>* w;
      Vector<T>* alphas;
      Vector<T>* hs;
      VectorPool<T>* pool;
      T theta, minimumStepSize;

    public:
      SemiLinearIDBD(const int& size, const T& theta) :
          w(new PVector<T>(size)), alphas(new PVector<T>(w->dimension())), //
          hs(new PVector<T>(w->dimension())), pool(new VectorPool<T>(size)), theta(theta), //
          minimumStepSize(1e-6)
      {
        alphas->set(1.0f / w->dimension());
      }

      virtual ~SemiLinearIDBD()
      {
        delete w;
        delete alphas;
        delete hs;
        delete pool;
      }

      T initialize()
      {
        return T(0);
      }

      T predict(const Vector<T>* x) const
      {
        return 1.0f / (1.0f + exp(-w->dot(x)));
      }

      void reset()
      {
        w->clear();
        alphas->set(1.0f / w->dimension());
        hs->clear();
      }

      T learn(const Vector<T>* x_t, const T& z_tp1)
      {
        T y_tp1 = predict(x_t);
        T delta = z_tp1 - y_tp1;
        Vector<T>* deltaX = pool->newVector(x_t)->mapMultiplyToSelf(delta);
        Vector<T>* deltaXH = pool->newVector(deltaX)->ebeMultiplyToSelf(hs);
        Vectors<T>::multiplySelfByExponential(alphas, theta, deltaXH, minimumStepSize);
        Vector<T>* alphasDeltaX = deltaX->ebeMultiplyToSelf(alphas);
        w->addToSelf(alphasDeltaX);
        Vector<T>* alphasX2YMinusOneMinusY =
            pool->newVector(x_t)->ebeMultiplyToSelf(x_t)->ebeMultiplyToSelf(alphas)->ebeMultiplyToSelf(
                hs)->mapMultiplyToSelf(y_tp1)->mapMultiplyToSelf(1.0f - y_tp1);
        hs->addToSelf(-1.0f, alphasX2YMinusOneMinusY);
        hs->addToSelf(alphasDeltaX);
        pool->releaseAll();
        return delta;
      }

      void persist(const char* f) const
      {
        w->persist(f);
      }

      void resurrect(const char* f)
      {
        w->resurrect(f);
      }

      Vector<T>* weights() const
      {
        return w;
      }
  };

  template<typename T>
  class K1: public LearningAlgorithm<T>, public LinearLearner<T>
  {
    protected:
      Vector<T>* w;
      Vector<T>* alphas;
      Vector<T>* betas;
      Vector<T>* hs;
      VectorPool<T>* pool;
      T theta;

    public:
      K1(const int& size, const T& theta) :
          w(new PVector<T>(size)), alphas(new PVector<T>(size)), betas(new PVector<T>(size)), //
          hs(new PVector<T>(size)), pool(new VectorPool<T>(size)), theta(theta)
      {
        betas->set(std::log(0.1f));
      }

      virtual ~K1()
      {
        delete w;
        delete alphas;
        delete betas;
        delete hs;
        delete pool;
      }

      T initialize()
      {
        return T(0);
      }

      T predict(const Vector<T>* x) const
      {
        return w->dot(x);
      }

      void reset()
      {
        w->clear();
        alphas->clear();
        betas->clear();
        hs->clear();
        betas->set(std::log(0.1f));
      }

    private:
      void updateHS(const Vector<T>* x, const Vector<T>* x2, const Vector<T>* pi,
          const Vector<T>* piX, const T& delta)
      {
        Vector<T>* piX2 = pool->newVector(x2)->ebeMultiplyToSelf(pi);
        const SparseVector<T>* sresult = RTTI<T>::constSparseVector(piX2);
        if (sresult)
        {
          const int* activeIndexes = sresult->nonZeroIndexes();
          for (int i = 0; i < sresult->nonZeroElements(); i++)
          {
            int index = activeIndexes[i];
            piX2->setEntry(index, std::max(0.0, 1.0 - piX2->getEntry(index)));
          }
        }
        else
        {
          for (int index = 0; index < piX2->dimension(); index++)
            piX2->setEntry(index, std::max(0.0, 1.0 - piX2->getEntry(index)));
        }
        Vector<T>* piDeltaXPiX2 = pool->newVector(piX)->mapMultiplyToSelf(delta)->ebeMultiplyToSelf(
            piX2);
        hs->ebeMultiplyToSelf(piX2)->addToSelf(piDeltaXPiX2);
      }

    public:
      T learn(const Vector<T>* x_t, const T& y_tp1)
      {
        T delta = y_tp1 - predict(x_t);
        Vector<T>* xHS = pool->newVector(x_t)->ebeMultiplyToSelf(hs);
        betas->addToSelf(theta * delta, xHS);
        Vectors<T>::expToSelf(alphas, betas);
        Vector<T>* x2 = pool->newVector(x_t)->ebeMultiplyToSelf(x_t);
        Vector<T>* alphasX2 = pool->newVector(x2)->ebeMultiplyToSelf(alphas);
        T pnorm = alphasX2->sum();
        Vector<T>* pi = pool->newVector(alphas)->mapMultiplyToSelf(1.0f / (1.0f + pnorm));
        Vector<T>* piX = pool->newVector(x_t)->ebeMultiplyToSelf(pi);
        w->addToSelf(delta, piX);
        updateHS(x_t, x2, pi, piX, delta);
        pool->releaseAll();
        return delta;
      }

      void persist(const char* f) const
      {
        w->persist(f);
      }

      void resurrect(const char* f)
      {
        w->resurrect(f);
      }

      Vector<T>* weights() const
      {
        return w;
      }
  };

  template<typename T>
  class Autostep: public LearningAlgorithm<T>, public LinearLearner<T>
  {
    protected:
      Vector<T>* w;
      Vector<T>* alphas;
      Vector<T>* h;
      Vector<T>* v;
      VectorPool<T>* pool;
      T tau, minimumStepsize, kappa, delta;

    public:
      Autostep(const int& nbFeatures, const T& kappa = 0.01f, const T& initStepsize = 1.0) :
          w(new PVector<T>(nbFeatures)), alphas(new PVector<T>(w->dimension())), //
          h(new PVector<T>(w->dimension())), v(new PVector<T>(w->dimension())), //
          pool(new VectorPool<T>(nbFeatures)), tau(10000), minimumStepsize(1e-6), kappa(kappa), //
          delta(0.0f)
      {
        alphas->set(initStepsize);
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
        Vector<T>* vUpdate =
            pool->newVector(absDeltaXH)->subtractToSelf(sparseV)->ebeMultiplyToSelf(x2)->ebeMultiplyToSelf(
                alphas);
        v->addToSelf(1.0f / tau, vUpdate);
        Vectors<T>::positiveMaxToSelf(v, absDeltaXH);
        Vectors<T>::multiplySelfByExponential(RTTI<T>::denseVector(alphas), kappa,
            deltaXH->ebeDivideToSelf(v), minimumStepsize);
        Vector<T>* x2ByAlphas = pool->newVector(x2)->ebeMultiplyToSelf(alphas);
        T sum = x2ByAlphas->sum();
        if (sum > 1.0f)
          Filters<T>::mapMultiplyToSelf(alphas, 1.0f / sum, x);
      }

    public:
      T initialize()
      {
        return T(0);
      }

      T predict(const Vector<T>* x) const
      {
        return w->dot(x);
      }
      void reset()
      {
        w->clear();
      }

      T learn(const Vector<T>* x_t, const T& y_tp1)
      {
        delta = y_tp1 - predict(x_t);
        Vector<T>* deltaX = pool->newVector(x_t)->mapMultiplyToSelf(delta);
        Vector<T>* x2 = pool->newVector(x_t)->ebeMultiplyToSelf(x_t);
        updateAlphas(x_t, x2, deltaX);
        Vector<T>* alphasDeltaX = deltaX->ebeMultiplyToSelf(alphas);
        w->addToSelf(alphasDeltaX);
        Vector<T>* x2AlphasH = x2->ebeMultiplyToSelf(alphas)->ebeMultiplyToSelf(h);
        h->addToSelf(-1.0f, x2AlphasH)->addToSelf(alphasDeltaX);
        pool->releaseAll();
        return delta;
      }

      void persist(const char* f) const
      {
        w->persist(f);
      }

      void resurrect(const char* f)
      {
        w->resurrect(f);
      }

      Vector<T>* weights() const
      {
        return w;
      }
  };

} // namespace RLLib

#endif /* SUPERVISEDALGORITHM_H_ */
