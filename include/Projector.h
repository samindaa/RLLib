/*
 * Copyright 2014 Saminda Abeyruwan (saminda@cs.miami.edu)
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
 * Projector.h
 *
 *  Created on: Aug 18, 2012
 *      Author: sam
 */

#ifndef PROJECTOR_H_
#define PROJECTOR_H_

#include "Tiles.h"
#include "Vector.h"
#include "Action.h"

namespace RLLib
{

/**
 * Feature extractor for function approximation.
 * @class T feature type
 */
template<class T>
class Projector
{
  public:
    virtual ~Projector()
    {
    }
    virtual const Vector<T>* project(const Vector<T>* x, const int& h1) =0;
    virtual const Vector<T>* project(const Vector<T>* x) =0;
    virtual T vectorNorm() const =0;
    virtual int dimension() const =0;
};

/**
 * @use
 * http://incompleteideas.net/rlai.cs.ualberta.ca/RLAI/RLtoolkit/tiles.html
 */

template<class T>
class TileCoder: public Projector<T>
{
  protected:
    bool includeActiveFeature;
    Vector<T>* vector;
    int nbTilings;

  public:
    TileCoder(const int& memorySize, const int& nbTilings, bool includeActiveFeature = true) :
        includeActiveFeature(includeActiveFeature), vector(
            new SVector<T>(includeActiveFeature ? memorySize + 1 : memorySize)), nbTilings(
            nbTilings)
    {
    }

    virtual ~TileCoder()
    {
      delete vector;
    }

    virtual void coder(const Vector<T>* x) =0;
    virtual void coder(const Vector<T>* x, const int& h1) =0;

    const Vector<T>* project(const Vector<T>* x, const int& h1)
    {
      vector->clear();
      if (x->empty())
        return vector;
      if (includeActiveFeature)
      {
        coder(x, h1);
        vector->setEntry(vector->dimension() - 1, 1.0);
      }
      else
        coder(x, h1);
      return vector;
    }

    const Vector<T>* project(const Vector<T>* x)
    {
      vector->clear();
      if (x->empty())
        return vector;
      if (includeActiveFeature)
      {
        coder(x);
        vector->setEntry(vector->dimension() - 1, 1.0);
      }
      else
        coder(x);
      return vector;
    }

    T vectorNorm() const
    {
      return includeActiveFeature ? nbTilings + 1 : nbTilings;
    }

    int dimension() const
    {
      return vector->dimension();
    }

};

template<class T>
class TileCoderHashing: public TileCoder<T>
{
  private:
    typedef TileCoder<T> Base;
    Vector<T>* gridResolutions;
    Vector<T>* inputs;
    Tiles<T>* tiles;

  public:
    TileCoderHashing(Hashing<T>* hashing, const int& nbInputs, const T& gridResolution,
        const int& nbTilings, const bool& includeActiveFeature = true) :
        TileCoder<T>(hashing->getMemorySize(), nbTilings, includeActiveFeature), gridResolutions(
            new PVector<T>(nbInputs)), inputs(new PVector<T>(nbInputs)), tiles(
            new Tiles<T>(hashing))
    {
      gridResolutions->set(gridResolution);
    }

    TileCoderHashing(Hashing<T>* hashing, const int& nbInputs, Vector<T>* gridResolutions,
        const int& nbTilings, const bool& includeActiveFeature = true) :
        TileCoder<T>(hashing->getMemorySize(), nbTilings, includeActiveFeature), gridResolutions(
            new PVector<T>(nbInputs)), inputs(new PVector<T>(nbInputs)), tiles(
            new Tiles<T>(hashing))
    {
      this->gridResolutions->set(gridResolutions);
    }

    virtual ~TileCoderHashing()
    {
      delete inputs;
      delete tiles;
    }

    void coder(const Vector<T>* x)
    {
      inputs->clear();
      inputs->addToSelf(x)->ebeMultiplyToSelf(gridResolutions);
      tiles->tiles(Base::vector, Base::nbTilings, inputs);
    }

    void coder(const Vector<T>* x, const int& h1)
    {
      inputs->clear();
      inputs->addToSelf(x)->ebeMultiplyToSelf(gridResolutions);
      tiles->tiles(Base::vector, Base::nbTilings, inputs, h1);
    }
};

template<class T>
class FourierBasis: public Projector<T>
{
  protected:
    Vector<T>* featureVector;
    std::vector<Vector<T>*> coefficientVectors;

  public:
    FourierBasis(const int& nbInputs, const int& order, const Actions<T>* actions)
    {
      computeFourierCoefficients(nbInputs, order);
      featureVector = new PVector<T>(coefficientVectors.size() * actions->dimension());
    }

    virtual ~FourierBasis()
    {
      delete featureVector;
      for (typename std::vector<Vector<T>*>::iterator iter = coefficientVectors.begin();
          iter != coefficientVectors.end(); ++iter)
        delete *iter;
    }

    /**
     * x must be unit normalized [0, 1)
     */
    const Vector<T>* project(const Vector<T>* x, const int& h1)
    {
      featureVector->clear();
      if (x->empty())
        return featureVector;
      // FixMe: SIMD
      const int stripWidth = coefficientVectors.size() * h1;
      for (size_t i = 0; i < coefficientVectors.size(); i++)
        featureVector->setEntry(i + stripWidth, std::cos(M_PI * x->dot(coefficientVectors[i])));
      return featureVector;

    }

    const Vector<T>* project(const Vector<T>* x)
    {
      return project(x, 0);
    }

    T vectorNorm() const
    {
      return T(1); //FixMe:
    }

    int dimension() const
    {
      return featureVector->dimension();
    }

    const std::vector<Vector<T>*>& getCoefficientVectors() const
    {
      return coefficientVectors;
    }

  private:
    inline void nextCoefficientVector(Vector<T>* coefficientVector, const int& nbInputs,
        const int& order)
    {
      coefficientVector->setEntry(nbInputs - 1, coefficientVector->getEntry(nbInputs - 1) + 1);
      if (coefficientVector->getEntry(nbInputs - 1) > order)
      {
        if (nbInputs > 1)
        {
          coefficientVector->setEntry(nbInputs - 1, 0);
          nextCoefficientVector(coefficientVector, nbInputs - 1, order);
        }
      }
    }

    inline void computeFourierCoefficients(const int& nbInputs, const int& order)
    {
      Vector<T>* coefficientVector = new PVector<T>(nbInputs);
      do
      {
        Vector<T>* newCoefficientVector = new PVector<T>(nbInputs);
        newCoefficientVector->set(coefficientVector);
        coefficientVectors.push_back(newCoefficientVector);
        nextCoefficientVector(coefficientVector, nbInputs, order);
      } while (coefficientVector->getEntry(0) <= order);
      ASSERT(coefficientVectors.size() == std::pow(order + 1, nbInputs));
      delete coefficientVector;
    }
};

} // namespace RLLib

#endif /* PROJECTOR_H_ */
