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
 * Projector.h
 *
 *  Created on: Aug 18, 2012
 *      Author: sam
 */

#ifndef PROJECTOR_H_
#define PROJECTOR_H_

#include "Tiles.h"
#include "Vector.h"

namespace RLLib
{

/**
 * Feature extractor for function approximation.
 * @class T feature type
 * @class O observation type
 */
template<class T>
class Projector
{
  public:
    virtual ~Projector()
    {
    }
    virtual const Vector<T>* project(const Vector<T>* x, int h1) =0;
    virtual const Vector<T>* project(const Vector<T>* x) =0;
    virtual double vectorNorm() const =0;
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
    int nbTiling;

  public:
    TileCoder(const int& memorySize, const int& nbTiling, bool includeActiveFeature = true) :
        includeActiveFeature(includeActiveFeature), vector(
            new SVector<T>(includeActiveFeature ? memorySize + 1 : memorySize)), nbTiling(nbTiling)
    {
    }

    virtual ~TileCoder()
    {
      delete vector;
    }

    virtual void coder(Vector<T>* vector, const Vector<T>* x, const int& nbTiling) =0;
    virtual void coder(Vector<T>* vector, const Vector<T>* x, const int& nbTiling,
        const int& h1) =0;

    const Vector<T>* project(const Vector<T>* x, int h1)
    {
      vector->clear();
      if (x->empty())
        return vector;
      if (includeActiveFeature)
      {
        coder(vector, x, nbTiling, h1);
        vector->setEntry(vector->dimension() - 1, 1.0);
      }
      else
        coder(vector, x, nbTiling, h1);
      return vector;
    }

    const Vector<T>* project(const Vector<T>* x)
    {
      vector->clear();
      if (x->empty())
        return vector;
      if (includeActiveFeature)
      {
        coder(vector, x, nbTiling);
        vector->setEntry(vector->dimension() - 1, 1.0);
      }
      else
        coder(vector, x, nbTiling);
      return vector;
    }

    double vectorNorm() const
    {
      return includeActiveFeature ? nbTiling + 1 : nbTiling;
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
    Tiles<T>* tiles;

  public:
    TileCoderHashing(Hashing* hashing, const int& nbTiling, const bool& includeActiveFeature = true) :
        TileCoder<T>(hashing->getMemorySize(), nbTiling, includeActiveFeature), tiles(
            new Tiles<T>(hashing))
    {
    }

    virtual ~TileCoderHashing()
    {
      delete tiles;
    }

    void coder(Vector<T>* vector, const Vector<T>* x, const int& nbTiling)
    {
      tiles->tiles(vector, nbTiling, x);
    }

    void coder(Vector<T>* vector, const Vector<T>* x, const int& nbTiling, const int& h1)
    {
      tiles->tiles(vector, nbTiling, x, h1);
    }
};

} // namespace RLLib

#endif /* PROJECTOR_H_ */
