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
template<class T, class O>
class Projector
{
  public:
    virtual ~Projector()
    {
    }
    virtual const SparseVector<T>& project(const DenseVector<O>& x, int h1) =0;
    virtual const SparseVector<T>& project(const DenseVector<O>& x) =0;
    virtual double vectorNorm() const =0;
    virtual int dimension() const =0;
};

/**
 * @use
 * http://incompleteideas.net/rlai.cs.ualberta.ca/RLAI/RLtoolkit/tiles.html
 */

template<class T, class O>
class TileCoder: public Projector<T, O>
{
  protected:
    bool includeActiveFeature;
    SparseVector<T>* vector;
    DenseVector<int>* tileIndices;
  public:
    TileCoder(const int& memorySize, const int& numTiling, bool includeActiveFeature = true) :
        includeActiveFeature(includeActiveFeature),
            vector(new SparseVector<T>(includeActiveFeature ? memorySize + 1 : memorySize)),
            tileIndices(new DenseVector<int>(numTiling))
    {
    }

    virtual ~TileCoder()
    {

      delete vector;
      delete tileIndices;
    }

    virtual void coder(DenseVector<int>& theTiles, const DenseVector<O>& x, const int& memory) =0;
    virtual void coder(DenseVector<int>& theTiles, const DenseVector<O>& x, const int& memory,
        const int& h1) =0;

    const SparseVector<T>& project(const DenseVector<O>& x, int h1)
    {
      vector->clear();
      if (includeActiveFeature)
      {
        coder(*tileIndices, x, vector->dimension() - 1, h1);
        vector->insertLast(1.0);
      }
      else
        coder(*tileIndices, x, vector->dimension(), h1);

      for (int i = 0; i < tileIndices->dimension(); i++)
        vector->insertEntry(tileIndices->at(i), 1.0);
      return *vector;
    }

    const SparseVector<T>& project(const DenseVector<O>& x)
    {
      vector->clear();
      if (includeActiveFeature)
      {
        coder(*tileIndices, x, vector->dimension() - 1);
        vector->insertLast(1.0);
      }
      else
        coder(*tileIndices, x, vector->dimension());

      for (int i = 0; i < tileIndices->dimension(); i++)
        vector->insertEntry(tileIndices->at(i), 1.0);
      return *vector;

    }

    double vectorNorm() const
    {
      return includeActiveFeature ? tileIndices->dimension() + 1 : tileIndices->dimension();
    }

    int dimension() const
    {
      return vector->dimension();
    }

};

template<class T, class O>
class TileCoderHashing: public TileCoder<T, O>
{
  private:
    Tiles* tiles;
  public:
    TileCoderHashing(const int& memorySize, const int& numTiling, bool includeActiveFeature = true,
        Hashing* hashing = 0) :
        TileCoder<T, O>(memorySize, numTiling, includeActiveFeature), tiles(new Tiles(hashing))
    {
    }

    virtual ~TileCoderHashing()
    {
      delete tiles;
    }

    void coder(DenseVector<int>& theTiles, const DenseVector<O>& x, const int& memory)
    {
      tiles->tiles(theTiles(), theTiles.dimension(), memory, x(), x.dimension());
    }

    void coder(DenseVector<int>& theTiles, const DenseVector<O>& x, const int& memory,
        const int& h1)
    {
      tiles->tiles(theTiles(), theTiles.dimension(), memory, x(), x.dimension(), h1);
    }

};

template<class T, class O>
class TileCoderNoHashing: public TileCoder<T, O>
{
  protected:
    Tiles* tiles;
    CollisionTable* ct;
  public:
    TileCoderNoHashing(const int& memorySize, const int& numTiling,
        bool includeActiveFeature = true) :
        TileCoder<T, O>(memorySize, numTiling, includeActiveFeature), tiles(new Tiles)
    {
      // http://graphics.stanford.edu/~seander/bithacks.html
      // Compute the next highest power of 2 of 32-bit v
      unsigned int v = TileCoder<T, O>::vector->dimension();
      v--;
      v |= v >> 1;
      v |= v >> 2;
      v |= v >> 4;
      v |= v >> 8;
      v |= v >> 16;
      v++;

      // The vector needs to reflect the correct memory size
      delete TileCoder<T, O>::vector;
      TileCoder<T, O>::vector = new SparseVector<T>(includeActiveFeature ? v + 1 : v);
      ct = new CollisionTable(v, 1);
    }

    virtual ~TileCoderNoHashing()
    {
      delete tiles;
      delete ct;
    }

    void coder(DenseVector<int>& theTiles, const DenseVector<O>& x, const int& memory)
    {
      tiles->tiles(theTiles(), theTiles.dimension(), ct, x(), x.dimension());
    }

    void coder(DenseVector<int>& theTiles, const DenseVector<O>& x, const int& memory,
        const int& h1)
    {
      tiles->tiles(theTiles(), theTiles.dimension(), ct, x(), x.dimension(), h1);
    }
};

// @@>>TODO: Yet to implement
class IndependentTilings
{
};

} // namespace RLLib

#endif /* PROJECTOR_H_ */
