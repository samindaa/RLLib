/*
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
    int nbTiling;
    bool includeActiveFeature;
    SparseVector<T>* vector;
    int* theTiles;
  public:
    TileCoder(const int& memorySize, const int& numTiling,
        bool includeActiveFeature = true) :
        nbTiling(numTiling), includeActiveFeature(includeActiveFeature), vector(
            new SparseVector<T>(
                includeActiveFeature ? memorySize + 1 : memorySize)), theTiles(
            new int[numTiling])
    {
      // Consistent hashing
      int dummyTiles[1];
      float dummyVars[1];
      srand(0);
      tiles(dummyTiles, 1, 1, dummyVars, 0); // initializes tiling code
      srand(time(0));
    }

    virtual ~TileCoder()
    {

      delete vector;
      delete[] theTiles;
    }

    virtual void coder(int* theTiles, const int& nt, const int& memory,
        float* floats, const int& nf) =0;
    virtual void coder(int* theTiles, const int& nt, const int& memory,
        float* floats, const int& nf, const int& h1) =0;

    const SparseVector<T>& project(const DenseVector<O>& x, int h1)
    {
      vector->clear();
      if (includeActiveFeature)
      {
        coder(theTiles, nbTiling, vector->dimension() - 1, x(), x.dimension(),
            h1);
        vector->insertLast(1.0);
      }
      else
        coder(theTiles, nbTiling, vector->dimension(), x(), x.dimension(), h1);

      for (int* i = theTiles; i < theTiles + nbTiling; ++i)
        vector->insertEntry(*i, 1.0);
      return *vector;
    }

    const SparseVector<T>& project(const DenseVector<O>& x)
    {
      vector->clear();
      if (includeActiveFeature)
      {
        coder(theTiles, nbTiling, vector->dimension() - 1, x(), x.dimension());
        vector->insertLast(1.0);
      }
      else
        coder(theTiles, nbTiling, vector->dimension(), x(), x.dimension());

      for (int* i = theTiles; i < theTiles + nbTiling; ++i)
        vector->insertEntry(*i, 1.0);
      return *vector;

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

template<class T, class O>
class TileCoderHashing: public TileCoder<T, O>
{

  public:
    TileCoderHashing(const int& memorySize, const int& numTiling,
        bool includeActiveFeature = true) :
        TileCoder<T, O>(memorySize, numTiling, includeActiveFeature)
    {
    }

    virtual ~TileCoderHashing()
    {
    }

    void coder(int* theTiles, const int& nt, const int& memory, float* floats,
        const int& nf)
    {
      tiles(theTiles, nt, memory, floats, nf);
    }

    void coder(int* theTiles, const int& nt, const int& memory, float* floats,
        const int& nf, const int& h1)
    {
      tiles(theTiles, nt, memory, floats, nf, h1);
    }

};

template<class T, class O>
class TileCoderNoHashing: public TileCoder<T, O>
{
  protected:
    typedef TileCoder<T, O> Base;
    collision_table* ct;
  public:
    TileCoderNoHashing(const int& memorySize, const int& numTiling,
        bool includeActiveFeature = true) :
        TileCoder<T, O>(memorySize, numTiling, includeActiveFeature)
    {
      // http://graphics.stanford.edu/~seander/bithacks.html
      // Compute the next highest power of 2 of 32-bit v
      unsigned int v = Base::vector->dimension();
      v--;
      v |= v >> 1;
      v |= v >> 2;
      v |= v >> 4;
      v |= v >> 8;
      v |= v >> 16;
      v++;

      // The vector needs to reflect the correct memory size
      delete Base::vector;
      Base::vector = new SparseVector<T>(includeActiveFeature ? v + 1 : v);
      ct = new collision_table(v, 1);
    }

    virtual ~TileCoderNoHashing()
    {
      delete ct;
    }

    void coder(int* theTiles, const int& nt, const int& memory, float* floats,
        const int& nf)
    {
      tiles(theTiles, nt, ct, floats, nf);
    }

    void coder(int* theTiles, const int& nt, const int& memory, float* floats,
        const int& nf, const int& h1)
    {
      tiles(theTiles, nt, ct, floats, nf, h1);
    }

};

// @@>>TODO: Yet to implement
class IndependentTilings
{
};

} // namespace RLLib

#endif /* PROJECTOR_H_ */
