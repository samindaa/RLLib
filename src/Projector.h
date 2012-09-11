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

namespace RLLib {

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
class FullTilings: public Projector<T, O>
{
  protected:
    int numTiling;
    bool includeActiveFeature;
    SparseVector<double>* vector;
    int* activeTiles;
  public:
    FullTilings(const int& memorySize, const int& numTiling,
        bool includeActiveFeature = true) :
        numTiling(numTiling), includeActiveFeature(includeActiveFeature),
            vector(new SparseVector<double>(includeActiveFeature ?
                memorySize + 1 : memorySize)), activeTiles(new int[numTiling])
    {
      // Consistent hashing
      int dummy_tiles[1];
      float dummy_vars[1];
      srand(0);
      tiles(dummy_tiles, 1, 1, dummy_vars, 0); // initializes tiling code
      srand(time(0));
    }

    virtual ~FullTilings()
    {

      delete vector;
      delete[] activeTiles;
    }

    const SparseVector<T>& project(const DenseVector<O>& x, int h1)
    {
      vector->clear();
      if (includeActiveFeature)
      {
        tiles(activeTiles, numTiling, vector->dimension() - 1, x(),
            x.dimension(), h1);
        vector->insertLast(1.0);
      }
      else tiles(activeTiles, numTiling, vector->dimension(), x(),
          x.dimension(), h1);

      for (int* i = activeTiles; i < activeTiles + numTiling; ++i)
        vector->insertEntry(*i, 1.0);
      return *vector;
    }

    const SparseVector<T>& project(const DenseVector<O>& x)
    {
      vector->clear();
      if (includeActiveFeature)
      {
        tiles(activeTiles, numTiling, vector->dimension() - 1, x(),
            x.dimension());
        vector->insertLast(1.0);
      }
      else tiles(activeTiles, numTiling, vector->dimension(), x(),
          x.dimension());

      for (int* i = activeTiles; i < activeTiles + numTiling; ++i)
        vector->insertEntry(*i, 1.0);
      return *vector;

    }

    double vectorNorm() const
    {
      return includeActiveFeature ?
          numTiling + 1 : numTiling;
    }

    int dimension() const
    {
      return vector->dimension();
    }
};

// @@>>TODO: Yet to implement
class IndependentTilings
{
};

} // namespace RLLib

#endif /* PROJECTOR_H_ */
