/*
 * AcrobotTest.h
 *
 *  Created on: Dec 10, 2013
 *      Author: sam
 */

#ifndef ACROBOTTEST_H_
#define ACROBOTTEST_H_

#include "Test.h"
#include "Acrobot.h"

RLLIB_TEST(AcrobotTest)
class AcrobotTest: public AcrobotTestBase
{
  public:
    void run();
  protected:
    void testAcrobotOnPolicy();
};

template<class T>
class AcrobotProjector: public Projector<T>
{
  protected:
    int nbTilings;
    int memory;
    Vector<T>* vector;
    Random<T>* random;
    Hashing<T>* hashing;
    Tiles<T>* tiles;
  public:
    AcrobotProjector(const int& nbVars, const int& nbActions, const int& toHash)
    {
      /**
       * Number of inputs | Tiling type | Number of intervals | Number of tilings
       *            4     |     1D      |       4             |       4
       *                  |     2D      |       4             |       4
       *                  |     2D + 1  |       4             |       4
       *                  |     2D + 2  |       4             |       4
       */
      nbTilings = nbVars * (4 + 4 + 4 + 4);
      memory = nbVars * (4 * 4 + 4 * 4 * 4 + 4 * 4 * 4 + 4 * 4 * 4) * nbActions * toHash;
      vector = new SVector<T>(memory + 1/*bias unit*/, nbTilings + 1/*bias unit*/);
      random = new Random<T>;
      hashing = new MurmurHashing<T>(random, memory);
      tiles = new Tiles<T>(hashing);
    }

    virtual ~AcrobotProjector()
    {
      delete vector;
      delete random;
      delete hashing;
      delete tiles;
    }

    const Vector<T>* project(const Vector<T>* x, const int& h1)
    {
      vector->clear();
      if (x->empty())
        return vector;

      int h2 = 0;
      for (int i = 0; i < x->dimension(); i++)
      {
        tiles->tiles1(vector, 4, memory, x->getEntry(i) * 4, h1, h2++);

        int j = (i + 1) % x->dimension();
        tiles->tiles2(vector, 4, memory, x->getEntry(i) * 4, x->getEntry(j) * 4, h1, h2++);

        j = (i + 2) % x->dimension();
        tiles->tiles2(vector, 4, memory, x->getEntry(i) * 4, x->getEntry(j) * 4, h1, h2++);

        j = (i + 3) % x->dimension();
        tiles->tiles2(vector, 4, memory, x->getEntry(i) * 4, x->getEntry(j) * 4, h1, h2++);
      }

      vector->setEntry(vector->dimension() - 1, 1.0f);
      return vector;
    }

    const Vector<T>* project(const Vector<T>* x)
    {
      vector->clear();
      if (x->empty())
        return vector;

      int h2 = 0;
      for (int i = 0; i < x->dimension(); i++)
      {
        tiles->tiles1(vector, 4, memory, x->getEntry(i) * 4, h2++);

        int j = (i + 1) % x->dimension();
        tiles->tiles2(vector, 4, memory, x->getEntry(i) * 4, x->getEntry(j) * 4, h2++);

        j = (i + 2) % x->dimension();
        tiles->tiles2(vector, 4, memory, x->getEntry(i) * 4, x->getEntry(j) * 4, h2++);

        j = (i + 3) % x->dimension();
        tiles->tiles2(vector, 4, memory, x->getEntry(i) * 4, x->getEntry(j) * 4, h2++);
      }

      vector->setEntry(vector->dimension() - 1, 1.0f);
      return vector;
    }

    double vectorNorm() const
    {
      return nbTilings + 1;
    }

    int dimension() const
    {
      return vector->dimension();
    }
};

#endif /* ACROBOTTEST_H_ */
