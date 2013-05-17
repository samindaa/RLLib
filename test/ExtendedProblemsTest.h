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
 * ExtendedProblemsTest.h
 *
 *  Created on: May 15, 2013
 *      Author: sam
 */

#ifndef EXTENDEDPROBLEMSTEST_H_
#define EXTENDEDPROBLEMSTEST_H_

#include "HeaderTest.h"

// From the RLLib
#include "Vector.h"
#include "Trace.h"
#include "Projector.h"
#include "ControlAlgorithm.h"
#include "Representation.h"

// From the simulation
#include "MCar3D.h"
#include "Acrobot.h"
#include "RandlovBike.h"
#include "PoleBalancing.h"
#include "TorquedPendulum.h"
#include "Simulator.h"

#include "util/Spline.h"

RLLIB_TEST(ExtendedProblemsTest)

class ExtendedProblemsTest: public ExtendedProblemsTestBase
{
  public:
    ExtendedProblemsTest() {}
    virtual ~ExtendedProblemsTest() {}
    void run();

  private:
    void testOffPACMountainCar3D_1();
    void testGreedyGQMountainCar3D();
    void testSarsaMountainCar3D();
    void testOffPACMountainCar3D_2();
    void testOffPACAcrobot();
    void testGreedyGQAcrobot();

    void testPoleBalancingPlant();
    void testPersistResurrect();
    void testExp();
    void testEigen3();
    void testTorquedPendulum();

    void testCubicSpline();
    void testMotion();
};

// Helpers
// ====================== Advanced projector ===================================
template<class T, class O>
class AdvancedTilesProjector: public Projector<T, O>
{
  protected:
    SparseVector<double>* vector;
    int* activeTiles;
    Tiles* tiles;

  public:
    AdvancedTilesProjector() :
        vector(new SparseVector<T>(100000 + 1)), activeTiles(new int[48]), tiles(new Tiles)
    {
      // Consistent hashing
      int dummy_tiles[1];
      float dummy_vars[1];
      srand(0);
      tiles->tiles(dummy_tiles, 1, 1, dummy_vars, 0); // initializes tiling code
      srand(time(0));
    }

    virtual ~AdvancedTilesProjector()
    {
      delete vector;
      delete[] activeTiles;
      delete tiles;
    }

  public:
    const SparseVector<T>& project(const DenseVector<O>& x, int h1)
    {
      vector->clear();
      // all 4
      tiles->tiles(&activeTiles[0], 12, vector->dimension() - 1, x(), x.dimension(), h1);
      // 3 of 4
      static DenseVector<O> x3(3);
      static int x3o[4][3] =
      {
      { 0, 1, 2 },
      { 1, 2, 3 },
      { 2, 3, 0 },
      { 1, 3, 0 } };
      for (int i = 0; i < 4; i++)
      {
        for (int j = 0; j < 3; j++)
          x3[j] = x[x3o[i][j]];
        tiles->tiles(&activeTiles[12 + i * 3], 3, vector->dimension() - 1, x3(), x3.dimension(),
            h1);
      }
      // 2 of 6
      static DenseVector<O> x2(2);
      static int x2o[6][2] =
      {
      { 0, 1 },
      { 1, 2 },
      { 2, 3 },
      { 0, 3 },
      { 0, 2 },
      { 1, 3 } };
      for (int i = 0; i < 6; i++)
      {
        for (int j = 0; j < 2; j++)
          x2[j] = x[x2o[i][j]];
        tiles->tiles(&activeTiles[24 + i * 2], 2, vector->dimension() - 1, x2(), x2.dimension(),
            h1);
      }

      // 3 of 4 of 1
      static DenseVector<O> x1(1);
      static int x1o[4] =
      { 0, 1, 2, 3 };
      for (int i = 0; i < 4; i++)
      {
        x1[0] = x[x1o[i]];
        tiles->tiles(&activeTiles[36 + i * 3], 3, vector->dimension() - 1, x1(), x1.dimension(),
            h1);
      }

      // bias
      vector->insertLast(1.0);
      for (int* i = activeTiles; i < activeTiles + 48; ++i)
        vector->insertEntry(*i, 1.0);

      return *vector;
    }
    const SparseVector<T>& project(const DenseVector<O>& x)
    {

      vector->clear();
      // all 4
      tiles->tiles(&activeTiles[0], 12, vector->dimension() - 1, x(), x.dimension());
      // 3 of 4
      static DenseVector<O> x3(3);
      static int x3o[4][3] =
      {
      { 0, 1, 2 },
      { 1, 2, 3 },
      { 2, 3, 0 },
      { 1, 3, 0 } };
      for (int i = 0; i < 4; i++)
      {
        for (int j = 0; j < 3; j++)
          x3[j] = x[x3o[i][j]];
        tiles->tiles(&activeTiles[12 + i * 3], 3, vector->dimension() - 1, x3(), x3.dimension());
      }
      // 2 of 6
      static DenseVector<O> x2(2);
      static int x2o[6][2] =
      {
      { 0, 1 },
      { 1, 2 },
      { 2, 3 },
      { 0, 3 },
      { 0, 2 },
      { 1, 3 } };
      for (int i = 0; i < 6; i++)
      {
        for (int j = 0; j < 2; j++)
          x2[j] = x[x2o[i][j]];
        tiles->tiles(&activeTiles[24 + i * 2], 2, vector->dimension() - 1, x2(), x2.dimension());
      }

      // 4 of 1
      static DenseVector<O> x1(1);
      static int x1o[4] =
      { 0, 1, 2, 3 };
      for (int i = 0; i < 4; i++)
      {
        x1[0] = x[x1o[i]];
        tiles->tiles(&activeTiles[36 + i], 3, vector->dimension() - 1, x1(), x1.dimension());
      }

      for (int* i = activeTiles; i < activeTiles + 48; ++i)
        vector->insertEntry(*i, 1.0);

      // bias
      vector->insertLast(1.0);

      return *vector;
    }

    double vectorNorm() const
    {
      return 48 + 1;
    }
    int dimension() const
    {
      return vector->dimension();
    }
};

// ====================== Mountain Car 3D =====================================
// Mountain Car 3D projector
template<class T, class O>
class MountainCar3DTilesProjector: public AdvancedTilesProjector<T, O>
{
  public:
};

// ====================== Acrobot projector ===================================

template<class T, class O>
class AcrobotTilesProjector: public AdvancedTilesProjector<T, O>
{
  public:
};

#endif /* EXTENDEDPROBLEMSTEST_H_ */
