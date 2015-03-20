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
 * ExtendedProblemsTest.h
 *
 *  Created on: May 15, 2013
 *      Author: sam
 */

#ifndef EXTENDEDPROBLEMSTEST_H_
#define EXTENDEDPROBLEMSTEST_H_

#include "Test.h"

// From the RLLib
#include "Vector.h"
#include "Trace.h"
#include "Projector.h"
#include "ControlAlgorithm.h"

// From the simulation
#include "MountainCar3D.h"
#include "Acrobot.h"
#include "RandlovBike.h"
#include "PoleBalancing.h"
#include "TorquedPendulum.h"
#include "UnderwaterVehicle.h"

#include "util/Spline.h"
#include "util/RK4.h"

using namespace std;
RLLIB_TEST(ExtendedProblemsTest)

class ExtendedProblemsTest: public ExtendedProblemsTestBase
{
  public:
    ExtendedProblemsTest()
    {
    }
    virtual ~ExtendedProblemsTest()
    {
    }
    void run();

  private:
    void testOffPACMountainCar3D_1();
    void testGreedyGQMountainCar3D();
    void testSarsaMountainCar3D();
    void testOffPACMountainCar3D_2();
    void testOffPACAcrobot();
    void testGreedyGQAcrobot();

    void testTrueSarsaUnderwaterVehicle();

    void testPoleBalancingPlant();
    void testPersistResurrect();
    void testTorquedPendulum();
    void testSupervisedProjector();

    void testFunction1RK4();
    void testFunction2RK4();

    // RK tests
    class Function1: public RK4
    {
      public:
        Function1(const int& m, const double& dt) :
            RK4(m, dt)
        {
        }

        void f(const double& time, const Action<double>* action, const Vector<double>* x,
            Vector<double>* x_dot)
        {
          x_dot->setEntry(0, x->getEntry(0) * cos(time));
        }

    };

    class Function2: public RK4
    {
      public:
        Function2(const int& m, const double& dt) :
            RK4(m, dt)
        {
        }

        void f(const double& time, const Action<double>* action, const Vector<double>* x,
            Vector<double>* x_dot)
        {
          x_dot->setEntry(0, +x->getEntry(1));
          x_dot->setEntry(1, -x->getEntry(0));
        }
    };

};

// Helpers
class CombinationGenerator
{
  public:
    typedef std::map<int, vector<int> > Combinations;

  private:
    int n;
    int k;
    Combinations combinations;
    vector<int> input;
    vector<int> combination;

  public:
    CombinationGenerator(const int& n, const int& k, const int& nbVars) :
        n(n), k(k)
    {
      for (int i = 0; i < nbVars; i++)
        input.push_back(i);
    }

  private:
    void addCombination(const vector<int>& v)
    {
      combinations.insert(make_pair(combinations.size(), v));
    }

    void nextCombination(int offset, int k)
    {
      if (k == 0)
      {
        addCombination(combination);
        return;
      }
      for (int i = offset; i <= n - k; ++i)
      {
        combination.push_back(input[i]);
        nextCombination(i + 1, k - 1);
        combination.pop_back();
      }
    }

  public:
    Combinations& getCombinations()
    {
      nextCombination(0, k);
      return combinations;
    }
};

// ====================== Advanced projector ===================================
template<class T>
class AdvancedTilesProjector: public Projector<T>
{
  protected:
    Hashing<T>* hashing;
    Tiles<T>* tiles;
    Vector<T>* vector;
    T gridResolution;

  public:
    AdvancedTilesProjector(Random<T>* random) :
        hashing(new MurmurHashing<T>(random, 1000000)), tiles(new Tiles<T>(hashing)), vector(
            new SVector<T>(hashing->getMemorySize() + 1)), gridResolution(6)
    {
    }

    virtual ~AdvancedTilesProjector()
    {
      delete hashing;
      delete tiles;
      delete vector;
    }

  public:
    const Vector<T>* project(const Vector<T>* x, const int& h2)
    {
      vector->clear();
      if (x->empty())
        return vector;
      int h1 = 0;
      static PVector<T> x4(4);
      x4.set(x)->mapMultiplyToSelf(gridResolution);
      // all 4
      tiles->tiles(vector, 12, &x4, h1++, h2);
      // 3 of 4
      static CombinationGenerator cg43(4, 3, 4); // We know x.dimension() == 4
      static CombinationGenerator::Combinations& c43 = cg43.getCombinations();
      static PVector<T> x3(3);
      for (int i = 0; i < (int) c43.size(); i++)
      {
        for (int j = 0; j < (int) c43[i].size(); j++)
          x3[j] = x->getEntry(c43[i][j]) * gridResolution;
        tiles->tiles(vector, 3, &x3, h1++, h2);
      }
      // 2 of 6
      static CombinationGenerator cg42(4, 2, 4);
      static CombinationGenerator::Combinations& c42 = cg42.getCombinations();
      static PVector<T> x2(2);
      for (int i = 0; i < (int) c42.size(); i++)
      {
        for (int j = 0; j < (int) c42[i].size(); j++)
          x2[j] = x->getEntry(c42[i][j]) * gridResolution;
        tiles->tiles(vector, 2, &x2, h1++, h2);
      }

      // 1 of 4
      static CombinationGenerator cg41(4, 1, 4);
      static CombinationGenerator::Combinations& c41 = cg41.getCombinations();
      static PVector<T> x1(1);
      for (int i = 0; i < (int) c41.size(); i++)
      {
        x1[0] = x->getEntry(c41[i][0]) * gridResolution; // there is only a single element
        tiles->tiles(vector, 3, &x1, h1++, h2);
      }

      // bias
      vector->setEntry(vector->dimension() - 1, 1.0);
      return vector;
    }

    const Vector<T>* project(const Vector<T>* x)
    {

      vector->clear();
      if (x->empty())
        return vector;
      int h1 = 0;
      static PVector<T> x4(4);
      x4.set(x)->mapMultiplyToSelf(gridResolution);
      // all 4
      tiles->tiles(vector, 12, &x4, h1++);
      // 3 of 4
      static CombinationGenerator cg43(4, 3, 4); // We know x.dimension() == 4
      static CombinationGenerator::Combinations& c43 = cg43.getCombinations();
      static PVector<T> x3(3);
      for (int i = 0; i < (int) c43.size(); i++)
      {
        for (int j = 0; j < (int) c43[i].size(); j++)
          x3[j] = x->getEntry(c43[i][j]) * gridResolution;
        tiles->tiles(vector, 3, &x3, h1++);
      }
      // 2 of 6
      static CombinationGenerator cg42(4, 2, 4);
      static CombinationGenerator::Combinations& c42 = cg42.getCombinations();
      static PVector<T> x2(2);
      for (int i = 0; i < (int) c42.size(); i++)
      {
        for (int j = 0; j < (int) c42[i].size(); j++)
          x2[j] = x->getEntry(c42[i][j]) * gridResolution;
        tiles->tiles(vector, 2, &x2, h1++);
      }

      // 1 of 4
      static CombinationGenerator cg41(4, 1, 4);
      static CombinationGenerator::Combinations& c41 = cg41.getCombinations();
      static PVector<T> x1(1);
      for (int i = 0; i < (int) c41.size(); i++)
      {
        x1[0] = x->getEntry(c41[i][0]) * gridResolution; // there is only a single element
        tiles->tiles(vector, 3, &x1, h1++);
      }

      // bias
      vector->setEntry(vector->dimension() - 1, 1.0);
      return vector;
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
template<class T>
class MountainCar3DTilesProjector: public AdvancedTilesProjector<T>
{
  public:
    MountainCar3DTilesProjector(Random<T>* random) :
        AdvancedTilesProjector<T>(random)
    {
    }
};

// ====================== Acrobot projector ===================================

template<class T>
class AcrobotTilesProjector: public AdvancedTilesProjector<T>
{
  public:
    AcrobotTilesProjector(Random<T>* random) :
        AdvancedTilesProjector<T>(random)
    {
    }
};

#endif /* EXTENDEDPROBLEMSTEST_H_ */
