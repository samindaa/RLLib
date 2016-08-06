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
 * Test.h
 *
 *  Created on: Dec 18, 2012
 *      Author: sam
 */

#ifndef HEADERTEST_H_
#define HEADERTEST_H_

#include <map>
#include <set>
#include <cmath>
#include <ctime>
#include <limits>
#include <string>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <fstream>
#include <istream>
#include <ostream>
#include <sstream>
#include <iostream>
#include <iterator>
#include <algorithm>

#include "RL.h"
#include "Trace.h"
#include "Vector.h"
#include "Projector.h"
#include "MountainCar.h"
#include "FourierBasis.h"
#include "NoisyInputSum.h"
#include "MountainCar3D.h"
#include "SwingPendulum.h"
#include "NoStateProblem.h"
#include "ControlAlgorithm.h"
#include "StateToStateAction.h"
#include "SupervisedAlgorithm.h"
#include "ContinuousGridworld.h"
//#include "StateGraph.h"

namespace RLLib
{

// Testing framework
  class RLLibTest
  {
    protected:
      std::vector<std::string> argv;

    public:
      RLLibTest()
      {
      }
    public:
      virtual ~RLLibTest()
      {
      }
    public:
      virtual void run() =0;
    public:
      virtual const char* getName() const =0;
    public:
      virtual void setArgv(const std::vector<std::string>& argv)
      {
        this->argv = argv;
      }
  };

  /** Test macro **/
#define RLLIB_TEST(NAME)                                                   \
class NAME;                                                                \
class NAME##Base : public RLLib::RLLibTest                                 \
{                                                                          \
  private:  typedef NAME##Base Me;                                         \
  public:  NAME##Base() :  RLLibTest() { }                                 \
  public:  virtual ~NAME##Base() {}                                        \
  public:  const char* getName() const { return #NAME ; }                  \
  public:  static const char* getNameStatic() { return #NAME ; }           \
};                                                                         \

  class RLLibTestRegistry
  {
    protected:
      std::map<std::string, RLLibTest*> registry;

    public:

      typedef std::map<std::string, RLLibTest*>::iterator iterator;
      typedef std::map<std::string, RLLibTest*>::const_iterator const_iterator;

      iterator begin()
      {
        return registry.begin();
      }

      const_iterator begin() const
      {
        return registry.begin();
      }

      iterator end()
      {
        return registry.end();
      }

      const_iterator end() const
      {
        return registry.end();
      }

      iterator find(const std::string& key)
      {
        return registry.find(key);
      }

      const_iterator find(const std::string& key) const
      {
        return registry.find(key);
      }

      const unsigned int dimension() const
      {
        return registry.size();
      }

      static RLLibTestRegistry* getInstance();
      static void deleteInstance();
      static void registerInstance(RLLibTest* test);

    protected:
      RLLibTestRegistry();
      ~RLLibTestRegistry();
      RLLibTestRegistry(RLLibTestRegistry const&);
      RLLibTestRegistry& operator=(RLLibTestRegistry const&);

    private:
      static RLLibTestRegistry* instance;
  };

  template<class T>
  class RLLibTestLoader
  {
    protected:
      T* theInstance;
    public:
      RLLibTestLoader() :
          theInstance(new T())
      {
        RLLibTestRegistry::registerInstance(theInstance);
      }

      ~RLLibTestLoader()
      {
        delete theInstance;
      }
  };

  /** Test make macro **/
#define RLLIB_TEST_MAKE(NAME) \
  RLLib::RLLibTestLoader<NAME> __theTestLoader##NAME;

  /** Test generic code **/
  typedef Trace<double> TraceDoubleType;
  typedef ATrace<double> ATraceDoubleType;
  typedef RTrace<double> RTraceDoubleType;
  typedef AMaxTrace<double> AMaxTraceDoubleType;
  typedef MaxLengthTrace<double> MaxLengthTraceDoubleType;

  enum TraceEnum
  {
    ATraceDouble = 0, RTraceDouble, AMaxTraceDouble, MaxLengthTraceDouble, TraceEnumDimension
  };

// Common assertion tests

  class VectorsTestsUtils
  {
    private:
      template<class T>
      static bool checkSparseVectorConsistency(const SparseVector<T>* v)
      {
        const int* indexesPositions = v->getIndexesPosition();
        const int* activeIndexes = v->nonZeroIndexes();
        int nbActiveCounted = 0;
        bool positionChecked[v->nonZeroElements()];
        std::fill(positionChecked, positionChecked + v->nonZeroElements(), false);
        for (int index = 0; index < v->dimension(); index++)
        {
          const int position = indexesPositions[index];
          if (position == -1)
            continue;
          if (nbActiveCounted >= v->nonZeroElements())
            return false;
          if (positionChecked[position])
            return false;
          if (activeIndexes[position] != index)
            return false;
          positionChecked[position] = true;
          ++nbActiveCounted;
        }
        return nbActiveCounted != v->nonZeroElements() ? false : true;
      }

    public:
      template<class T>
      static bool checkConsistency(const Vector<T>* v)
      {
        const SparseVector<T>* vec = RTTI<T>::constSparseVector(v);
        if (!vec)
          return true;
        else
          return checkSparseVectorConsistency(vec);
      }

      template<class T>
      static bool checkValues(const Vector<T>* x)
      {
        const SparseVector<T>* v = RTTI<T>::constSparseVector(x);
        const T* values = x->getValues();
        int nbChecks = 0;
        if (v)
          nbChecks = v->nonZeroElements();
        else
          nbChecks = x->dimension();
        for (int position = 0; position < nbChecks; position++)
        {
          if (!Boundedness::checkValue(values[position]))
            return false;
        }
        return true;
      }

      template<class T>
      static bool checkVectorEquals(const Vector<T>* a, const Vector<T>* b, const double& margin)
      {
        if (!a || !b)
          return false;
        if (a == b) // pointer wise
          return true;
        if (a->dimension() != b->dimension())
          return false;
        for (int i = 0; i < a->dimension(); i++)
        {
          double diff = std::fabs(a->getEntry(i) - b->getEntry(i));
          if (diff > margin)
            return false;
        }
        return true;
      }

      template<class T>
      static double diff(const Vector<T>* a, const Vector<T>* b)
      {
        double value = 0;
        for (int i = 0; i < a->dimension(); ++i)
          value = std::max(value, std::fabs(a->getEntry(i) - b->getEntry(i)));
        return value;
      }

      template<class T>
      static bool checkVectorEquals(const Vector<T>* a, const Vector<T>* b)
      {
        return checkVectorEquals(a, b, 0);
      }

  };

  class Assert
  {
    public:
      template<class T>
      static void assertEquals(const Vector<T>* a, const Vector<T>* b)
      {
        ASSERT(VectorsTestsUtils::checkConsistency(a));
        ASSERT(VectorsTestsUtils::checkConsistency(b));
        ASSERT(VectorsTestsUtils::checkVectorEquals(a, b, std::numeric_limits<float>::epsilon()));
      }

      template<class T>
      static void assertEquals(const Vector<T>* a, const Vector<T>* b, double margin)
      {
        ASSERT(VectorsTestsUtils::checkConsistency(a));
        ASSERT(VectorsTestsUtils::checkConsistency(b));
        ASSERT(VectorsTestsUtils::checkVectorEquals(a, b, margin));
      }

      template<class T>
      static void checkConsistency(const Vector<T>* v)
      {
        ASSERT(VectorsTestsUtils::checkConsistency(v));
      }

      template<class T>
      static void checkValues(const Vector<T>* v)
      {
        ASSERT(VectorsTestsUtils::checkValues(v));
      }

      static void assertPasses(const bool& condition)
      {
        ASSERT(condition);
      }

      static void assertFails(const bool& condition)
      {
        ASSERT(!condition);
      }

      template<class T1, class T2>
      static void assertObjectEquals(const T1& a, const T2& b)
      {
        ASSERT(a == b);
      }

      template<class T1, class T2>
      static void assertObjectEquals(const T1& a, const T2& b, const double& margin)
      {
        //double tmp = std::fabs(a - b);
        ASSERT(std::fabs(a - b) <= margin);
      }

      template<class T>
      static void assertNotSame(const Vector<T>* a, const Vector<T>* b)
      {
        ASSERT(a != b);
      }
  };

  class Arrays
  {
    public:
      template<class T, unsigned int N>
      static inline unsigned int length(T (&pX)[N])
      {
        return N;
      }
  };

}  // namespace RLLib

using namespace std;
using namespace RLLib;

#endif /* HEADERTEST_H_ */
