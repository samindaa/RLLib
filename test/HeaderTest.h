/*
 * HeaderTest.h
 *
 *  Created on: Dec 18, 2012
 *      Author: sam
 */

#ifndef HEADERTEST_H_
#define HEADERTEST_H_

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <string>
#include <map>
#include <fstream>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <algorithm>
#include <limits>

#include "Vector.h"
#include "Trace.h"
#include "Math.h"

using namespace std;
using namespace RLLib;

typedef SparseVector<double> SVecDoubleType;
typedef SparseVector<float> SVecFloatType;
typedef DenseVector<float> DVecFloatType;
typedef Trace<double> TraceDoubleType;
typedef ATrace<double> ATraceDoubleType;
typedef RTrace<double> RTraceDoubleType;
typedef AMaxTrace<double> AMaxTraceDoubleType;
typedef MaxLengthTrace<double> MaxLengthTraceDoubleType;

enum TraceEnum
{
  ATraceDouble = 0, RTraceDouble, AMaxTraceDouble, MaxLengthTraceDouble, TraceEnumDimension
};

inline bool checkSparseVectorConsistency(const SVecDoubleType& v)
{
  const int* indexesPositions = v.getIndexesPosition();
  const int* activeIndexes = v.getActiveIndexes();
  int nbActiveCounted = 0;
  bool positionChecked[v.nbActiveEntries()];
  std::fill(positionChecked, positionChecked + v.nbActiveEntries(), false);
  for (int index = 0; index < v.dimension(); index++)
  {
    const int position = indexesPositions[index];
    if (position == -1)
      continue;
    if (nbActiveCounted >= v.nbActiveEntries())
      return false;
    if (positionChecked[position])
      return false;
    if (activeIndexes[position] != index)
      return false;
    positionChecked[position] = true;
    ++nbActiveCounted;
  }
  return nbActiveCounted != v.nbActiveEntries() ? false : true;
}

inline bool checkVectorEquals(const SVecDoubleType& a, const SVecDoubleType& b, double margin)
{
  if (a.dimension() != b.dimension())
    return false;
  for (int i = 0; i < a.dimension(); i++)
  {
    double diff = fabs(a.getEntry(i) - b.getEntry(i));
    if (diff > margin)
      return false;
  }
  return true;
}

inline void checkVectorEquals(const SVecDoubleType& a, const SVecDoubleType& b)
{
  assert(checkVectorEquals(a, b, numeric_limits<float>::epsilon()));
}

// Testing framework

class RLLibTestCase
{
  public:
    RLLibTestCase()
    {
    }
    virtual ~RLLibTestCase()
    {
    }
    virtual void run() =0;
    virtual const char* getName() const =0;
};

/** Test macro **/
#define RLLIB_TEST(NAME)                            \
class NAME;                                         \
class NAME##Base : public RLLibTestCase             \
{                                                   \
  public:                                           \
    NAME##Base() :  RLLibTestCase() {}              \
    virtual ~NAME##Base() {}                        \
    const char* getName() const { return #NAME ; }  \
};                                                  \

class RLLibTestRegistory
{
  protected:
    std::vector<RLLibTestCase*> registory;

  public:
    ~RLLibTestRegistory();


    typedef std::vector<RLLibTestCase*>::iterator iterator;
    typedef std::vector<RLLibTestCase*>::const_iterator const_iterator;

    iterator begin()
    {
      return registory.begin();
    }

    const_iterator begin() const
    {
      return registory.begin();
    }

    iterator end()
    {
      return registory.end();
    }

    const_iterator end() const
    {
      return registory.end();
    }

    static RLLibTestRegistory* instance();
    static void registerInstance(RLLibTestCase* testCase);

  protected:
    RLLibTestRegistory()
    {
    }

    RLLibTestRegistory(RLLibTestRegistory const&);
    RLLibTestRegistory& operator=(RLLibTestRegistory const&);

  private:
    static RLLibTestRegistory* inst;
};

template<class T>
class RLLibTestCaseLoader
{
  protected:
    T theInstance;
  public:
    RLLibTestCaseLoader()
    {
      RLLibTestRegistory::registerInstance(&theInstance);
    }
    virtual ~RLLibTestCaseLoader() {}
};

/** Test make macro **/
#define RLLIB_TEST_MAKE(NAME) \
  RLLibTestCaseLoader<NAME> _testcase_loader_##NAME;

#endif /* HEADERTEST_H_ */
