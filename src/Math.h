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
 * Math.h
 *
 *  Created on: Sep 3, 2012
 *      Author: sam
 */

#ifndef MATH_H_
#define MATH_H_

#include <cmath>
#include <cassert>
#include <limits>
#include <vector>
#include "Vector.h"

namespace RLLib
{

// Some number checking

class Boundedness
{
  public:
    template<class T>
    inline static bool checkValue(const T& value)
    {
      bool bvalue = !isnan(value) && !isinf(value);
      assert(bvalue);
      return bvalue;
    }

    template<class T>
    inline static bool checkDistribution(const DenseVector<T>& distribution)
    {
      double sum = 0.0;
      for (int i = 0; i < distribution.dimension(); i++)
        sum += distribution[i];
      bool bvalue = ::fabs(1.0 - sum) < 10e-8;
      assert(bvalue);
      return bvalue;
    }
};

// Helper class for range management for testing environments
template<class T>
class Range
{
  private:
    T minv, maxv;

  public:
    Range(const T& minv = std::numeric_limits<T>::min(), const T& maxv =
        std::numeric_limits<T>::max()) :
        minv(minv), maxv(maxv)
    {
    }

    T bound(const T& value) const
    {
      return std::max(minv, std::min(maxv, value));
    }

    bool in(const T& value) const
    {
      return value >= minv && value <= maxv;
    }

    T length() const
    {
      return maxv - minv;
    }

    T min() const
    {
      return minv;
    }

    T max() const
    {
      return maxv;
    }

    T center() const
    {
      return min() + (length() / 2.0);
    }
};

template<class T>
class Ranges
{
  protected:
    typename std::vector<Range<T>*>* ranges;
  public:
    typedef typename std::vector<Range<T>*>::iterator iterator;
    typedef typename std::vector<Range<T>*>::const_iterator const_iterator;

    Ranges() :
        ranges(new std::vector<Range<T>*>())
    {
    }

    ~Ranges()
    {
      ranges->clear();
      delete ranges;
    }

    Ranges(const Range<T>& that) :
        ranges(new std::vector<Range<T>*>())
    {
      for (typename SparseVectors<T>::iterator iter = that.begin(); iter != that.end(); ++iter)
        ranges->push_back(*iter);
    }

    Ranges<T>& operator=(const Ranges<T>& that)
    {
      if (this != that)
      {
        ranges->clear();
        for (typename SparseVectors<T>::iterator iter = that.begin(); iter != that.end(); ++iter)
          ranges->push_back(*iter);
      }
      return *this;
    }

    void push_back(Range<T>* range)
    {
      ranges->push_back(range);
    }

    iterator begin()
    {
      return ranges->begin();
    }

    const_iterator begin() const
    {
      return ranges->begin();
    }

    iterator end()
    {
      return ranges->end();
    }

    const_iterator end() const
    {
      return ranges->end();
    }

    unsigned int dimension() const
    {
      return ranges->size();
    }

    const Range<T>& operator[](const unsigned index) const
    {
      assert(index >= 0 && index < dimension());
      return *ranges->at(index);
    }

    Range<T>* at(const unsigned index) const
    {
      assert(index >= 0 && index < dimension());
      return ranges->at(index);
    }
};

class Signum
{
  public:
    template<typename T>
    inline static int valueOf(const T& val)
    {
      return (T(0) < val) - (val < T(0));
    }
};

// Important distributions
class Probabilistic
{
  public:

    inline static void srand(unsigned int& seed)
    {
      ::srand(seed);
    }

    // [0..1]
    inline static float nextFloat()
    {
      return float(rand()) * (1.0f / static_cast<float>(RAND_MAX));
    }

    // [0..1]
    inline static double nextDouble()
    {
      return double(rand()) / RAND_MAX;
    }

    // A gaussian random deviate
    inline static double nextNormalGaussian()
    {
      double r, v1, v2;
      do
      {
        v1 = 2.0 * nextDouble() - 1.0;
        v2 = 2.0 * nextDouble() - 1.0;
        r = v1 * v1 + v2 * v2;
      } while (r >= 1.0 || r == 0);
      const double fac(sqrt(-2.0 * log(r) / r));
      return v1 * fac;
    }

    inline static double gaussianProbability(const float& x, const float& m, const float& s)
    {
      return exp(-0.5 * pow((x - m) / s, 2)) / (s * sqrt(2.0 * M_PI));
    }

    // http://en.literateprograms.org/Box-Muller_transform_(C)
    inline static double nextGaussian(const double& mean, const double& stddev)
    {
      static double n2 = 0.0;
      static int n2_cached = 0;
      if (!n2_cached)
      {
        double x, y, r;
        do
        {
          x = 2.0 * nextDouble() - 1;
          y = 2.0 * nextDouble() - 1;

          r = x * x + y * y;
        } while (r == 0.0 || r > 1.0);
        {
          double d = sqrt(-2.0 * log(r) / r);
          double n1 = x * d;
          n2 = y * d;
          double result = n1 * stddev + mean;
          n2_cached = 1;
          return result;
        }
      }
      else
      {
        n2_cached = 0;
        return n2 * stddev + mean;
      }
    }

    inline float sampleNormalDistribution(float b)
    {
      float result(0.0f);
      for (int i = 0; i < 12; i++)
        result += 2.0f * ((nextFloat() - 0.5f) * b);
      return result / 2.0f;
    }

    inline float sampleTriangularDistribution(float b)
    {
      float randResult = 2.0f * ((nextFloat() - 0.5f) * b) + 2.0f * ((nextFloat() - 0.5f) * b);
      return (sqrt(6.0f) / 2.0f) * randResult;
    }
};

} // namespace RLLib

#endif /* MATH_H_ */
