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
 * Math.h
 *
 *  Created on: Sep 3, 2012
 *      Author: sam
 */

#ifndef MATHEMA_H_
#define MATHEMA_H_

#include <cmath>
#include <limits>
#include <limits.h>
#include <stdint.h>

#include "Vector.h"

// Visual Studio 2013
#ifdef _MSC_VER
#define NOMINMAX
#endif

namespace RLLib
{

  // Some number checking

  class Boundedness
  {
    public:
      template<typename T>
      inline static bool checkValue(const T& value)
      {
        return !std::isnan(value) && !std::isinf(value);
      }

      template<typename T>
      inline static bool checkDistribution(const Vector<T>* distribution)
      {
        T sum = T(0);
        for (int i = 0; i < distribution->dimension(); i++)
          sum += distribution->getEntry(i);
        return std::abs(1.0f - sum) < 1e-6f;/*for stability*/
      }
  };

//-----------------------------------------------------------------------------
// Xorshift RNG based on code by George Marsaglia
// http://en.wikipedia.org/wiki/Xorshift

  class Xorshift
  {
    private:
      uint32_t x;
      uint32_t y;
      uint32_t z;
      uint32_t w;

    public:
      Xorshift()
      {
        reseed(uint32_t(0));
      }

      Xorshift(uint32_t seed)
      {
        reseed(seed);
      }

      void reseed(uint32_t seed)
      {
        x = 0x498b3bc5 ^ seed;
        y = 0;
        z = 0;
        w = 0;

        for (int i = 0; i < 10; i++)
          mix();
      }

      void reseed(uint64_t seed)
      {
        x = 0x498b3bc5 ^ (uint32_t) (seed >> 0);
        y = 0x5a05089a ^ (uint32_t) (seed >> 32);
        z = 0;
        w = 0;

        for (int i = 0; i < 10; i++)
          mix();
      }

      //-----------------------------------------------------------------------------

      void mix(void)
      {
        uint32_t t = x ^ (x << 11);
        x = y;
        y = z;
        z = w;
        w = w ^ (w >> 19) ^ t ^ (t >> 8);
      }

      uint32_t rand_u32(void)
      {
        mix();

        return x;
      }

      uint64_t rand_u64(void)
      {
        mix();

        uint64_t a = x;
        uint64_t b = y;

        return (a << 32) | b;
      }

      void rand_p(void * blob, int bytes)
      {
        uint32_t * blocks = reinterpret_cast<uint32_t*>(blob);

        while (bytes >= 4)
        {
          blocks[0] = rand_u32();
          blocks++;
          bytes -= 4;
        }

        uint8_t * tail = reinterpret_cast<uint8_t*>(blocks);

        for (int i = 0; i < bytes; i++)
        {
          tail[i] = (uint8_t) rand_u32();
        }
      }
  };

// Important distributions
  template<typename T>
  class Random
  {
    private:
      Xorshift xorshift;

    public:
      Random()
      {
      }

      inline void reseed(const uint32_t& seed)
      {
        //::srand(seed);
        xorshift.reseed(seed);
      }

      inline int rand()
      {
        //return ::rand();
        return xorshift.rand_u32() % RAND_MAX;
      }

      inline uint32_t randu32(void)
      {
        return xorshift.rand_u32();
      }

      // [0 .. size)
      inline int nextInt(const int& size)
      {
        return rand() % size;
      }

      // [0..1)
      inline T nextReal()
      {
        return T(rand()) / static_cast<T>(RAND_MAX);
      }

      // A gaussian random deviate
      inline T nextNormalGaussian()
      {
        T r, v1, v2;
        do
        {
          v1 = T(2) * nextReal() - T(1);
          v2 = T(2) * nextReal() - T(1);
          r = v1 * v1 + v2 * v2;
        }
        while (r >= 1.0 || r == 0);
        const T fac(sqrt(-T(2) * log(r) / r));
        return v1 * fac;
      }

      inline T gaussianProbability(const T& x, const T& m, const T& s) const
      {
        return exp(-0.5f * pow((x - m) / s, 2)) / (s * sqrt(2.0f * M_PI));
      }

      // http://en.literateprograms.org/Box-Muller_transform_(C)
      inline T nextGaussian(const T& mean, const T& stddev)
      {
        static T n2 = T(0);
        static int n2_cached = 0;
        if (!n2_cached)
        {
          T x, y, r;
          do
          {
            x = T(2) * nextReal() - T(1);
            y = T(2) * nextReal() - T(1);

            r = x * x + y * y;
          }
          while (r == T(0) || r > T(1));
          {
            T d = sqrt(-T(2) * log(r) / r);
            T n1 = x * d;
            n2 = y * d;
            T result = n1 * stddev + mean;
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

  };

// Helper class for range management for testing environments
  template<typename T>
  class Range
  {
    private:
      T minValue, maxValue;

    public:
      Range(const T& minValue = std::numeric_limits<T>::min(), const T& maxValue =
          std::numeric_limits<T>::max()) :
          minValue(minValue), maxValue(maxValue)
      {
      }

      T bound(const T& value) const
      {
        return std::max(minValue, std::min(maxValue, value));
      }

      bool in(const T& value) const
      {
        return value >= minValue && value <= maxValue;
      }

      T length() const
      {
        return maxValue - minValue;
      }

      T min() const
      {
        return minValue;
      }

      T max() const
      {
        return maxValue;
      }

      T center() const
      {
        return min() + (length() / 2.0f);
      }

      T choose(Random<T>* random) const
      {
        return random->nextReal() * length() + min();
      }

      // Unit output [0,1]
      T toUnit(const T& value) const
      {
        return (bound(value) - min()) / length();
      }

      T toUnit(const T& value, const Range<T>* other) const
      {
        return ((bound(value) - min()) / length()) * other->length() + other->min();
      }
  };

  template<typename T>
  class Ranges
  {
    protected:
      typename std::vector<Range<T>*> ranges;
    public:
      typedef typename std::vector<Range<T>*>::iterator iterator;
      typedef typename std::vector<Range<T>*>::const_iterator const_iterator;

      Ranges()
      {
      }

      ~Ranges()
      {
        ranges.clear();
      }

      Ranges(const Range<T>& that)
      {
        for (typename Vectors<T>::iterator iter = that.begin(); iter != that.end(); ++iter)
          ranges.push_back(*iter);
      }

      Ranges<T>& operator=(const Ranges<T>& that)
      {
        if (this != that)
        {
          ranges.clear();
          for (typename Vectors<T>::iterator iter = that.begin(); iter != that.end(); ++iter)
            ranges.push_back(*iter);
        }
        return *this;
      }

      void push_back(Range<T>* range)
      {
        ranges.push_back(range);
      }

      iterator begin()
      {
        return ranges.begin();
      }

      const_iterator begin() const
      {
        return ranges.begin();
      }

      iterator end()
      {
        return ranges.end();
      }

      const_iterator end() const
      {
        return ranges.end();
      }

      int dimension() const
      {
        return ranges.size();
      }

      Range<T>& operator[](const int& index)
      {
        ASSERT(index < static_cast<int>(ranges.size()));
        return *ranges[index];
      }

      const Range<T>& operator[](const int& index) const
      {
        ASSERT(index < static_cast<int>(ranges.size()));
        return *ranges[index];
      }

      Range<T>* at(const int& index)
      {
        ASSERT(index < static_cast<int>(ranges.size()));
        return ranges[index];
      }

      const Range<T>* at(const int& index) const
      {
        ASSERT(index < static_cast<int>(ranges.size()));
        return ranges[index];
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

  class Angle
  {
    public:
      /**
       * Normalize to [-pi, pi)
       */
      template<typename T>
      inline static T normalize(const T& data)
      {
        T x = fmod(data + M_PI, 2.0 * M_PI);
        if (x < 0)
          x += 2.0 * M_PI;
        return x - M_PI;
      }
  };

  template<class T, int n>
  class History
  {
    private:
      int current;
      int numberOfEntries;
      T buffer[n];
      T sum;

    public:

      History()
      {
        init();
      }

      inline void init()
      {
        current = n - 1;
        numberOfEntries = 0;
        sum = T();
      }

      void add(const T& value)
      {
        if (numberOfEntries == n)
          sum -= getEntry(numberOfEntries - 1);
        sum += value;
        current++;
        current %= n;
        if (++numberOfEntries >= n)
          numberOfEntries = n;
        buffer[current] = value;
      }

      void fill(const T& value)
      {
        for (int i = 0; i < n; i++)
          add(value);
      }

      T getEntry(const int& i) const
      {
        return buffer[(n + current - i) % n];
      }

      T getSum() const
      {
        return sum;
      }

      T getMinimum() const
      {
        // Return 0 if buffer is empty
        if (0 == numberOfEntries)
          return T();

        T min = buffer[0];
        for (int i = 0; i < numberOfEntries; i++)
        {
          if (buffer[i] < min)
            min = buffer[i];
        }
        return min;
      }

      T getAverage() const
      {
        // Return 0 if buffer is empty
        if (0 == numberOfEntries)
          return T();

        return (sum / numberOfEntries);
      }

      T operator[](const int& i) const
      {
        return buffer[(n + current - i) % n];
      }

      int getNumberOfEntries() const
      {
        return numberOfEntries;
      }

      int getMaxEntries() const
      {
        return n;
      }
  };

} // namespace RLLib

#endif /* MATHEMA_H_ */
