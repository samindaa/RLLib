/*
 * Math.h
 *
 *  Created on: Sep 3, 2012
 *      Author: sam
 */

#ifndef MATH_H_
#define MATH_H_

#include <cmath>
#include <limits>

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
};

template<typename T>
inline int sgn(T val)
{
  return (T(0) < val) - (val < T(0));
}

// Important distributions

class Gaussian
{
  public:
    inline static float probability(const float& x, const float& mu,
        const float& sigma)
    {
      return exp(-pow((x - mu), 2) / (2.0 * pow(sigma, 2)))
          / (sigma * sqrt(2.0 * M_PI));
    }

    // http://en.literateprograms.org/Box-Muller_transform_(C)
    inline double nextGaussian(const double& mean = 0, const double& stddev =
        1.0)
    {
      static double n2 = 0.0;
      static int n2_cached = 0;
      if (!n2_cached)
      {
        double x, y, r;
        do
        {
          x = drand48() - 1;
          y = drand48() - 1;

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
};

#endif /* MATH_H_ */
