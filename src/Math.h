/*
 * Math.h
 *
 *  Created on: Sep 3, 2012
 *      Author: sam
 */

#ifndef MATH_H_
#define MATH_H_

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

#endif /* MATH_H_ */
