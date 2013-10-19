/*
 * Vec.h
 *
 *  Created on: Oct 11, 2013
 *      Author: sam
 */

#ifndef VEC_H_
#define VEC_H_

#include <cmath>
#include <iostream>

namespace RLLibViz
{

class Vec
{
public:
  double x;
  double y;
  double z;
  double w;

  //
  //  --- Constructors and Destructors ---
  //

  Vec() :
      x(0.0f), y(0.0f), z(0.0f), w(0.0f)
  {
  }

  Vec(double const& x, double const& y, double const& z = 0.0f, double const & w = 0.0f) :
      x(x), y(y), z(z), w(w)
  {
  }

  Vec(const Vec& v) :
      x(v.x), y(v.y), z(v.z), w(v.w)
  {
  }

  double& operator [](int i)
  {
    return *(&x + i);
  }
  const double& operator [](int i) const
  {
    return *(&x + i);
  }

  Vec operator -() const  // unary minus operator
  {
    return Vec(-x, -y, -z, -w);
  }

  Vec operator +(const Vec& v) const
  {
    return Vec(x + v.x, y + v.y, z + v.z, w + v.w);
  }

  Vec operator -(const Vec& v) const
  {
    return Vec(x - v.x, y - v.y, z - v.z, w - v.w);
  }

  Vec operator *(const double s) const
  {
    return Vec(s * x, s * y, s * z, s * w);
  }

  Vec operator *(const Vec& v) const
  {
    return Vec(x * v.x, y * v.y, z * v.z, w * v.z);
  }

  friend Vec operator *(const double s, const Vec& v)
  {
    return v * s;
  }

  Vec operator /(const double s) const
  {
#ifdef RLLIBVIZ_DEBUG
    static double divideByZeroTolerance = 1.0e-07;
    if ( std::fabs(s) < DivideByZeroTolerance )
    {
      std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] "
      << "Division by zero" << std::endl;
      return Vec();
    }
#endif // RLLIBVIZ_DEBUG
    double r = double(1.0) / s;
    return *this * r;
  }

  //
  //  --- (modifying) Arithematic Operators ---
  //

  Vec& operator +=(const Vec& v)
  {
    x += v.x;
    y += v.y;
    z += v.z;
    w += v.w;
    return *this;
  }

  Vec& operator -=(const Vec& v)
  {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    w -= v.w;
    return *this;
  }

  Vec& operator *=(const double s)
  {
    x *= s;
    y *= s;
    z *= s;
    w *= s;
    return *this;
  }

  Vec& operator *=(const Vec& v)
  {
    x *= v.x, y *= v.y, z *= v.z, w *= v.w;
    return *this;
  }

  Vec& operator /=(const double s)
  {
#ifdef RLLIBVIZ_DEBUG
    static double divideByZeroTolerance = 1.0e-07;
    if ( std::fabs(s) < DivideByZeroTolerance )
    {
      std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] "
      << "Division by zero" << std::endl;
    }
#endif // RLLIBVIZ_DEBUG
    double r = 1.0f / s;
    *this *= r;
    return *this;
  }

  friend std::ostream& operator <<(std::ostream& os, const Vec& v)
  {
    return os << "( " << v.x << ", " << v.y << ", " << v.z << ", " << v.w << " )";
  }

  friend std::istream& operator >>(std::istream& is, Vec& v)
  {
    return is >> v.x >> v.y >> v.z >> v.w;
  }

  //  --- Conversion Operators ---
  //

  operator const double*() const
  {
    return static_cast<const double*>(&x);
  }

  operator double*()
  {
    return static_cast<double*>(&x);
  }
};

inline double dot(const Vec& u, const Vec& v)
{
  return u.x * v.x + u.y * v.y + u.z * v.z + u.w + v.w;
}

inline double length(const Vec& v)
{
  return std::sqrt(dot(v, v));
}

inline Vec normalize(const Vec& v)
{
  return v / length(v);
}

inline Vec cross(const Vec& a, const Vec& b)
{
  return Vec(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

}  // namespace RLLibViz

#endif /* VEC_H_ */
