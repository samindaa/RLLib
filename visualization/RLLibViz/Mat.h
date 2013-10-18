/*
 * Mat.h
 *
 *  Created on: Oct 11, 2013
 *      Author: sam
 */

#ifndef MAT_H_
#define MAT_H_

#include "Vec.h"

namespace RLLibViz
{

class Mat
{
    Vec m[4];

  public:

    Mat(const double d = 1.0f)  // Create a diagional matrix
    {
      m[0].x = d;
      m[1].y = d;
      m[2].z = d;
      m[3].w = d;
    }

    Mat(const Vec& a, const Vec& b, const Vec& c, const Vec& d)
    {
      m[0] = a;
      m[1] = b;
      m[2] = c;
      m[3] = d;
    }

    Mat(double const& m00, double const& m10, double const& m20, double const& m30,
        double const& m01, double const& m11, double const& m21, double const& m31,
        double const& m02, double const& m12, double const& m22, double const& m32,
        double const& m03, double const& m13, double const& m23, double const& m33)
    {
      m[0] = Vec(m00, m01, m02, m03);
      m[1] = Vec(m10, m11, m12, m13);
      m[2] = Vec(m20, m21, m22, m23);
      m[3] = Vec(m30, m31, m32, m33);
    }

    Mat(const Mat& that)
    {
      if (*this != that)
      {
        m[0] = that[0];
        m[1] = that[1];
        m[2] = that[2];
        m[3] = that[3];
      }
    }

    Mat& operator =(const Mat& that)
    {
      if (*this != that)
      {
        m[0] = that[0];
        m[1] = that[1];
        m[2] = that[2];
        m[3] = that[3];
      }
      return *this;
    }

    Vec& operator [](int i)
    {
      return m[i];
    }
    const Vec& operator [](int i) const
    {
      return m[i];
    }

    Mat operator +(const Mat& that) const
    {
      return Mat(that[0] + that[0], that[1] + that[1], that[2] + that[2], that[3] + that[3]);
    }

    Mat operator -(const Mat& that) const
    {
      return Mat(that[0] - that[0], that[1] - that[1], that[2] - that[2], that[3] - that[3]);
    }

    Mat operator *(const double s) const
    {
      return Mat(s * m[0], s * m[1], s * m[2], s * m[3]);
    }

    Mat operator /(const double s) const
    {
#ifdef RLLIBVIZ_DEBUG
      static double divideByZeroTolerance = 1.0e-07;
      if ( std::fabs(s) < DivideByZeroTolerance )
      {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] "
        << "Division by zero" << std::endl;
        return Mat();
      }
#endif // RLLIBVIZ_DEBUG
      double r = 1.0f / s;
      return *this * r;
    }

    friend Mat operator *(const double s, const Mat& that)
    {
      return that * s;
    }

    Mat operator *(const Mat& that) const
    {
      Mat a(0.0);

      for (int i = 0; i < 4; ++i)
      {
        for (int j = 0; j < 4; ++j)
        {
          for (int k = 0; k < 4; ++k)
          {
            a[i][j] += m[i][k] * that[k][j];
          }
        }
      }

      return a;
    }

    Mat& operator +=(const Mat& that)
    {
      m[0] += that[0];
      m[1] += that[1];
      m[2] += that[2];
      m[3] += that[3];
      return *this;
    }

    Mat& operator -=(const Mat& that)
    {
      m[0] -= that[0];
      m[1] -= that[1];
      m[2] -= that[2];
      m[3] -= that[3];
      return *this;
    }

    Mat& operator *=(const double s)
    {
      m[0] *= s;
      m[1] *= s;
      m[2] *= s;
      m[3] *= s;
      return *this;
    }

    Mat& operator *=(const Mat& that)
    {
      Mat a(0.0);

      for (int i = 0; i < 4; ++i)
      {
        for (int j = 0; j < 4; ++j)
        {
          for (int k = 0; k < 4; ++k)
          {
            a[i][j] += m[i][k] * that[k][j];
          }
        }
      }

      return *this = a;
    }

    Mat& operator /=(const double s)
    {
#ifdef RLLIBVIZ_DEBUG
      static double divideByZeroTolerance = 1.0e-07;
      if ( std::fabs(s) < DivideByZeroTolerance )
      {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] "
        << "Division by zero" << std::endl;
        return Mat();
      }
#endif // RLLIBVIZ_DEBUG
      double r = double(1.0) / s;
      return *this *= r;
    }

    Vec operator *(const Vec& v) const
    {  // m * v
      return Vec(m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3] * v.w,
          m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3] * v.w,
          m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3] * v.w,
          m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3] * v.w);
    }

    friend std::ostream& operator <<(std::ostream& os, const Mat& m)
    {
      return os << std::endl << m[0] << std::endl << m[1] << std::endl << m[2] << std::endl << m[3]
          << std::endl;
    }

    friend std::istream& operator >>(std::istream& is, Mat& m)
    {
      return is >> m.m[0] >> m.m[1] >> m.m[2] >> m.m[3];
    }

    operator const double*() const
    {
      return static_cast<const double*>(&m[0].x);
    }

    operator double*()
    {
      return static_cast<double*>(&m[0].x);
    }

    inline static Mat matrixCompMult(const Mat& A, const Mat& B)
    {
      return Mat(A[0][0] * B[0][0], A[0][1] * B[0][1], A[0][2] * B[0][2], A[0][3] * B[0][3],
          A[1][0] * B[1][0], A[1][1] * B[1][1], A[1][2] * B[1][2], A[1][3] * B[1][3],
          A[2][0] * B[2][0], A[2][1] * B[2][1], A[2][2] * B[2][2], A[2][3] * B[2][3],
          A[3][0] * B[3][0], A[3][1] * B[3][1], A[3][2] * B[3][2], A[3][3] * B[3][3]);
    }

    inline static Mat transpose(const Mat& A)
    {
      return Mat(A[0][0], A[1][0], A[2][0], A[3][0], A[0][1], A[1][1], A[2][1], A[3][1], A[0][2],
          A[1][2], A[2][2], A[3][2], A[0][3], A[1][3], A[2][3], A[3][3]);
    }

    inline static Mat rotateX(double const& angle)
    {

      Mat c;
      c[2][2] = c[1][1] = cos(angle);
      c[2][1] = sin(angle);
      c[1][2] = -c[2][1];
      return c;
    }

    inline static Mat rotateY(double const& angle)
    {
      Mat c;
      c[2][2] = c[0][0] = cos(angle);
      c[0][2] = sin(angle);
      c[2][0] = -c[0][2];
      return c;
    }

    inline static Mat rotateZ(double const& angle)
    {
      Mat c;
      c[0][0] = c[1][1] = cos(angle);
      c[1][0] = sin(angle);
      c[0][1] = -c[1][0];
      return c;
    }

    inline static Mat translate(double const& x, double const& y, double const& z)
    {
      Mat c;
      c[0][3] = x;
      c[1][3] = y;
      c[2][3] = z;
      return c;
    }

    inline static Mat translate(const Vec& v)
    {
      return translate(v.x, v.y, v.z);
    }

    inline static Mat scale(double const& x, double const& y, double const& z)
    {
      Mat c;
      c[0][0] = x;
      c[1][1] = y;
      c[2][2] = z;
      return c;
    }

    inline static Mat scale(const Vec& v)
    {
      return scale(v.x, v.y, v.z);
    }

};

}  // namespace RLLibViz

#endif /* MAT_H_ */
