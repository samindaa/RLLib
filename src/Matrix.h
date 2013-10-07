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
 * Matrix.h
 *
 *  Created on: Oct 7, 2013
 *      Author: sam
 */

#ifndef MATRIX_H_
#define MATRIX_H_

#include <vector>
#include <cstdarg>
#include <iostream>

#ifdef DEBUG
#define ASSERT(b) { if (! (b)) { char msg[256]; sprintf(msg, "ASSERT failed in file %s, line %d", __FILE__, __LINE__); throw msg; } }
// #define ASSERT(b) { if (! (b)) {*((int*)(NULL)) = 0; } }   // trap the debugger
#else
#define ASSERT(b) { }
#endif

namespace RLLib
{

class Matrix
{
  protected:
    // thread-safe comparison predicate for indices
    class CmpIndex
    {
      public:
        CmpIndex(const Matrix& lambda, std::vector<unsigned int>& index) :
            m_lambda(lambda), m_index(index)
        {
        }

        bool operator ()(unsigned int i1, unsigned int i2) const
        {
          return (m_lambda(i1) > m_lambda(i2));
        }

      protected:
        const Matrix& m_lambda;
        std::vector<unsigned int>& m_index;
    };

    unsigned int m_rows;
    unsigned int m_cols;
    std::vector<double> m_data;

  public:
    inline bool isValid() const
    {
      return (m_data.size() > 0);
    }

    inline bool isVector() const
    {
      return (isColumnVector() || isRowVector());
    }

    inline bool isColumnVector() const
    {
      return (m_rows == 1);
    }

    inline bool isRowVector() const
    {
      return (m_cols == 1);
    }

    inline bool isSquare() const
    {
      return (m_rows == m_cols);
    }

    inline unsigned int rows() const
    {
      return m_rows;
    }

    inline unsigned int cols() const
    {
      return m_cols;
    }

    inline unsigned int size() const
    {
      return m_data.size();
    }

    // type cast for 1x1 matrices
    operator double() const
    {
      ASSERT(m_rows == 1 && m_cols == 1);
      return m_data[0];
    }

    inline double& operator [](unsigned int index)
    {
      ASSERT(index < m_data.size());
      return m_data[index];
    }
    inline const double& operator [](unsigned int index) const
    {
      ASSERT(index < m_data.size());
      return m_data[index];
    }

    inline double& operator ()(unsigned int index)
    {
      return operator [](index);
    }
    inline const double& operator ()(unsigned int index) const
    {
      return operator [](index);
    }

    inline double& operator ()(unsigned int y, unsigned int x)
    {
      return operator [](y * m_cols + x);
    }

    inline const double& operator ()(unsigned int y, unsigned int x) const
    {
      return operator [](y * m_cols + x);
    }

    Matrix row(unsigned int r) const
    {
      ASSERT(r < m_rows);
      Matrix ret(1, m_cols);
      unsigned int i;
      for (i = 0; i < m_cols; i++)
        ret(0, i) = operator ()(r, i);
      return ret;
    }

    Matrix col(unsigned int c) const
    {
      ASSERT(c < m_cols);
      Matrix ret(m_rows, 1);
      unsigned int i;
      for (i = 0; i < m_rows; i++)
        ret(i, 0) = operator ()(i, c);
      return ret;
    }

    void resize(unsigned int rows, unsigned int cols = 1)
    {
      m_rows = rows;
      m_cols = cols;
      unsigned int sz = rows * cols;
      m_data.resize(sz);
      if (sz > 0)
        memset(&m_data[0], 0, sz * sizeof(double));
    }

    const Matrix& operator =(const Matrix& other)
    {
      resize(other.rows(), other.cols());
      if (size() > 0)
        memcpy(&m_data[0], &other[0], size() * sizeof(double));
      return other;
    }

    bool operator ==(const Matrix& other)
    {
      return (m_rows == other.rows() && m_cols == other.cols()
          && memcmp(&m_data[0], &other[0], size() * sizeof(double)) == 0);
    }

    bool operator !=(const Matrix& other)
    {
      return (m_rows != other.rows() || m_cols != other.cols()
          || memcmp(&m_data[0], &other[0], size() * sizeof(double)) != 0);
    }

    Matrix operator +(const Matrix& other) const
    {
      Matrix ret(*this);
      ret += other;
      return ret;
    }

    Matrix operator -(const Matrix& other) const
    {
      Matrix ret(*this);
      ret -= other;
      return ret;
    }

    Matrix operator *(const Matrix& other) const
    {
      unsigned int x, xc = other.m_cols;
      unsigned int y, yc = m_rows;
      unsigned int k, kc = m_cols;
      ASSERT(other.m_rows == kc);
      Matrix ret(yc, xc);
      for (y = 0; y < yc; y++)
      {
        for (x = 0; x < xc; x++)
        {
          double v = 0.0;
          for (k = 0; k < kc; k++)
            v += operator ()(y, k) * other(k, x);
          ret(y, x) = v;
        }
      }
      return ret;
    }

    void operator +=(const Matrix& other)
    {
      ASSERT(m_rows == other.rows());
      ASSERT(m_cols == other.cols());
      unsigned int i, ic = size();
      for (i = 0; i < ic; i++)
        m_data[i] += other[i];
    }
    void operator -=(const Matrix& other)
    {
      ASSERT(m_rows == other.rows());
      ASSERT(m_cols == other.cols());
      unsigned int i, ic = size();
      for (i = 0; i < ic; i++)
        m_data[i] -= other[i];
    }

    void operator *=(double scalar)
    {
      unsigned int i, ic = size();
      for (i = 0; i < ic; i++)
        m_data[i] *= scalar;
    }

    void operator /=(double scalar)
    {
      unsigned int i, ic = size();
      for (i = 0; i < ic; i++)
        m_data[i] /= scalar; // TODO: check for stability
    }

    friend Matrix operator *(double scalar, const Matrix& mat)
    {
      Matrix ret(mat.rows(), mat.cols());
      unsigned int i, ic = mat.size();
      for (i = 0; i < ic; i++)
        ret[i] = scalar * mat[i];
      return ret;
    }

    Matrix operator /(double scalar) const
    {
      Matrix ret(*this);
      ret /= scalar;
      return ret;
    }

    static Matrix zeros(unsigned int rows, unsigned int cols = 1)
    {
      return Matrix(rows, cols);
    }

    static Matrix ones(unsigned int rows, unsigned int cols = 1)
    {
      Matrix m(rows, cols);
      unsigned int i, ic = rows * cols;
      for (i = 0; i < ic; i++)
        m[i] = 1.0;
      return m;
    }

    static Matrix eye(unsigned int n)
    {
      Matrix m(n, n);
      unsigned int i;
      for (i = 0; i < n; i++)
        m(i, i) = 1.0;
      return m;
    }

    static Matrix fromEig(const Matrix& U, const Matrix& lambda)
    {
      return U * lambda.diag() * U.T();
    }

    // eigen-decomposition of a symmetric matrix
    void eig(Matrix& U, Matrix& lambda, unsigned int iter = 200, bool ignoreError = true) const
    {
      ASSERT(isValid() && isSquare());
      Matrix basic = *this;
      Matrix eigenval(m_rows);

      // 1-dim case
      if (m_rows == 1)
      {
        basic(0, 0) = 1.0;
        eigenval(0) = m_data[0];
        return;
      }

      std::vector<double> oD(m_rows);
      unsigned int i, j, k, l, m;
      double b, c, f, g, h, hh, p, r, s, scale;

      // reduction to tridiagonal form
      for (i = m_rows; i-- > 1;)
      {
        h = 0.0;
        scale = 0.0;
        if (i > 1)
          for (k = 0; k < i; k++)
            scale += fabs(basic(i, k));

        if (scale == 0.0)
          oD[i] = basic(i, i - 1);
        else
        {
          for (k = 0; k < i; k++)
          {
            basic(i, k) /= scale;
            h += basic(i, k) * basic(i, k);
          }

          f = basic(i, i - 1);
          g = (f > 0.0) ? -::sqrt(h) : ::sqrt(h);
          oD[i] = scale * g;
          h -= f * g;
          basic(i, i - 1) = f - g;
          f = 0.0;

          for (j = 0; j < i; j++)
          {
            basic(j, i) = basic(i, j) / (scale * h);
            g = 0.0;
            for (k = 0; k <= j; k++)
              g += basic(j, k) * basic(i, k);
            for (k = j + 1; k < i; k++)
              g += basic(k, j) * basic(i, k);
            f += (oD[j] = g / h) * basic(i, j);
          }
          hh = f / (2.0 * h);

          for (j = 0; j < i; j++)
          {
            f = basic(i, j);
            g = oD[j] - hh * f;
            oD[j] = g;
            for (k = 0; k <= j; k++)
              basic(j, k) -= f * oD[k] + g * basic(i, k);
          }

          for (k = i; k--;)
            basic(i, k) *= scale;
        }
        eigenval(i) = h;
      }
      eigenval(0) = oD[0] = 0.0;

      // accumulation of transformation matrices
      for (i = 0; i < m_rows; i++)
      {
        if (eigenval(i) != 0.0)
        {
          for (j = 0; j < i; j++)
          {
            g = 0.0;
            for (k = 0; k < i; k++)
              g += basic(i, k) * basic(k, j);
            for (k = 0; k < i; k++)
              basic(k, j) -= g * basic(k, i);
          }
        }
        eigenval(i) = basic(i, i);
        basic(i, i) = 1.0;
        for (j = 0; j < i; j++)
          basic(i, j) = basic(j, i) = 0.0;
      }

      // eigenvalues from tridiagonal form
      for (i = 1; i < m_rows; i++)
        oD[i - 1] = oD[i];
      oD[m_rows - 1] = 0.0;

      for (l = 0; l < m_rows; l++)
      {
        j = 0;
        do
        {
          // look for small sub-diagonal element
          for (m = l; m < m_rows - 1; m++)
          {
            s = fabs(eigenval(m)) + fabs(eigenval(m + 1));
            if (fabs(oD[m]) + s == s)
              break;
          }
          p = eigenval(l);

          if (m != l)
          {
            if (j++ == iter)
            {
              // Too many iterations --> numerical instability!
              if (ignoreError)
                break;
              else
                throw("[Matrix::eig] numerical problems");
            }

            // form shift
            g = (eigenval(l + 1) - p) / (2.0 * oD[l]);
            r = ::sqrt(g * g + 1.0);
            g = eigenval(m) - p + oD[l] / (g + ((g > 0.0) ? fabs(r) : -fabs(r)));
            s = 1.0;
            c = 1.0;
            p = 0.0;

            for (i = m; i-- > l;)
            {
              f = s * oD[i];
              b = c * oD[i];
              if (fabs(f) >= fabs(g))
              {
                c = g / f;
                r = ::sqrt(c * c + 1.0);
                oD[i + 1] = f * r;
                s = 1.0 / r;
                c *= s;
              }
              else
              {
                s = f / g;
                r = ::sqrt(s * s + 1.0);
                oD[i + 1] = g * r;
                c = 1.0 / r;
                s *= c;
              }

              g = eigenval(i + 1) - p;
              r = (eigenval(i) - g) * s + 2.0 * c * b;
              p = s * r;
              eigenval(i + 1) = g + p;
              g = c * r - b;

              for (k = 0; k < m_rows; k++)
              {
                f = basic(k, i + 1);
                basic(k, i + 1) = s * basic(k, i) + c * f;
                basic(k, i) = c * basic(k, i) - s * f;
              }
            }

            eigenval(l) -= p;
            oD[l] = g;
            oD[m] = 0.0;
          }
        } while (m != l);
      }

      // normalize eigenvectors
      for (j = m_rows; j--;)
      {
        s = 0.0;
        for (i = m_rows; i--;)
          s += basic(i, j) * basic(i, j);
        s = ::sqrt(s);
        for (i = m_rows; i--;)
          basic(i, j) /= s;
      }

      // sort by eigenvalues
      std::vector<unsigned int> index(m_rows);
      for (i = 0; i < m_rows; i++)
        index[i] = i;
      CmpIndex cmpidx(eigenval, index);
      std::sort(index.begin(), index.end(), cmpidx);
      U.resize(m_rows, m_rows);
      lambda.resize(m_rows);
      for (i = 0; i < m_rows; i++)
      {
        j = index[i];
        lambda(i) = eigenval(j);
        for (k = 0; k < m_rows; k++)
          U(k, i) = basic(k, j);
      }
    }

    Matrix T() const
    {
      Matrix ret(m_cols, m_rows);
      unsigned int r, c;
      for (c = 0; c < m_cols; c++)
      {
        for (r = 0; r < m_rows; r++)
        {
          ret(c, r) = (*this)(r, c);
        }
      }
      return ret;
    }

    Matrix exp() const
    {
      ASSERT(isSquare());
      Matrix U;
      Matrix lambda;
      eig(U, lambda);
      unsigned int i;
      for (i = 0; i < m_rows; i++)
        lambda(i) = ::exp(lambda(i));
      Matrix ret = fromEig(U, lambda);
      return ret;
    }

    Matrix log() const
    {
      ASSERT(isSquare());
      Matrix U;
      Matrix lambda;
      eig(U, lambda);
      unsigned int i;
      for (i = 0; i < m_rows; i++)
        lambda(i) = ::log(lambda(i));
      return fromEig(U, lambda);
    }

    Matrix pow(double e) const
    {
      ASSERT(isSquare());
      Matrix U;
      Matrix lambda;
      eig(U, lambda);
      unsigned int i;
      for (i = 0; i < m_rows; i++)
        lambda(i) = ::pow(lambda(i), e);
      return fromEig(U, lambda);
    }

    inline Matrix power(double e) const
    {
      return pow(e);
    }
    inline Matrix sqrt() const
    {
      return pow(0.5);
    }
    inline Matrix inv() const
    {
      return pow(-1.0);
    }
    double det() const
    {
      Matrix U;
      Matrix lambda;
      eig(U, lambda);
      double ret = 1.0;
      unsigned int i, ic = lambda.size();
      for (i = 0; i < ic; i++)
        ret *= lambda(i);
      return ret;
    }

    double logdet() const
    {
      Matrix U;
      Matrix lambda;
      eig(U, lambda);
      double ret = 0.0;
      unsigned int i, ic = lambda.size();
      for (i = 0; i < ic; i++)
        ret += ::log(lambda(i));
      return ret;
    }

    double tr() const
    {
      ASSERT(isSquare());
      double ret = 0.0;
      unsigned int i;
      for (i = 0; i < m_rows; i++)
        ret += operator()(i, i);
      return ret;
    }

    double min() const
    {
      double ret = m_data[0];
      unsigned int i, ic = size();
      for (i = 1; i < ic; i++)
        if (m_data[i] < ret)
          ret = m_data[i];
      return ret;
    }

    double max() const
    {
      double ret = m_data[0];
      unsigned int i, ic = size();
      for (i = 1; i < ic; i++)
        if (m_data[i] > ret)
          ret = m_data[i];
      return ret;
    }

    double norm(double p = 2.0) const
    {
      unsigned int i, ic = size();
      double sum = 0.0;
      for (i = 0; i < ic; i++)
        sum += ::pow(fabs(m_data[i]), p);
      return ::pow(sum, 1.0 / p);
    }

    double onenorm() const
    {
      unsigned int i, ic = size();
      double sum = 0.0;
      for (i = 0; i < ic; i++)
        sum += fabs(m_data[i]);
      return sum;
    }

    double twonorm() const
    {
      unsigned int i, ic = size();
      double sum = 0.0;
      for (i = 0; i < ic; i++)
      {
        double v = m_data[i];
        sum += v * v;
      }
      return ::sqrt(sum);
    }

    double twonorm2() const
    {
      unsigned int i, ic = size();
      double sum = 0.0;
      for (i = 0; i < ic; i++)
      {
        double v = m_data[i];
        sum += v * v;
      }
      return sum;
    }

    double maxnorm() const
    {
      unsigned int i, ic = size();
      double m = fabs(m_data[0]);
      for (i = 1; i < ic; i++)
      {
        double v = fabs(m_data[i]);
        if (v > m)
          m = v;
      }
      return m;
    }

    Matrix diag() const
    {
      Matrix ret;
      unsigned int i;
      if (isSquare())
      {
        ret.resize(m_rows);
        for (i = 0; i < m_rows; i++)
          ret(i) = operator()(i, i);
      }
      else if (m_cols == 1)
      {
        ret.resize(m_rows, m_rows);
        for (i = 0; i < m_rows; i++)
          ret(i, i) = operator()(i);
      }
      else
        ASSERT(false);

      return ret;
    }

    friend std::ostream& operator<<(std::ostream& out, const Matrix& that);

    // A simple way to insert stuff
    // Make suer that the inputs are properly formatted
    void insert(double args, ...)
    {
      ASSERT(size() > 0);
      va_list vaargs;
      unsigned int i = 0;
      va_start(vaargs, args);
      for (unsigned y = 0; y < rows(); y++)
      {
        for (unsigned x = 0; x < cols(); x++)
        {
          if (y == 0 && x == 0)
            operator()(y, x) = args;
          else
            operator()(y, x) = va_arg(vaargs, double);
          ++i;
          ASSERT(i < size());
        }
      }
      va_end(vaargs);
    }

    Matrix()
    {
      resize(0, 0);
    }

    explicit Matrix(unsigned int rows, unsigned int cols = 1)
    {
      resize(rows, cols);
    }

    Matrix(const Matrix& other)
    {
      operator =(other);
    }

    Matrix(double* data, unsigned int rows, unsigned int cols = 1)
    {
      resize(rows, cols);
      if (size() > 0)
        memcpy(&m_data[0], data, size() * sizeof(double));
    }

    virtual ~Matrix()
    {
    }
};

std::ostream& operator<<(std::ostream& out, const Matrix& that)
{
  unsigned int r, c;
  out << that.m_rows << " x " << that.m_cols << " matrix at object address " << &that
      << " and memory address " << ((that.size() > 0) ? &that.m_data[0] : 0) << std::endl;
  for (r = 0; r < that.m_rows; r++)
  {
    for (c = 0; c < that.m_cols; c++)
    {
      out << that(r, c) << "\t";
    }
    out << std::endl;
  }
  return out;
}

}  // namespace RLLib

#endif /* MATRIX_H_ */
