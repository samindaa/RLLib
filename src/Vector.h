/*
 * Vector.h
 *
 *  Created on: Aug 18, 2012
 *      Author: sam
 */

#ifndef VECTOR_H_
#define VECTOR_H_

#include <algorithm>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <cstdio>

namespace RLLib
{

// Forward declarations
template<class T> class DenseVector;
template<class T> class SparseVector;
template<class T> std::ostream& operator<<(std::ostream& out,
    const DenseVector<T>& that);
template<class T> std::ostream& operator<<(std::ostream& out,
    const SparseVector<T>& that);

/**
 * This is used in parameter representation in a given vector space for
 * Machine Learning purposes. This implementation is specialized for sparse
 * vector representation, that is much common in Reinforcement Learning.
 */
template<class T>
class Vector
{
  public:
    virtual ~Vector()
    {
    }
    virtual int dimension() const =0;
    virtual double maxNorm() const =0;
    virtual double euclideanNorm() const =0;
    virtual T* operator()() const =0; // return the data as an array

    virtual void persist(const std::string& f) =0;
    virtual void resurrect(const std::string& f) =0;

  protected:
    template<class U> void write(std::ostream &o, U& value)
    {
      char *s = (char *) &value;
      o.write(s, sizeof(value));
    }

    template<class U> void read(std::istream &i, U& value)
    {
      char *s = (char *) &value;
      i.read(s, sizeof(value));
    }
};

template<class T>
class DenseVector: public Vector<T>
{
  protected:
    int capacity;
    T* data;

  public:
    DenseVector(const int& capacity = 1) :
        capacity(capacity), data(new T[capacity])
    {
      std::fill(data, data + capacity, 0);
    }

    virtual ~DenseVector()
    {
      delete[] data;
    }

    // Implementation details for copy constructor and operator
    DenseVector(const DenseVector<T>& that) :
        capacity(that.capacity), data(new T[that.capacity])
    {
      std::copy(that.data, that.data + that.capacity, data);
    }

    DenseVector<T>& operator=(const DenseVector<T>& that)
    {
      if (this != &that)
      {
        delete[] data; // delete old
        capacity = that.capacity;
        data = new T[capacity];
        std::copy(that.data, that.data + capacity, data);
      }
      return *this;
    }

    // dot product
    double operator*(const DenseVector<T>& that) const
    {
      assert(capacity == that.capacity);
      double tmp = 0;
      for (int i = 0; i < capacity; i++)
        tmp += data[i] * that.data[i];
      return tmp;
    }
    DenseVector<T>& operator*(const double& d)
    {
      for (T* i = data; i < data + capacity; ++i)
        *i *= d;
      return *this;
    }
    DenseVector<T>& operator+(const DenseVector<T>& that)
    {
      assert(capacity == that.capacity);
      for (int i = 0; i < capacity; i++)
        data[i] += that.data[i];
      return *this;
    }
    DenseVector<T>& operator-(const DenseVector<T>& that)
    {
      assert(capacity == that.capacity);
      for (int i = 0; i < capacity; i++)
        data[i] -= that.data[i];
      return *this;
    }

    T& operator[](const int& index) const
    {
      assert(index >= 0 && index < capacity);
      return data[index];
    }

    T& at(const int& index) const
    {
      return (*this)[index];
    }

    int dimension() const
    {
      return capacity;
    }
    double maxNorm() const
    {
      double maxv = capacity > 0 ?
          fabs(data[0]) : 0.0;
      if (capacity > 0)
      {
        for (int i = 1; i < capacity; i++)
        {
          if (fabs(data[i]) > maxv) maxv = fabs(data[i]);
        }
      }
      return maxv;
    }
    double euclideanNorm() const
    {
      return sqrt((*this) * (*this));
    }

    T* operator()() const
    {
      return data;
    }

    void clear()
    {
      std::fill(data, data + capacity, 0);
    }

    void set(const DenseVector<T>& that)
    {
      assert(capacity == that.capacity);
      for (int i = 0; i < capacity; i++)
        data[i] = that.data[i];
    }

    void persist(const std::string& f)
    {
      std::ofstream of;
      of.open(f.c_str(), std::ofstream::out);
      if (of.is_open())
      {
        // write vector type (int)
        int vectorType = 0;
        Vector<T>::write(of, vectorType);
        // write data size (int)
        Vector<T>::write(of, capacity);
        // write data
        for (int j = 0; j < capacity; j++)
          Vector<T>::write(of, data[j]);
        of.close();
        std::cout << "## DenseVector persisted=" << f << std::endl;
      }
      else std::cerr << "ERROR! (persist) file=" << f << std::endl;
    }

    void resurrect(const std::string& f)
    {
      std::ifstream ifs;
      ifs.open(f.c_str(), std::ifstream::in);
      if (ifs.is_open())
      {
        // Read vector type;
        int vectorType;
        Vector<T>::read(ifs, vectorType);
        assert(vectorType == 0);
        // Read capacity
        int rcapacity;
        Vector<T>::read(ifs, rcapacity);
        assert(capacity == rcapacity);
        printf("vectorType=%i rcapacity=%i \n", vectorType, rcapacity);
        // Read data
        for (int j = 0; j < capacity; j++)
          Vector<T>::read(ifs, data[j]);
        ifs.close();
        std::cout << "## DenseVector persist=" << f << std::endl;
      }
      else
      {
        std::cerr << "ERROR! (resurrected) file=" << f << std::endl;
        exit(-1);
      }
    }

    template<class O> friend std::ostream& operator<<(std::ostream& out,
        const DenseVector<O>& that);

};

/**
 * @use Tile coding; traces etc, where only a handful of features are
 *      active in out of billions of features; O(M) << O(N).
 */
template<class T>
class SparseVector: public Vector<T>
{
  protected:
    int indexesPositionLength;
    int activeLength;
    int numActive;

    int* indexesPosition;
    int* activeIndexes;
    T* values;
  public:

    SparseVector(const int& capacity = 1) :
        indexesPositionLength(capacity), activeLength(10), numActive(0),
            indexesPosition(new int[capacity]), activeIndexes(new int[10]),
            values(new T[10])
    {
      std::fill(indexesPosition, indexesPosition + capacity, -1);
    }

    virtual ~SparseVector()
    {
      delete[] indexesPosition;
      delete[] activeIndexes;
      delete[] values;
    }

    SparseVector(const SparseVector<T>& that) :
        indexesPositionLength(that.indexesPositionLength),
            activeLength(that.activeLength), numActive(that.numActive),
            indexesPosition(new int[that.indexesPositionLength]),
            activeIndexes(new int[that.activeLength]),
            values(new T[that.activeLength])
    {
      std::copy(that.indexesPosition,
          that.indexesPosition + that.indexesPositionLength, indexesPosition);
      std::copy(that.activeIndexes, that.activeIndexes + that.numActive,
          activeIndexes);
      std::copy(that.values, that.values + that.numActive, values);
    }
    SparseVector<T>& operator=(const SparseVector& that)
    {
      if (this != &that)
      {
        delete[] indexesPosition;
        delete[] activeIndexes;
        delete[] values;
        indexesPositionLength = that.indexesPositionLength;
        activeLength = that.activeLength;
        numActive = that.numActive;
        indexesPosition = new int[indexesPositionLength];
        activeIndexes = new int[activeLength];
        values = new T[activeLength];

        std::copy(that.indexesPosition,
            that.indexesPosition + that.indexesPositionLength, indexesPosition);
        std::copy(that.activeIndexes, that.activeIndexes + that.numActive,
            activeIndexes);
        std::copy(that.values, that.values + that.numActive, values);

      }
      return *this;
    }

    // bunch of helper methods
  private:
    void updateEntry(const int& index, const T& value, const int& position)
    {
      values[position] = value;
    }

    void swapEntry(const int& positionA, const int& positionB)
    {
      int indexA = activeIndexes[positionA];
      T valueA = values[positionA];
      int indexB = activeIndexes[positionB];
      T valueB = values[positionB];
      indexesPosition[indexA] = positionB;
      indexesPosition[indexB] = positionA;
      activeIndexes[positionA] = indexB;
      activeIndexes[positionB] = indexA;
      values[positionA] = valueB;
      values[positionB] = valueA;
    }

    void removeEntry(const int& position, const int& index)
    {
      swapEntry(numActive - 1, position);
      indexesPosition[activeIndexes[numActive - 1]] = -1;
      numActive--;
    }

  public:
    void removeEntry(const int& index)
    {
      int position = indexesPosition[index];
      if (position != -1) removeEntry(position, index);
    }

    void setEntry(const int& index, const T& value)
    {
      if (value == 0) removeEntry(index);
      else setNonZeroEntry(index, value);
    }

  private:
    void allocate(int sizeRequired)
    {
      if (activeLength >= sizeRequired) return;
      int newCapacity = (sizeRequired * 3) / 2 + 1;
      int* newActiveIndexes = new int[newCapacity];
      T* newValues = new T[newCapacity];

      std::copy(activeIndexes, activeIndexes + activeLength, newActiveIndexes);
      std::fill(newActiveIndexes + activeLength, newActiveIndexes + newCapacity,
          0);

      std::copy(values, values + activeLength, newValues);
      std::fill(newValues + activeLength, newValues + newCapacity, 0);

      activeLength = newCapacity;
      // remove old pointers
      delete[] activeIndexes;
      delete[] values;
      // set new pointers
      activeIndexes = newActiveIndexes;
      values = newValues;
    }

    void appendEntry(const int& index, const T& value)
    {
      allocate(numActive + 1);
      activeIndexes[numActive] = index;
      values[numActive] = value;
      indexesPosition[index] = numActive;
      numActive++;
    }

  public:
    void insertLast(const T& value)
    {
      appendEntry(indexesPositionLength - 1, value);
    }

    void insertEntry(const int& index, const T& value)
    {
      appendEntry(index, value);
    }

    const T getEntry(const int& index) const
    {
      int position = indexesPosition[index];
      return position != -1 ?
          values[position] : T(0);
    }

  private:
    void setNonZeroEntry(const int& index, const T& value)
    {
      int position = indexesPosition[index];
      if (position != -1) updateEntry(index, value, position);
      else insertEntry(index, value);
    }

  public:
    void clear()
    {
      for (int i = 0; i < numActive; i++)
        indexesPosition[activeIndexes[i]] = -1;
      numActive = 0;
    }

    SparseVector<T>& addToSelf(const double& factor,
        const SparseVector<T>& that)
    {
      for (int position = 0; position < that.numActive; position++)
      {
        int index = that.activeIndexes[position];
        setNonZeroEntry(index,
            getEntry(index) + factor * that.values[position]);
      }
      return *this;
    }

    SparseVector<T>& addToSelf(const double& factor,
        const SparseVector<T>& that, const int& offset)
    {
      for (int position = 0; position < that.numActive; position++)
      {
        int index = that.activeIndexes[position] + offset;
        setNonZeroEntry(index,
            getEntry(index) + factor * that.values[position]);
      }
      return *this;
    }

    SparseVector<T>& addToSelf(const SparseVector<T>& that)
    {
      return addToSelf(1, that);
    }

    SparseVector<T>& substractToSelf(const SparseVector<T>& that)
    {
      return addToSelf(-1.0, that);
    }

    SparseVector<T>& multiplyToSelf(const double& factor)
    {
      if (factor == 0)
      {
        clear();
        return *this;
      }

      for (T* position = values; position < values + numActive; ++position)
        *position *= factor;
      return *this;
    }

  private:
    double dot(const SparseVector<T>& _this, const SparseVector<T>& _that) const
    {
      double tmp = 0;
      for (int position = 0; position < _this.numActive; position++)
        tmp += _that.getEntry(_this.activeIndexes[position])
            * _this.values[position];
      return tmp;
    }

  public:
    // w'* phi
    double dot(const SparseVector<T>& that) const
    {
      assert(dimension() == that.dimension());
      if (numActive < that.numActive) return dot(*this, that);
      else return dot(that, *this);
    }

    // Shallow copy of that to this.
    void set(const SparseVector<T>& that)
    {
      clear();
      for (int i = 0; i < that.numActive; i++)
        insertEntry(that.activeIndexes[i], that.values[i]);
    }

    const T* getValues() const
    {
      return values;
    }

    const int* getActiveIndexes() const
    {
      return activeIndexes;
    }

    int numActiveEntries() const
    {
      return numActive;
    }

    int dimension() const
    {
      return indexesPositionLength;
    }
    double maxNorm() const
    {
      double maxv = numActive > 0 ?
          fabs(values[0]) : 0.0;
      if (numActive > 0)
      {
        for (int position = 1; position < numActive; position++)
        {
          if (fabs(values[position]) > maxv) maxv = fabs(values[position]);
        }
      }
      return maxv;
    }
    double euclideanNorm() const
    {
      return sqrt(dot(*this));
    }

    T* operator()() const
    {
      return values;
    }

    void persist(const std::string& f)
    {
      std::ofstream of;
      of.open(f.c_str(), std::ofstream::out);
      if (of.is_open())
      {
        // Write vector type (int)
        int vectorType = 1;
        Vector<T>::write(of, vectorType);
        // Write indexesPositionLength (int)
        Vector<T>::write(of, indexesPositionLength);
        // Write numActive (int)
        Vector<T>::write(of, numActive);
        // Write active indexes
        for (int position = 0; position < numActive; position++)
          Vector<T>::write(of, activeIndexes[position]);
        // Write active values
        for (int position = 0; position < numActive; position++)
          Vector<T>::write(of, values[position]);
        of.close();
        std::cout << "## SparseVector persisted=" << f << std::endl;
      }
      else std::cerr << "ERROR! (persist) file=" << f << std::endl;
    }

    void resurrect(const std::string& f)
    {
      std::ifstream ifs;
      ifs.open(f.c_str(), std::ifstream::in);
      if (ifs.is_open())
      {
        // Read vector type;
        int vectorType;
        Vector<T>::read(ifs, vectorType);
        assert(vectorType == 1);
        // Read indexesPositionLength
        int rcapacity;
        Vector<T>::read(ifs, rcapacity);
        // Read numActive
        int rnumActive;
        Vector<T>::read(ifs, rnumActive);
        assert(indexesPositionLength == rcapacity);
        printf("vectorType=%i rcapacity=%i rnumActive=%i\n", vectorType,
            rcapacity, rnumActive);
        // Read active indexes
        int* ractiveIndexes = new int[rnumActive];
        for (int position = 0; position < rnumActive; position++)
          Vector<T>::read(ifs, ractiveIndexes[position]);
        // Read active values
        for (int position = 0; position < rnumActive; position++)
        {
          T rvalue;
          Vector<T>::read(ifs, rvalue);
          insertEntry(ractiveIndexes[position], rvalue);
        }
        ifs.close();

        delete[] ractiveIndexes;
        std::cout << "## DenseVector persist=" << f << std::endl;
      }
      else
      {
        std::cerr << "ERROR! (resurrected) file=" << f << std::endl;
        exit(-1);
      }
    }

    template<class O> friend std::ostream& operator<<(std::ostream& out,
        const SparseVector<O>& that);
};

// Global implementations
template<class T>
std::ostream& operator<<(std::ostream& out, const DenseVector<T>& that)
{
  for (T* i = that.data; i < that.data + that.capacity; ++i)
    out << *i << " ";
  return out;
}

template<class T>
std::ostream& operator<<(std::ostream& out, const SparseVector<T>& that)
{
  out << "index=";
  for (int index = 0; index < that.indexesPositionLength; index++)
    out << that.indexesPosition[index] << " ";
  out << std::endl;

  for (int position = 0; position < that.numActive; position++)
    out << "[p=" << position << " i=" << that.activeIndexes[position] << " v="
        << that.getEntry(that.activeIndexes[position]) << "] ";
  return out;
}

} // namespace RLLib

#endif /* VECTOR_H_ */
