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
#include <sstream>
#include <vector>
#include <functional>
#include <numeric>
#include <cmath>
#include <cassert>
#include <cstdio>

namespace RLLib
{

/**
 * Forward declarations
 */
template<class T> class DenseVector;
template<class T> class SparseVector;
template<class T> std::ostream& operator<<(std::ostream& out, const DenseVector<T>& that);
template<class T> std::ostream& operator<<(std::ostream& out, const SparseVector<T>& that);

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

    // Some operations
    virtual int dimension() const =0;
    virtual bool empty() const =0;
    virtual T maxNorm() const =0;
    virtual T l1Norm() const =0;
    virtual T l2Norm() const =0;
    virtual T sum() const =0;

    // Return the data as an array
    virtual T* getValues() =0;
    virtual const T* getValues() const =0;

    // Get elements
    virtual T getEntry(const int& index) const =0;
    // Dot product
    virtual T dot(const Vector<T>* that) const =0;

    // Mutable Vector<T>
    virtual void clear() =0;
    // Insert T(value) within the Vector<T> capacity
    virtual void setEntry(const int& index, const T& value) =0;
    // This method only reset the value at index to T(0)
    virtual void removeEntry(const int& index) =0;
    virtual Vector<T>* addToSelf(const T& value) =0;
    virtual Vector<T>* addToSelf(const T& factor, const Vector<T>* that) =0;
    virtual Vector<T>* addToSelf(const Vector<T>* that) =0;
    virtual Vector<T>* subtractToSelf(const Vector<T>* that) =0;
    virtual Vector<T>* mapMultiplyToSelf(const T& factor) = 0;
    virtual Vector<T>* ebeMultiplyToSelf(const Vector<T>* that) =0;
    virtual Vector<T>* ebeDivideToSelf(const Vector<T>* that) =0;
    virtual Vector<T>* set(const Vector<T>* that) =0;
    virtual Vector<T>* set(const Vector<T>* that, const int& offset) =0;
    virtual Vector<T>* set(const T& value) =0;

    // This is a deep copy of Vector<T>.
    // The copied object exists even after the parent object is deleted.
    virtual Vector<T>* copy() const =0;
    virtual Vector<T>* newInstance(const int& dimension) const =0;
    // Storage management
    virtual void persist(const std::string& f) const =0;
    virtual void resurrect(const std::string& f) =0;

  protected:
    template<class U> void write(std::ostream &o, U& value) const
    {
      char *s = (char *) &value;
      o.write(s, sizeof(value));
    }

    template<class U> void read(std::istream &i, U& value) const
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

  public:
    int dimension() const
    {
      return capacity;
    }

    bool empty() const
    {
      return (dimension() == 0);
    }

    T maxNorm() const
    {
      T maxv = capacity > 0 ? fabs(data[0]) : 0.0;
      if (capacity > 0)
      {
        for (int i = 1; i < capacity; i++)
        {
          if (fabs(data[i]) > maxv)
            maxv = fabs(data[i]);
        }
      }
      return maxv;
    }

    T l1Norm() const
    {
      T result = T(0);
      for (int i = 0; i < capacity; i++)
        result += std::fabs(data[i]);
      return result;
    }

    T sum() const
    {
      return std::accumulate(data, data + capacity, T(0));
    }

    // Return the data as an array
    T* getValues()
    {
      return data;
    }

    const T* getValues() const
    {
      return data;
    }

    // Get elements
    T& operator[](const int& index)
    {
      assert(index >= 0 && index < capacity);
      return data[index];
    }

    const T& operator[](const int& index) const
    {
      assert(index >= 0 && index < capacity);
      return data[index];
    }

    T& at(const int& index)
    {
      return operator[](index);
    }

    const T& at(const int& index) const
    {
      return operator[](index);
    }

    T getEntry(const int& index) const
    {
      assert(index >= 0 && index < capacity);
      return data[index];
    }

    // Mutable Vector<T>
    void clear()
    {
      std::fill(data, data + capacity, 0);
    }

    void setEntry(const int& index, const T& value)
    {
      this->at(index) = value;
    }

    void removeEntry(const int& index)
    {
      this->at(index) = T(0);
    }

    Vector<T>* addToSelf(const T& value)
    {
      for (int i = 0; i < capacity; i++)
        data[i] += value;
      return this;
    }

    Vector<T>* mapMultiplyToSelf(const T& d)
    {
      for (T* i = data; i < data + capacity; ++i)
        *i *= d;
      return this;
    }

    Vector<T>* ebeMultiplyToSelf(const Vector<T>* that)
    {
      assert(this->dimension() == that->dimension());
      for (int i = 0; i < this->dimension(); i++)
        data[i] *= that->getEntry(i);
      return this;
    }

    Vector<T>* ebeDivideToSelf(const Vector<T>* that)
    {
      assert(this->dimension() == that->dimension());
      for (int i = 0; i < this->dimension(); i++)
      {
        const T& thatValue = that->getEntry(i);
        if (thatValue != 0)
          data[i] /= thatValue;
      }
      return this;
    }

    Vector<T>* set(const T& value)
    {
      std::fill(data, data + capacity, value);
      return this;
    }

    void persist(const std::string& f) const
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
        std::cout << "## DenseVector (sum=" << sum() << ", l1Norm=" << l1Norm() << ", maxNorm="
            << maxNorm() << ") persisted=" << f;
        std::cout << std::endl;
      }
      else
        std::cerr << "ERROR! (persist) file=" << f << std::endl;
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
        clear();
        printf("vectorType=%i rcapacity=%i \n", vectorType, rcapacity);
        // Read data
        for (int j = 0; j < capacity; j++)
          Vector<T>::read(ifs, data[j]);
        ifs.close();
        std::cout << "## DenseVector (sum=" << sum() << ", l1Norm=" << l1Norm() << ", maxNorm="
            << maxNorm() << ") resurrected=" << f;
        std::cout << std::endl;
      }
      else
      {
        std::cerr << "ERROR! (resurrect) file=" << f << std::endl;
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
    int activeIndexesLength;
    int nbActive;

    int* indexesPosition;
    int* activeIndexes;
    T* values;
  public:

    SparseVector(const int& capacity = 1, const int& activeIndexesLength = 10) :
        indexesPositionLength(capacity), activeIndexesLength(activeIndexesLength), nbActive(0), indexesPosition(
            new int[indexesPositionLength]), activeIndexes(new int[activeIndexesLength]), values(
            new T[activeIndexesLength])
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
        indexesPositionLength(that.indexesPositionLength), activeIndexesLength(
            that.activeIndexesLength), nbActive(that.nbActive), indexesPosition(
            new int[that.indexesPositionLength]), activeIndexes(new int[that.activeIndexesLength]), values(
            new T[that.activeIndexesLength])
    {
      std::copy(that.indexesPosition, that.indexesPosition + that.indexesPositionLength,
          indexesPosition);
      std::copy(that.activeIndexes, that.activeIndexes + that.nbActive, activeIndexes);
      std::copy(that.values, that.values + that.nbActive, values);
    }
    SparseVector<T>& operator=(const SparseVector& that)
    {
      if (this != &that)
      {
        delete[] indexesPosition;
        delete[] activeIndexes;
        delete[] values;
        indexesPositionLength = that.indexesPositionLength;
        activeIndexesLength = that.activeIndexesLength;
        nbActive = that.nbActive;
        indexesPosition = new int[indexesPositionLength];
        activeIndexes = new int[activeIndexesLength];
        values = new T[activeIndexesLength];

        std::copy(that.indexesPosition, that.indexesPosition + that.indexesPositionLength,
            indexesPosition);
        std::copy(that.activeIndexes, that.activeIndexes + that.nbActive, activeIndexes);
        std::copy(that.values, that.values + that.nbActive, values);

      }
      return *this;
    }

    // bunch of helper methods
  protected:
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
      swapEntry(nbActive - 1, position);
      indexesPosition[activeIndexes[nbActive - 1]] = -1;
      nbActive--;
    }

  public:
    void removeEntry(const int& index)
    {
      int position = indexesPosition[index];
      if (position != -1)
        removeEntry(position, index);
    }

    void setEntry(const int& index, const T& value)
    {
      if (value == 0)
        removeEntry(index);
      else
        setNonZeroEntry(index, value);
    }

  private:
    void allocate(int sizeRequired)
    {
      if (activeIndexesLength >= sizeRequired)
        return;
      int newCapacity = (sizeRequired * 3) / 2 + 1;
      int* newActiveIndexes = new int[newCapacity];
      T* newValues = new T[newCapacity];

      std::copy(activeIndexes, activeIndexes + activeIndexesLength, newActiveIndexes);
      std::fill(newActiveIndexes + activeIndexesLength, newActiveIndexes + newCapacity, 0);

      std::copy(values, values + activeIndexesLength, newValues);
      std::fill(newValues + activeIndexesLength, newValues + newCapacity, 0);

      activeIndexesLength = newCapacity;
      // remove old pointers
      delete[] activeIndexes;
      delete[] values;
      // set new pointers
      activeIndexes = newActiveIndexes;
      values = newValues;
    }

    void appendEntry(const int& index, const T& value)
    {
      allocate(nbActive + 1);
      activeIndexes[nbActive] = index;
      values[nbActive] = value;
      indexesPosition[index] = nbActive;
      nbActive++;
    }

  public:
    void insertEntry(const int& index, const T& value)
    {
      appendEntry(index, value);
    }

    T getEntry(const int& index) const
    {
      int position = indexesPosition[index];
      return position != -1 ? values[position] : T(0);
    }

  protected:
    void setNonZeroEntry(const int& index, const T& value)
    {
      int position = indexesPosition[index];
      if (position != -1)
        updateEntry(index, value, position);
      else
        insertEntry(index, value);
    }

  public:
    void clear()
    {
      for (int i = 0; i < nbActive; i++)
        indexesPosition[activeIndexes[i]] = -1;
      nbActive = 0;
    }

    T sum() const
    {
      return std::accumulate(values, values + nbActive, T(0));
    }

    const int* nonZeroIndexes() const
    {
      return activeIndexes;
    }

    const int* getIndexesPosition() const
    {
      return indexesPosition;
    }

    int nonZeroElements() const
    {
      return nbActive;
    }

    int dimension() const
    {
      return indexesPositionLength;
    }

    bool empty() const
    {
      return (dimension() == 0);
    }

    T maxNorm() const
    {
      T maxv = nbActive > 0 ? fabs(values[0]) : T(0);
      if (nbActive > 0)
      {
        for (int position = 1; position < nbActive; position++)
        {
          if (fabs(values[position]) > maxv)
            maxv = fabs(values[position]);
        }
      }
      return maxv;
    }

    T l1Norm() const
    {
      T result = T(0);
      for (int position = 0; position < nbActive; position++)
        result += fabs(values[position]);
      return result;
    }

    T* getValues()
    {
      return values;
    }

    const T* getValues() const
    {
      return values;
    }

    T dotData(const T* data) const
    {
      T result = T(0);
      for (int position = 0; position < nbActive; position++)
        result += data[activeIndexes[position]] * values[position];
      return result;
    }

    void addToData(const T& factor, T* data) const
    {
      for (int position = 0; position < nbActive; position++)
        data[activeIndexes[position]] += factor * values[position];
    }

    void subtractToData(T* data) const
    {
      for (int position = 0; position < nbActive; position++)
        data[activeIndexes[position]] -= values[position];
    }

    void persist(const std::string& f) const
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
        Vector<T>::write(of, nbActive);
        // Verbose
        printf("vectorType=%i capacity=%i nbActive=%i\n", vectorType, indexesPositionLength,
            nbActive);
        // Write active indexes
        for (int position = 0; position < nbActive; position++)
          Vector<T>::write(of, activeIndexes[position]);
        // Write active values
        for (int position = 0; position < nbActive; position++)
          Vector<T>::write(of, values[position]);
        of.close();
        std::cout << "## SparseVector (sum=" << sum() << ", l1Norm=" << l1Norm() << ", maxNorm="
            << maxNorm() << ") persisted=" << f;
        std::cout << std::endl;
      }
      else
        std::cerr << "ERROR! (persist) file=" << f << std::endl;
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
        int rnbActive;
        Vector<T>::read(ifs, rnbActive);
        assert(indexesPositionLength == rcapacity);
        clear();
        // Verbose
        printf("vectorType=%i rcapacity=%i rnbActive=%i\n", vectorType, rcapacity, rnbActive);
        // Read active indexes
        int* ractiveIndexes = new int[rnbActive];
        for (int position = 0; position < rnbActive; position++)
          Vector<T>::read(ifs, ractiveIndexes[position]);
        // Read active values
        for (int position = 0; position < rnbActive; position++)
        {
          T rvalue;
          Vector<T>::read(ifs, rvalue);
          insertEntry(ractiveIndexes[position], rvalue);
        }
        ifs.close();

        delete[] ractiveIndexes;
        std::cout << "## SparseVector (sum=" << sum() << ", l1Norm=" << l1Norm() << ", maxNorm"
            << maxNorm() << ") resurrected=" << f;
        std::cout << std::endl;
      }
      else
      {
        std::cerr << "ERROR! (resurrect) file=" << f << std::endl;
        exit(-1);
      }
    }

    template<class O> friend std::ostream& operator<<(std::ostream& out,
        const SparseVector<O>& that);

};

template<class T>
class PVector: public DenseVector<T>
{
  private:
    typedef DenseVector<T> Base;
  public:
    PVector(const int& capacity = 1) :
        DenseVector<T>(capacity)
    {
    }

    PVector(const DenseVector<T>* that) :
        DenseVector<T>(*that)
    {
    }

    PVector(const PVector<T>& that) :
        DenseVector<T>(that)
    {
    }

    PVector<T>& operator=(const PVector<T>& that)
    {
      if (this != &that)
        Base::operator =(that);
      return *this;
    }

    virtual ~PVector()
    {
    }

    PVector<T>& operator*(const T& d)
    {
      for (T* i = Base::data; i < Base::data + this->dimension(); ++i)
        *i *= d;
      return *this;
    }

    // Dot product
    T dot(const Vector<T>* that) const
    {
      assert(this->dimension() == that->dimension());

      const SparseVector<T>* other = dynamic_cast<const SparseVector<T>*>(that);
      if (other)
        return other->dotData(this->getValues());

      T result = T(0);
      for (int i = 0; i < this->dimension(); i++)
        result += Base::data[i] * that->getEntry(i);
      return result;
    }

    PVector<T>& operator-(const Vector<T>* that)
    {
      assert(this->dimension() == that->dimension());

      const SparseVector<T>* other = dynamic_cast<const SparseVector<T>*>(that);
      if (other)
      {
        other->subtractToSelfT(this->getValues());
        return *this;
      }

      for (int i = 0; i < this->dimension(); i++)
        Base::data[i] -= that->getEntry(i);
      return *this;
    }

    PVector<T>& operator+(const Vector<T>* that)
    {
      assert(this->dimension() == that->dimension());

      const SparseVector<T>* other = dynamic_cast<const SparseVector<T>*>(that);
      if (other)
      {
        other->addToSelfT(this->getValues());
        return *this;
      }

      for (int i = 0; i < this->dimension(); i++)
        Base::data[i] += that->at(i);
      return *this;
    }

    PVector<T>& operator/(const Vector<T>* that)
    {
      assert(this->dimension() == that->dimension());
      for (int i = 0; i < this->dimension(); i++)
      {
        const T& thatValue = that->getEntry(i);
        if (thatValue != 0)
          Base::data[i] /= thatValue;
      }
      return *this;
    }

    Vector<T>* addToSelf(const T& factor, const Vector<T>* that)
    {
      assert(this->dimension() == that->dimension());

      const SparseVector<T>* other = dynamic_cast<const SparseVector<T>*>(that);
      if (other)
      {
        other->addToData(factor, this->getValues());
        return this;
      }

      for (int i = 0; i < this->dimension(); i++)
        Base::data[i] += factor * that->getEntry(i);
      return this;
    }

    Vector<T>* addToSelf(const Vector<T>* that)
    {
      return addToSelf(1.0f, that);
    }

    Vector<T>* subtractToSelf(const Vector<T>* that)
    {
      assert(this->dimension() == that->dimension());

      const SparseVector<T>* other = dynamic_cast<const SparseVector<T>*>(that);
      if (other)
      {
        other->subtractToData(this->getValues());
        return this;
      }

      for (int i = 0; i < this->dimension(); i++)
        Base::data[i] -= that->getEntry(i);
      return this;
    }

    Vector<T>* set(const Vector<T>* that, const int& offset)
    { // FixMe:
      //assert(this->dimension() == that->dimension());

      const DenseVector<T>* other = dynamic_cast<const DenseVector<T>*>(that);
      if (other)
      {
        std::copy(other->getValues() + offset, other->getValues() + this->dimension(),
            this->getValues());
        return this;
      }
      // FixMe: This is very expensive for SVector<T>
      for (int i = 0; i < that->dimension(); i++)
        Base::data[i] = that->getEntry(i);
      return this;
    }

    Vector<T>* set(const Vector<T>* that)
    {
      assert(this->dimension() == that->dimension());
      return set(that, 0);
    }

    T l2Norm() const
    {
      return sqrt(this->dot(this));
    }

    Vector<T>* copy() const
    {
      return new PVector<T>(*this);
    }

    Vector<T>* newInstance(const int& dimension) const
    {
      return new PVector<T>(dimension);
    }
};

// ================================================================================================
template<class T>
class SVector: public SparseVector<T>
{
  private:
    typedef SparseVector<T> Base;
  public:
    SVector(const int& capacity = 1, const int& activeIndexesLength = 10) :
        SparseVector<T>(capacity, activeIndexesLength)
    {
    }

    virtual ~SVector()
    {
    }

    SVector(const SparseVector<T>* that) :
        SparseVector<T>(*that)
    {
    }

    SVector(const SVector<T>& that) :
        SparseVector<T>(that)
    {
    }

    SVector<T>& operator=(const SVector& that)
    {
      if (this != &that)
        Base::operator =(that);
      return *this;
    }

    T dot(const Vector<T>* that) const
    {
      const SparseVector<T>* other = dynamic_cast<const SparseVector<T>*>(that);
      if (other && other->nonZeroElements() < this->nonZeroElements())
        return other->dot(this);

      T result = T(0);
      for (int position = 0; position < Base::nbActive; position++)
        result += that->getEntry(Base::activeIndexes[position]) * Base::values[position];
      return result;
    }

    Vector<T>* addToSelf(const T& value)
    {
      for (int index = 0; index < Base::indexesPositionLength; index++)
        this->setNonZeroEntry(index, value + this->getEntry(index));
      return this;
    }

    Vector<T>* addToSelf(const T& factor, const Vector<T>* that)
    {
      const SparseVector<T>* other = dynamic_cast<const SparseVector<T>*>(that);
      if (other)
      {
        for (int position = 0; position < other->nonZeroElements(); position++)
        {
          const int index = other->nonZeroIndexes()[position];
          this->setNonZeroEntry(index,
              this->getEntry(index) + factor * other->getValues()[position]);
        }
        return this;
      }

      for (int i = 0; i < that->dimension(); i++)
        this->setEntry(i, this->getEntry(i) + factor * that->getEntry(i));
      return this;
    }

    Vector<T>* addToSelf(const Vector<T>* that)
    {
      return addToSelf(1.0f, that);
    }

    Vector<T>* subtractToSelf(const Vector<T>* that)
    {
      return addToSelf(-1.0f, that);
    }

    Vector<T>* mapMultiplyToSelf(const T& factor)
    {
      if (factor == 0)
      {
        this->clear();
        return this;
      }

      for (T* position = Base::values; position < Base::values + Base::nbActive; ++position)
        *position *= factor;
      return this;
    }

    Vector<T>* ebeMultiplyToSelf(const Vector<T>* that)
    {
      assert(this->dimension() == that->dimension());
      int position = 0;
      while (position < Base::nbActive)
      {
        int index = Base::activeIndexes[position];
        T value = Base::values[position] * that->getEntry(index);
        if (value != 0)
        {
          Base::values[position] = value;
          ++position;
        }
        else
          this->removeEntry(position, index);
      }
      return this;
    }

    Vector<T>* ebeDivideToSelf(const Vector<T>* that)
    {
      for (int position = 0; position < Base::nbActive; position++)
      {
        int index = Base::activeIndexes[position];
        Base::values[position] /= that->getEntry(index); // prior check
      }
      return this;
    }

    Vector<T>* set(const Vector<T>* that)
    {
      assert(this->dimension() == that->dimension());
      this->clear();
      const SparseVector<T>* other = dynamic_cast<const SparseVector<T>*>(that);
      if (other)
      {
        for (int i = 0; i < other->nonZeroElements(); i++)
          this->setNonZeroEntry(other->nonZeroIndexes()[i], other->getValues()[i]);
        return this;
      }

      for (int i = 0; i < that->dimension(); i++)
        this->setEntry(i, that->getEntry(i));
      return this;
    }

    Vector<T>* set(const Vector<T>* that, const int& start)
    {
      // Dimension check is relaxed.
      this->clear();
      const SparseVector<T>* other = dynamic_cast<const SparseVector<T>*>(that);
      if (other)
      {
        for (int i = 0; i < other->nonZeroElements(); i++)
          this->setNonZeroEntry(other->nonZeroIndexes()[i] + start, other->getValues()[i]);
        return this;
      }

      for (int i = 0; i < that->dimension(); i++)
        this->setEntry(start + i, that->getEntry(i));
      return this;
    }

    Vector<T>* set(const T& value)
    {
      // This will set 'value' to all the elements
      for (int index = 0; index < Base::indexesPositionLength; index++)
        this->setNonZeroEntry(index, value);
      return this;
    }

    T l2Norm() const
    {
      return sqrt(this->dot(this));
    }

    Vector<T>* copy() const
    {
      return new SVector<T>(*this);
    }

    Vector<T>* newInstance(const int& dimension) const
    {
      return new SVector<T>(dimension);
    }

};

// ================================================================================================
template<class T>
class Vectors
{
  protected:
    typename std::vector<Vector<T>*> vectors;
  public:
    typedef typename std::vector<Vector<T>*>::iterator iterator;
    typedef typename std::vector<Vector<T>*>::const_iterator const_iterator;

    Vectors()
    {
    }

    ~Vectors()
    {
      vectors.clear();
    }

    Vectors(const Vectors<T>& that)
    {
      for (typename Vectors<T>::iterator iter = that.begin(); iter != that.end(); ++iter)
        vectors.push_back(*iter);
    }

    Vectors<T>& operator=(const Vectors<T>& that)
    {
      if (this != that)
      {
        vectors.clear();
        for (typename Vectors<T>::iterator iter = that.begin(); iter != that.end(); ++iter)
          vectors.push_back(*iter);
      }
      return *this;
    }

    void push_back(Vector<T>* vector)
    {
      vectors.push_back(vector);
    }

    iterator begin()
    {
      return vectors.begin();
    }

    const_iterator begin() const
    {
      return vectors.begin();
    }

    iterator end()
    {
      return vectors.end();
    }

    const_iterator end() const
    {
      return vectors.end();
    }

    void clear()
    {
      for (typename Vectors<T>::iterator iter = begin(); iter != end(); ++iter)
        (*iter)->clear();
    }

    int dimension() const
    {
      return vectors.size();
    }

    Vector<T>* operator[](const int& index)
    {
      return vectors.at(index);
    }

    const Vector<T>* operator[](const int& index) const
    {
      return vectors.at(index);
    }

    Vector<T>* at(const int& index)
    {
      return vectors.at(index);
    }

    const Vector<T>* at(const int& index) const
    {
      return vectors.at(index);
    }

    void persist(std::string f) const
    {
      int i = 0;
      for (typename Vectors<T>::const_iterator iter = begin(); iter != end(); ++iter)
      {
        std::string fi(f);
        std::stringstream ss;
        ss << "." << i;
        fi.append(ss.str());
        (*iter)->persist(fi);
        ++i;
      }
    }

    void resurrect(std::string f) const
    {
      int i = 0;
      for (typename Vectors<T>::const_iterator iter = begin(); iter != end(); ++iter)
      {
        std::string fi(f);
        std::stringstream ss;
        ss << "." << i;
        fi.append(ss.str());
        (*iter)->resurrect(fi);
        ++i;
      }
    }

    // Static
    inline static Vector<T>* absToSelf(Vector<T>* other)
    {
      SparseVector<T>* that = dynamic_cast<SparseVector<T>*>(other);
      if (that)
      {
        T* values = that->getValues();
        for (T* position = values; position < values + that->nonZeroElements(); ++position)
          *position = fabs(*position);
      }
      else
      {
        T* values = other->getValues();
        for (T* position = values; position < values + other->dimension(); ++position)
          *position = fabs(*position);
      }
      return other;
    }

    inline static void positiveMaxToSelf(Vector<T>* result, const Vector<T>* that)
    {
      const SparseVector<T>* other = dynamic_cast<const SparseVector<T>*>(that);
      if (other)
      {
        const int* activeIndexes = other->nonZeroIndexes();
        for (int i = 0; i < other->nonZeroElements(); i++)
        {
          int index = activeIndexes[i];
          result->setEntry(index, std::max(result->getEntry(index), other->getEntry(index)));
        }
      }
      else
      {
        for (int index = 0; index < that->dimension(); index++)
          result->setEntry(index, std::max(result->getEntry(index), that->getEntry(index)));
      }
    }

    inline static void expToSelf(Vector<T>* result, const Vector<T>* that)
    {
      const SparseVector<T>* other = dynamic_cast<const SparseVector<T>*>(that);
      if (other)
      {
        const int* activeIndexes = other->nonZeroIndexes();
        for (int i = 0; i < other->nonZeroElements(); i++)
        {
          int index = activeIndexes[i];
          result->setEntry(index, std::exp(other->getEntry(index)));
        }
      }
      else
      {
        for (int index = 0; index < that->dimension(); index++)
          result->setEntry(index, std::exp(that->getEntry(index)));
      }
    }

    inline static void multiplySelfByExponential(SparseVector<T>* result, const T& factor,
        const SparseVector<T>* other, const T& min)
    {
      const int* activeIndexes = other->nonZeroIndexes();
      for (int i = 0; i < other->nonZeroElements(); i++)
      {
        int index = activeIndexes[i];
        result->setEntry(index,
            std::max(min, result->getEntry(index) * std::exp(factor * other->getEntry(index))));
      }
    }

    static void multiplySelfByExponential(DenseVector<T>* result, const T& factor,
        const SparseVector<T>* other, const T& min)
    {
      const int* activeIndexes = other->nonZeroIndexes();
      T* resultValues = result->getValues();
      const T* otherValues = other->getValues();
      for (int i = 0; i < other->nonZeroElements(); i++)
      {
        int index = activeIndexes[i];
        resultValues[index] = std::max(min,
            resultValues[index] * std::exp(factor * otherValues[i]));
      }
    }

    static void multiplySelfByExponential(DenseVector<T>* result, const T& factor,
        const Vector<T>* other, const T& min)
    {
      const SparseVector<T>* that = dynamic_cast<const SparseVector<T>*>(other);
      if (that)
        multiplySelfByExponential(result, factor, that, min);
      else
      {
        T* resultValues = result->getValues();
        for (int i = 0; i < result->dimension(); i++)
          resultValues[i] = std::max(min, resultValues[i] * std::exp(factor * other->getEntry(i)));
      }
    }

    static void multiplySelfByExponential(Vector<T>* result, const T& factor,
        const Vector<T>* other, const T& min)
    {
      SparseVector<T>* sresult = dynamic_cast<SparseVector<T>*>(result);
      if (sresult)
        multiplySelfByExponential(sresult, factor, other, min);
      else
      {
        DenseVector<T>* dresult = dynamic_cast<DenseVector<T>*>(result);
        multiplySelfByExponential(dresult, factor, other, min);
      }
    }

    static void multiplySelfByExponential(DenseVector<T>* result, const T& factor,
        const Vector<T>* other)
    {
      multiplySelfByExponential(result, factor, other, 0);
    }

    static bool isNull(const Vector<T>* v)
    {
      if (!v)
        return true;
      const SparseVector<T>* that = dynamic_cast<const SparseVector<T>*>(v);
      if (that)
        return that->nonZeroElements() == 0;
      const T* values = v->getValues();
      for (int i = 0; i < v->dimension(); i++)
      {
        if (values[i] != 0)
          return false;
      }
      return true;
    }

    static Vector<T>* bufferedCopy(const Vector<T>* source, Vector<T>*& target)
    {
      Vector<T>* result = target ? target : source->copy();
      result->set(source);
      target = result;
      return target;
    }

    static Vector<T>* toBinary(Vector<T>* result, const Vector<T>* v)
    {
      assert(result->dimension() == v->dimension());
      result->clear();
      const SparseVector<T>* sv = dynamic_cast<const SparseVector<T>*>(v);
      if (sv)
      {
        for (int i = 0; i < sv->nonZeroElements(); i++)
          result->setEntry(sv->nonZeroIndexes()[i], 1.0f);
        return result;
      }

      for (int i = 0; i < v->dimension(); i++)
      {
        if (v->getValues()[i] != 0)
          result->setEntry(i, 1.0f);
      }
      return result;
    }
};

template<class T>
class Filters
{
  public:
    static Vector<T>* mapMultiplyToSelf(Vector<T>* result, const T& d, const Vector<T>* filter)
    {
      const SparseVector<T>* sfilter = dynamic_cast<const SparseVector<T>*>(filter);
      if (sfilter)
      {
        for (int i = 0; i < sfilter->nonZeroElements(); i++)
        {
          const int index = sfilter->nonZeroIndexes()[i];
          result->setEntry(index, result->getEntry(index) * d);
        }
        return result;
      }
      return result->mapMultiplyToSelf(d);
    }
};

template<class T>
class VectorPool
{
  protected:
    std::vector<Vector<T>*>* stackedVectors;
    int nbAllocation;
    int dimension;
  public:
    VectorPool(const int& dimension) :
        stackedVectors(new std::vector<Vector<T>*>()), nbAllocation(0), dimension(dimension)
    {
    }

    ~VectorPool()
    {
      for (typename std::vector<Vector<T>*>::iterator iter = stackedVectors->begin();
          iter != stackedVectors->end(); ++iter)
        delete *iter;
      stackedVectors->clear();
      delete stackedVectors;
    }

  public:
    Vector<T>* newVector(const Vector<T>* v)
    {
      ++nbAllocation;
      if (nbAllocation > static_cast<int>(stackedVectors->size()))
        stackedVectors->push_back(v->newInstance(dimension));
      return stackedVectors->at(nbAllocation - 1)->set(v);
    }

    void releaseAll()
    {
      nbAllocation = 0;
    }
};

// Global implementations
template<class T>
std::ostream& operator<<(std::ostream& out, const DenseVector<T>& that)
{
  out << "DenseVector ";
  for (T* i = that.data; i < that.data + that.capacity; ++i)
    out << *i << " ";
  return out;
}

template<class T>
std::ostream& operator<<(std::ostream& out, const SparseVector<T>& that)
{
  out << "SparseVector(" << that.nbActive << ") index=";
  for (int index = 0; index < that.indexesPositionLength; index++)
    out << that.indexesPosition[index] << " ";
  out << std::endl;

  for (int position = 0; position < that.nbActive; position++)
    out << "[p=" << position << " i=" << that.activeIndexes[position] << " v="
        << that.getEntry(that.activeIndexes[position]) << "] ";
  return out;
}

template<class T>
static void printVector(const Vector<T>* other)
{
  const SparseVector<T>* that = dynamic_cast<const SparseVector<T>*>(other);
  if (that)
  {
    std::cout << "SparseVector(" << that->nonZeroElements() << ") index=";
    //for (int index = 0; index < that->dimension(); index++)
    //  std::cout << that->getIndexesPosition()[index] << " ";
    //std::cout << std::endl;

    for (int position = 0; position < that->nonZeroElements(); position++)
      std::cout << "[p=" << position << " i=" << that->nonZeroIndexes()[position] << " v="
          << that->getEntry(that->nonZeroIndexes()[position]) << "] ";
    std::cout << std::endl;
  }

  const DenseVector<T>* theOther = dynamic_cast<const DenseVector<T>*>(other);
  if (theOther)
  {
    std::cout << "DenseVector(" << theOther->dimension() << ") ";
    for (const T* i = theOther->getValues(); i < theOther->getValues() + theOther->dimension(); ++i)
      std::cout << *i << " ";
    std::cout << std::endl;
  }
}

} // namespace RLLib

#endif /* VECTOR_H_ */
