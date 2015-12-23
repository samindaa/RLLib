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
 * Trace.h
 *
 *  Created on: Aug 19, 2012
 *      Author: sam
 */

#ifndef TRACE_H_
#define TRACE_H_

#include "Mathema.h"
#include "Vector.h"

namespace RLLib
{

  template<typename T>
  class Trace
  {
    public:
      virtual ~Trace()
      {
      }
      virtual void update(const T& lambda, const Vector<T>* phi, const T& factor = T(1)) =0;
      virtual void clear() =0;
      virtual Vector<T>* vect() const =0;
  };

  template<typename T>
  class ATrace: public Trace<T>
  {
    protected:
      T defaultThreshold;
      T threshold;
      Vector<T>* vector;
    public:
      ATrace(const int& numFeatures, const T& threshold = T(1e-8)) :
          defaultThreshold(threshold), threshold(threshold), vector(new SVector<T>(numFeatures))
      {
      }

      virtual ~ATrace()
      {
        delete vector;
      }

    private:
      virtual void clearBelowThreshold()
      {
        SparseVector<T>* svector = RTTI<T>::sparseVector(vector);
        const T* values = svector->getValues();
        const int* indexes = svector->nonZeroIndexes();
        int i = 0;
        while (i < svector->nonZeroElements())
        {
          T absValue = std::abs(values[i]);
          if (absValue <= threshold)
            svector->removeEntry(indexes[i]);
          else
            i++;
        }
      }

    public:
      virtual void updateVector(const T& lambda, const Vector<T>* phi, const T& factor)
      {
        vector->mapMultiplyToSelf(lambda);
        vector->addToSelf(factor, phi);
      }

      void update(const T& lambda, const Vector<T>* phi, const T& factor = T(1))
      {
        updateVector(lambda, phi, factor);
        adjustUpdate();
        clearBelowThreshold();
      }

      void clear()
      {
        vector->clear();
        threshold = defaultThreshold;
      }

      Vector<T>* vect() const
      {
        return vector;
      }

    protected:
      virtual void adjustUpdate()
      { // Nothing to be adjusted.
      }

  };

  template<typename T>
  class RTrace: public ATrace<T>
  {
    public:
      RTrace(const int& capacity, const T& threshold = T(1e-8)) :
          ATrace<T>(capacity, threshold)
      {
      }

      virtual ~RTrace()
      {
      }

    private:
      void replaceWith(const Vector<T>* x, const T& factor)
      { // FixMe:
        const SparseVector<T>* phi = RTTI<T>::constSparseVector(x);
        const int* indexes = phi->nonZeroIndexes();
        for (const int* index = indexes; index < indexes + phi->nonZeroElements(); ++index)
          ATrace<T>::vector->setEntry(*index, factor * phi->getEntry(*index));
      }
    public:
      virtual void updateVector(const T& lambda, const Vector<T>* phi, const T& factor)
      {
        ATrace<T>::vector->mapMultiplyToSelf(lambda);
        replaceWith(phi, factor);
      }

  };

  template<typename T>
  class AMaxTrace: public ATrace<T>
  {
    protected:
      T maximumValue;
    public:
      AMaxTrace(const int& capacity, const T& threshold = T(1e-8), const T& maximumValue = T(1)) :
          ATrace<T>(capacity, threshold), maximumValue(maximumValue)
      {
      }

      virtual ~AMaxTrace()
      {
      }

    private:
      void adjustValue(T& value) const
      {
        if (std::abs(value) > maximumValue)
          value = Signum::valueOf(value) * maximumValue;
      }

      void adjustValues(T* data, const int& size) const
      {
        for (int i = 0; i < size; i++)
          adjustValue(data[i]);
      }

    public:

      virtual void adjustUpdate()
      {
        T* data = ATrace<T>::vector->getValues();
        SparseVector<T>* svector = RTTI<T>::sparseVector(ATrace<T>::vector);
        int size = 0;
        if (svector)
          size = svector->nonZeroElements();
        else
          size = ATrace<T>::vector->dimension();
        adjustValues(data, size);
      }
  };

  template<typename T>
  class MaxLengthTrace: public Trace<T>
  {
    protected:
      Trace<T>* trace;
      int maximumLength;

    public:
      MaxLengthTrace(Trace<T>* trace, const int& maximumLength) :
          trace(trace), maximumLength(maximumLength)
      {
        const SparseVector<T>* v = (const SparseVector<T>*) trace->vect();
#if !defined(EMBEDDED_MODE)
        if (!v)
          std::cerr << "MaxLengthTraces supports only traces SparseVector<T>" << std::endl;
#endif
      }

      virtual ~MaxLengthTrace()
      {
      }

    private:
      void controlLength()
      {
        const SparseVector<T>* v = (const SparseVector<T>*) trace->vect();
        if (v->nonZeroElements() < maximumLength)
          return;
        while (v->nonZeroElements() > maximumLength)
        {
          const T* values = v->getValues();
          const int* indexes = v->nonZeroIndexes();
          T minValue = values[0];
          int minIndex = indexes[0];
          for (int i = 1; i < v->nonZeroElements(); i++)
          {
            if (values[i] < minValue)
            {
              minValue = values[i];
              minIndex = indexes[i];
            }
          }
          trace->vect()->setEntry(minIndex, 0);
        }
      }

    public:

      void update(const T& lambda, const Vector<T>* phi, const T& factor = T(1))
      {
        trace->update(lambda, phi, factor);
        controlLength();
      }

      void clear()
      {
        trace->clear();
      }

      Vector<T>* vect() const
      {
        return trace->vect();
      }
  };

  template<typename T>
  class Traces
  {
    protected:
      typename std::vector<Trace<T>*> traces;
    public:
      typedef typename std::vector<Trace<T>*>::iterator iterator;
      typedef typename std::vector<Trace<T>*>::const_iterator const_iterator;

      Traces()
      {
      }

      ~Traces()
      {
        traces.clear();
      }

      Traces(const Traces<T>& that)
      {
        for (typename Traces<T>::iterator iter = that.begin(); iter != that.end(); ++iter)
          traces.push_back(*iter);
      }

      Traces<T>& operator=(const Traces<T>& that)
      {
        if (this != that)
        {
          traces.clear();
          for (typename Traces<T>::iterator iter = that.begin(); iter != that.end(); ++iter)
            traces.push_back(*iter);
        }
        return *this;
      }

      void push_back(Trace<T>* trace)
      {
        traces.push_back(trace);
      }

      iterator begin()
      {
        return traces.begin();
      }

      const_iterator begin() const
      {
        return traces.begin();
      }

      iterator end()
      {
        return traces.end();
      }

      const_iterator end() const
      {
        return traces.end();
      }

      int dimension() const
      {
        return traces.size();
      }

      Trace<T>* getEntry(const int& index)
      {
        return traces.at(index);
      }

      const Trace<T>* getEntry(const int& index) const
      {
        return traces.at(index);
      }

      void clear()
      {
        for (typename Traces<T>::iterator iter = begin(); iter != end(); ++iter)
          (*iter)->clear();
      }
  };

} // namespace RLLib

#endif /* TRACE_H_ */
