/*
 * Trace.h
 *
 *  Created on: Aug 19, 2012
 *      Author: sam
 */

#ifndef TRACE_H_
#define TRACE_H_
#include <iostream>
#include "Vector.h"
#include "Math.h"

namespace RLLib
{

template<class T>
class Trace
{
  public:
    virtual ~Trace()
    {
    }
    virtual void update(const double& lambda, const SparseVector<T>& phi) =0;
    virtual void multiplyToSelf(const double& factor) =0;
    virtual void clear() =0;
    virtual const SparseVector<T>& vect() const =0;

    virtual void updateThreshold() =0;
    virtual void controlThreshold() =0;
};

template<class T>
class ATrace: public Trace<T>
{
  protected:
    double defaultThreshold;
    double threshold;
    SparseVector<T>* vector;
  public:
    ATrace(const int& numFeatures, const double& threshold = 1e-8) :
        defaultThreshold(threshold), threshold(threshold), vector(
            new SparseVector<T>(numFeatures))
    {
    }
    virtual ~ATrace()
    {
      delete vector;
    }

  public:

    virtual void controlThreshold() //clearBelowThreshold
    {
      const T* values = vector->getValues();
      const int* indexes = vector->getActiveIndexes();
      int i = 0;
      while (i < vector->nbActiveEntries())
      {
        T absValue = fabs(values[i]);
        if (absValue <= threshold)
          vector->removeEntry(indexes[i]);
        else
          i++;
      }
    }

    virtual void updateVector(const double& lambda, const SparseVector<T>& phi)
    {
      vector->multiplyToSelf(lambda);
      vector->addToSelf(phi);
    }

    void update(const double& lambda, const SparseVector<T>& phi)
    {
      updateVector(lambda, phi);
      adjustUpdate();
      controlThreshold();
    }

    void multiplyToSelf(const double& factor)
    {
      vector->multiplyToSelf(factor);
    }

    void clear()
    {
      vector->clear();
      threshold = defaultThreshold;
    }
    const SparseVector<T>& vect() const
    {
      return *vector;
    }

    void updateThreshold()
    {
      threshold += 0.1 * threshold; // A small update
    }

  protected:
    virtual void adjustUpdate()
    { // Nothing to be adjusted.
    }

};

template<class T>
class RTrace: public ATrace<T>
{
  public:
    RTrace(const int& capacity, const double& threshold = 1e-8) :
        ATrace<T>(capacity, threshold)
    {
    }
    virtual ~RTrace()
    {
    }

  private:
    void replaceWith(const SparseVector<T>& phi)
    {
      const int* indexes = phi.getActiveIndexes();
      for (const int* index = indexes; index < indexes + phi.nbActiveEntries();
          ++index)
        ATrace<T>::vector->setEntry(*index, 1.0);
    }
  public:
    virtual void updateVector(const double& lambda, const SparseVector<T>& phi)
    {
      ATrace<T>::vector->multiplyToSelf(lambda);
      replaceWith(phi);
    }

};

template<class T>
class AMaxTrace: public ATrace<T>
{
  protected:
    double maximumValue;
  public:
    AMaxTrace(const int& capacity, const double& threshold = 1e-8,
        const double& maximumValue = 1.0) :
        ATrace<T>(capacity, threshold), maximumValue(maximumValue)
    {
    }

  public:

    void adjustUpdate()
    {
      const T* values = ATrace<T>::vector->getValues();
      const int* indexes = ATrace<T>::vector->getActiveIndexes();

      for (int i = 0; i < ATrace<T>::vector->nbActiveEntries(); i++)
      {
        T absValue = fabs(values[i]);
        if (absValue > maximumValue)
          ATrace<T>::vector->setEntry(indexes[i],
              Signum::valueOf(absValue) * maximumValue);
      }
    }

};

template<class T>
class MaxLengthTrace: public Trace<T>
{
  protected:
    Trace<T>* trace;
    int maximumLength;

  public:
    MaxLengthTrace(Trace<T>* trace, const int& maximumLength) :
        trace(trace), maximumLength(maximumLength)
    {
    }

    virtual ~MaxLengthTrace()
    {
    }

  public:

    void update(const double& lambda, const SparseVector<T>& phi)
    {
      trace->update(lambda, phi);
      controlThreshold();
    }
    void multiplyToSelf(const double& factor)
    {
      trace->multiplyToSelf(factor);
    }

    void clear()
    {
      trace->clear();
    }
    const SparseVector<T>& vect() const
    {
      return trace->vect();
    }

    void updateThreshold()
    {
      trace->updateThreshold();
    }
    void controlThreshold()
    {
      while (trace->vect().nbActiveEntries() > maximumLength)
      {
        updateThreshold();
        trace->controlThreshold();
        //std::cout << "@@>> cullingTraces " << ATrace<T>::threshold << std::endl;
      }
    }
};

template<class T>
class MultiTrace
{
  protected:
    typename std::vector<Trace<T>*>* traces;
  public:
    typedef typename std::vector<Trace<T>*>::iterator iterator;
    typedef typename std::vector<Trace<T>*>::const_iterator const_iterator;

    MultiTrace() :
        traces(new std::vector<Trace<T>*>())
    {
    }

    ~MultiTrace()
    {
      traces->clear();
      delete traces;
    }

    MultiTrace(const MultiTrace<T>& that) :
        traces(new std::vector<Trace<T>*>())
    {
      for (typename MultiTrace<T>::iterator iter = that.begin();
          iter != that.end(); ++iter)
        traces->push_back(*iter);
    }

    MultiTrace<T>& operator=(const MultiTrace<T>& that)
    {
      if (this != that)
      {
        traces->clear();
        delete traces;
        traces = new MultiTrace<T>();
        for (typename MultiTrace<T>::iterator iter = that.begin();
            iter != that.end(); ++iter)
          traces->push_back(*iter);
      }
      return *this;
    }

    void push_back(Trace<T>* trace)
    {
      traces->push_back(trace);
    }

    iterator begin()
    {
      return traces->begin();
    }

    const_iterator begin() const
    {
      return traces->begin();
    }

    iterator end()
    {
      return traces->end();
    }

    const_iterator end() const
    {
      return traces->end();
    }

    unsigned int dimension() const
    {
      return traces->size();
    }

    Trace<T>* at(const unsigned index) const
    {
      assert(index >= 0 && index < dimension());
      return traces->at(index);
    }

    void clear()
    {
      for (typename MultiTrace<T>::iterator iter = begin(); iter != end();
          ++iter)
        (*iter)->clear();
    }
};

} // namespace RLLib

#endif /* TRACE_H_ */
