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
        defaultThreshold(threshold), threshold(threshold),
            vector(new SparseVector<T>(numFeatures))
    {
    }
    virtual ~ATrace()
    {
      delete vector;
    }

  public:

    virtual void clearBelowThreshold()
    {
      const T* values = vector->getValues();
      const int* indexes = vector->getActiveIndexes();
      int i = 0;
      while (i < vector->numActiveEntries())
      {
        T absValue = fabs(values[i]);
        if (absValue <= threshold) vector->removeEntry(indexes[i]);
        else i++;
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
      clearBelowThreshold();
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
};

template<class T>
class AMaxTrace: public ATrace<T>
{
  protected:
    int nonZeroTraces;
  public:
    AMaxTrace(const int& capacity, const int& nonZeroTraces,
        const double& threshold = 1e-8) :
        ATrace<T>(capacity, threshold), nonZeroTraces(nonZeroTraces)
    {
    }

  private:
    void cullingTraces()
    {
      while (ATrace<T>::vector->numActiveEntries() > nonZeroTraces)
      {
        ATrace<T>::threshold += 0.1 * ATrace<T>::threshold;
        ATrace<T>::clearBelowThreshold();
        std::cout << "@@>> cullingTraces " << ATrace<T>::threshold << std::endl;
      }
    }

  public:
    void updateVector(const double& lambda, const SparseVector<T>& phi)
    {
      cullingTraces();
      ATrace<T>::updateVector(lambda, phi);
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
      for (const int* index = indexes; index < indexes + phi.numActiveEntries();
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
class RMaxTrace: public RTrace<T>
{
  protected:
    int nonZeroTraces;
  public:
    RMaxTrace(const int& capacity, const int& nonZeroTraces,
        const double& threshold = 1e-8) :
        RTrace<T>(capacity, threshold), nonZeroTraces(nonZeroTraces)
    {
    }

  private:
    void cullingTraces()
    {
      while (ATrace<T>::vector->numActiveEntries() > nonZeroTraces)
      {
        ATrace<T>::threshold += 0.1 * ATrace<T>::threshold;
        ATrace<T>::clearBelowThreshold();
        //std::cout << "@@>> cullingTraces " << ATrace<T>::threshold << std::endl;
      }
    }

  public:
    void updateVector(const double& lambda, const SparseVector<T>& phi)
    {
      cullingTraces();
      RTrace<T>::updateVector(lambda, phi);
    }
};

#endif /* TRACE_H_ */
