/*
 * FourierBasis.h
 *
 *  Created on: Aug 5, 2016
 *      Author: sabeyruw
 */

#ifndef INCLUDE_FOURIERBASIS_H_
#define INCLUDE_FOURIERBASIS_H_

/**
 * http://library.rl-community.org/wiki/Sarsa_Lambda_Fourier_Basis_(Java)
 * Based on reference implementation.
 */

#include "Projector.h"

namespace RLLib
{
  template<typename T>
  class FourierCoefficientGenerator
  {
    public:
      virtual ~FourierCoefficientGenerator()
      {
      }
      virtual void computeFourierCoefficients(std::vector<Vector<T>*>& multipliers,
          const int& nbInputs, const int& order) =0;
  };

  template<typename T>
  class FullFourierCoefficientGenerator: public FourierCoefficientGenerator<T>
  {
    public:
      void computeFourierCoefficients(std::vector<Vector<T>*>& multipliers, const int& nbInputs,
          const int& order)
      {
        Vector<T>* multiplierVector = new PVector<T>(nbInputs);
        do
        {
          Vector<T>* newMultiplierVector = new PVector<T>(nbInputs);
          newMultiplierVector->set(multiplierVector);
          multipliers.push_back(newMultiplierVector);
          nextMultiplierVector(multiplierVector, nbInputs, order);
        } while (multiplierVector->getEntry(0) <= order);
        ASSERT(multipliers.size() == std::pow(order + 1, nbInputs));
        delete multiplierVector;
      }

    private:
      void nextMultiplierVector(Vector<T>* multiplierVector, const int& nbInputs, const int& order)
      {
        multiplierVector->setEntry(nbInputs - 1, multiplierVector->getEntry(nbInputs - 1) + 1);
        if (multiplierVector->getEntry(nbInputs - 1) > order)
        {
          if (nbInputs > 1)
          {
            multiplierVector->setEntry(nbInputs - 1, 0);
            nextMultiplierVector(multiplierVector, nbInputs - 1, order);
          }
        }
      }
  };

  template<typename T>
  class IndependentFourierCoefficientGenerator: public FourierCoefficientGenerator<T>
  {
    public:
      void computeFourierCoefficients(std::vector<Vector<T>*>& multipliers, const int& nbInputs,
          const int& order)
      {
        const int nbTerms = (order * nbInputs) + 1;
        for (int i = 0; i < nbTerms; ++i)
        {
          Vector<T>* multiplierVector = new PVector<T>(nbInputs);
          multipliers.push_back(multiplierVector);
        }

        int pos = 0;
        ++pos;

        for (int v = 0; v < nbInputs; v++)
        {
          // For each variable, cycle up to its order.
          for (int ord = 1; ord <= order; ord++)
          {
            multipliers[pos]->setEntry(v, ord);
            pos++;
          }
        }

        ASSERT(nbTerms == pos);

      }
  };

  template<typename T>
  class FourierBasis: public Projector<T>
  {
    protected:
      FourierCoefficientGenerator<T>* generator;
      Vector<T>* featureVector;
      std::vector<Vector<T>*> multipliers;

    public:
      FourierBasis(const int& nbInputs, const int& order, const Actions<T>* actions,
          FourierCoefficientGenerator<T>* generator = NULL/*fixme*/) :
          generator(new FullFourierCoefficientGenerator<T>())
      {
        this->generator->computeFourierCoefficients(multipliers, nbInputs, order);
        featureVector = new PVector<T>(multipliers.size() * actions->dimension());
      }

      virtual ~FourierBasis()
      {
        delete generator;
        delete featureVector;
        for (typename std::vector<Vector<T>*>::iterator iter = multipliers.begin();
            iter != multipliers.end(); ++iter)
          delete *iter;
      }

      /**
       * x must be unit normalized [0, 1)
       */
      const Vector<T>* project(const Vector<T>* x, const int& h1)
      {
        featureVector->clear();
        if (x->empty())
          return featureVector;
        // FixMe: SIMD
        const int stripWidth = multipliers.size() * h1;
        for (size_t i = 0; i < multipliers.size(); i++)
        {
          featureVector->setEntry(i + stripWidth, std::cos(M_PI * x->dot(multipliers[i])));
        }
        return featureVector;

      }

      const Vector<T>* project(const Vector<T>* x)
      {
        return project(x, 0);
      }

      T vectorNorm() const
      {
        return T(1); //FixMe:
      }

      int dimension() const
      {
        return featureVector->dimension();
      }

      const std::vector<Vector<T>*>& getMultipliers() const
      {
        return multipliers;
      }
  };

}  // namespace RLLib 

#endif /* INCLUDE_FOURIERBASIS_H_ */
