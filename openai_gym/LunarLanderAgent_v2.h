/*
 * LunarLanderAgent_v2.h
 *
 *  Created on: Jul 18, 2016
 *      Author: sabeyruw
 */

#ifndef OPENAI_GYM_LUNARLANDERAGENT_V2_H_
#define OPENAI_GYM_LUNARLANDERAGENT_V2_H_

#include "RLLibOpenAiGymAgentMacro.h"
#include "FourierBasis.h"

//Env
class LunarLander_v2: public OpenAiGymRLProblem
{
  protected:

  public:
    LunarLander_v2() :
        OpenAiGymRLProblem(8, 4, 1/*NA*/)
    {

      /*The values are collected from preliminary experiments*/
      observationRanges->push_back(new RLLib::Range<double>(-2.0, 2.0)); // x
      observationRanges->push_back(new RLLib::Range<double>(-0.5, 2.0)); // y
      observationRanges->push_back(new RLLib::Range<double>(-3.0, 3.0)); // xdot
      observationRanges->push_back(new RLLib::Range<double>(-3.0, 1.0)); // ydot
      observationRanges->push_back(new RLLib::Range<double>(-M_PI, M_PI)); // angle
      observationRanges->push_back(new RLLib::Range<double>(-M_PI, M_PI)); // angleDot
      observationRanges->push_back(new RLLib::Range<double>(0.0f, 1.0f)); // touch 1
      observationRanges->push_back(new RLLib::Range<double>(0.0f, 1.0f)); // touch 2

      for (int i = 0; i < discreteActions->dimension(); ++i)
      {
        discreteActions->push_back(i, i);
      }

      // subject to change
      continuousActions->push_back(0, 0.0);

    }

    virtual ~LunarLander_v2()
    {
    }
};

// Create the feature extractor
// based on Fourier Basis (therefore, dense)
class LunarLanderProjector_v2: public RLLib::Projector<double>
{
  private:
    RLLib::Vector<double>* x_t;
    RLLib::FourierCoefficientGenerator<double>* generator;
    RLLib::Vector<double>* featureVector;
    std::vector<RLLib::Vector<double>*> multipliers;
    std::vector<std::vector<int>> indexSets;

  public:
    LunarLanderProjector_v2(const int& order, const RLLib::Actions<double>* actions) :
        x_t(new RLLib::PVector<double>(4)), //
        generator(new RLLib::FullFourierCoefficientGenerator<double>())
    {

      generator->computeFourierCoefficients(multipliers, x_t->dimension(), order);

      // remove the first multiplier => 1, then add it later
      multipliers.erase(multipliers.begin());

      populateIndexSets();

      std::cout << "multipliers: " << multipliers.size() << " indexSets: " << indexSets.size()
          << std::endl;
      for (size_t i = 0; i < indexSets.size(); ++i)
      {
        std::cout << "i: " << i << " size: " << indexSets[i].size() << "  [";
        for (size_t j = 0; j < indexSets[i].size(); ++j)
        {
          std::cout << indexSets[i][j] << " ";
        }
        std::cout << "]" << std::endl;
      }

      featureVector = new RLLib::PVector<double>(
          multipliers.size() * indexSets.size() * actions->dimension() + 2 + 1);

      std::cout << "featureVector: " << featureVector->dimension() << std::endl;

    }

    virtual ~LunarLanderProjector_v2()
    {
      delete x_t;
      delete generator;
      delete featureVector;
      for (std::vector<RLLib::Vector<double>*>::iterator iter = multipliers.begin();
          iter != multipliers.end(); ++iter)
        delete *iter;

    }

    /**
     * x must be unit normalized [0, 1)
     */
    const RLLib::Vector<double>* project(const RLLib::Vector<double>* x, const int& h1)
    {
      featureVector->clear();
      if (x->empty())
        return featureVector;

      const int stripWidth = multipliers.size() * indexSets.size() * h1;

      for (size_t k = 0; k < indexSets.size(); ++k)
      {
        x_t->clear();
        const int offset = k * multipliers.size() + stripWidth;
        auto& set = indexSets[k];
        for (size_t i = 0; i < set.size(); ++i)
        {
          ASSERT(set[i] < x->dimension());
          x_t->setEntry(i, x->getEntry(set[i]));
        }

        for (size_t i = 0; i < multipliers.size(); i++)
        {
          featureVector->setEntry(offset + i, std::cos(M_PI * x_t->dot(multipliers[i])));
        }
      }

      featureVector->setEntry(featureVector->dimension() - 3, x->getEntry(x->dimension() - 2));
      featureVector->setEntry(featureVector->dimension() - 2, x->getEntry(x->dimension() - 1));
      featureVector->setEntry(featureVector->dimension() - 1, 1.0f);

      return featureVector;

    }

    const RLLib::Vector<double>* project(const RLLib::Vector<double>* x)
    {
      return project(x, 0);
    }

    double vectorNorm() const
    {
      return 1.0f;
    }

    int dimension() const
    {
      return featureVector->dimension();
    }

  private:
    void populateIndexSets()
    {
      std::vector<int> curr;
      std::vector<int> indices(6); // TODO: param
      std::iota(indices.begin(), indices.end(), 0);

      populateIndexSets(indices, curr, 0);
    }

    void populateIndexSets(const std::vector<int>& indices, std::vector<int>& curr, const size_t& i)
    {
      if (curr.size() == static_cast<size_t>(x_t->dimension()))
      {
        indexSets.push_back(curr);
        return;
      }

      for (size_t j = i; j < indices.size(); ++j)
      {
        curr.push_back(indices[j]); // use
        populateIndexSets(indices, curr, j + 1);
        curr.pop_back(); // remove
      }
    }
};

OPENAI_AGENT(LunarLanderAgent_v2, LunarLander-v2)
class LunarLanderAgent_v2: public LunarLanderAgent_v2Base
{
  private:
    // RLLib
    RLLib::Random<double>* random;
    int order;
    RLLib::Projector<double>* projector;
    RLLib::StateToStateAction<double>* toStateAction;
    RLLib::Trace<double>* e;
    double alpha;
    double gamma;
    double lambda;
    RLLib::Sarsa<double>* sarsa;
    double epsilon;
    RLLib::Policy<double>* acting;
    RLLib::OnPolicyControlLearner<double>* control;
    RLLib::RLAgent<double>* agent;
    RLLib::RLRunner<double>* simulator;

  public:
    LunarLanderAgent_v2();
    virtual ~LunarLanderAgent_v2();

    const RLLib::Action<double>* step();

};

#endif /* OPENAI_GYM_LUNARLANDERAGENT_V2_H_ */
