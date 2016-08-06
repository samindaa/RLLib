/*
 * LunarLanderAgent_v2.h
 *
 *  Created on: Jul 18, 2016
 *      Author: sabeyruw
 */

#ifndef OPENAI_GYM_LUNARLANDERAGENT_V2_H_
#define OPENAI_GYM_LUNARLANDERAGENT_V2_H_

#include "RLLibOpenAiGymAgentMacro.h"

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
class LunarLanderProjector_v2: public RLLib::Projector<double>
{
  private:
    const double gridResolution;
    int totalTilings;
    RLLib::Hashing<double>* hashing;
    RLLib::Tiles<double>* tiles;
    RLLib::Vector<double>* vector;
    std::vector<RLLib::Vector<double>*> x_tvec;
    std::vector<int> numTilings;
    std::vector<std::vector<int>> events;

  public:
    LunarLanderProjector_v2(RLLib::Random<double>* random) :
        gridResolution(4), totalTilings(0)
    {
      const int memory = 1000000;
      std::cout << "memory: " << memory << std::endl;

      hashing = new RLLib::MurmurHashing<double>(random, memory);
      tiles = new RLLib::Tiles<double>(hashing);
      vector = new RLLib::SVector<double>(hashing->getMemorySize() + 2 + 1/*bias*/);

      for (int i = 1; i <= 3; ++i)
      {
        x_tvec.push_back(new RLLib::PVector<double>(i));
      }

      numTilings.push_back(2);
      numTilings.push_back(4);
      numTilings.push_back(8);

      calculateEvents();

      std::cout << "evens: " << events.size() << std::endl;
      for (size_t i = 0; i < events.size(); ++i)
      {
        std::cout << "i: " << i << " size: " << events[i].size() << "  [";
        for (size_t j = 0; j < events[i].size(); ++j)
        {
          std::cout << events[i][j] << " ";
        }
        std::cout << "]" << std::endl;

        totalTilings += numTilings[events[i].size() - 1];

      }
      std::cout << "totalTilings: " << totalTilings << std::endl;

    }

    virtual ~LunarLanderProjector_v2()
    {
      delete hashing;
      delete tiles;
      delete vector;

      for (std::vector<RLLib::Vector<double>*>::iterator iter = x_tvec.begin();
          iter != x_tvec.end(); ++iter)
      {
        delete *iter;
      }

    }

    const RLLib::Vector<double>* project(const RLLib::Vector<double>* x, const int& h2)
    {
      vector->clear();
      if (x->empty())
      {
        return vector;
      }

      ASSERT(x->dimension() == 8);

      int h1 = 0;
      int size = 0;
      RLLib::Vector<double>* x_t = NULL;
      for (size_t i = 0; i < events.size(); ++i)
      {
        size = events[i].size();
        x_t = x_tvec.at(size - 1);
        x_t->clear();

        for (size_t j = 0; j < size; ++j)
        {
          x_t->setEntry(j, x->getEntry(events[i][j]) * gridResolution);
        }

        tiles->tiles(vector, numTilings[size - 1], x_t, h1++, h2);

      }

      vector->setEntry(vector->dimension() - 3, x->getEntry(6));
      vector->setEntry(vector->dimension() - 2, x->getEntry(7));
      vector->setEntry(vector->dimension() - 1, 1.0);

      return vector;
    }

    const RLLib::Vector<double>* project(const RLLib::Vector<double>* x)
    {
      return project(x, 0);
    }

    double vectorNorm() const
    {
      return totalTilings + 1 + 2;
    }

    int dimension() const
    {
      return vector->dimension();
    }

  private:
    void calculateEvents()
    {
      std::vector<int> currVec;
      std::vector<int> indexVec;
      for (int i = 0; i < 6; ++i)
      {
        indexVec.push_back(i);
      }

      calculateEvents(indexVec, currVec, 0);
    }

    void calculateEvents(const std::vector<int>& indexVec, std::vector<int>& currVec,
        const size_t& i)
    {
      if (currVec.size() > 3)
      {
        return;
      }

      if (!currVec.empty())
      {
        events.push_back(currVec);
      }

      for (size_t j = i; j < indexVec.size(); ++j)
      {
        currVec.push_back(indexVec[j]);
        calculateEvents(indexVec, currVec, j + 1);
        currVec.pop_back();
      }
    }
};

OPENAI_AGENT(LunarLanderAgent_v2, LunarLander-v2)
class LunarLanderAgent_v2: public LunarLanderAgent_v2Base
{
  private:
    // RLLib
    RLLib::Random<double>* random;
    RLLib::Projector<double>* projector;
    RLLib::StateToStateAction<double>* toStateAction;

    double alpha_v;
    double alpha_u;
    double alpha_r;
    double gamma;
    double lambda;

    RLLib::Trace<double>* critice;
    RLLib::TDLambda<double>* critic;
    RLLib::PolicyDistribution<double>* acting;
    RLLib::Trace<double>* actore1;
    RLLib::Traces<double>* actoreTraces;
    RLLib::ActorOnPolicy<double>* actor;
    RLLib::OnPolicyControlLearner<double>* control;
    RLLib::RLAgent<double>* agent;
    RLLib::RLRunner<double>* simulator;

  public:
    LunarLanderAgent_v2();
    virtual ~LunarLanderAgent_v2();

    const RLLib::Action<double>* step();

};

#endif /* OPENAI_GYM_LUNARLANDERAGENT_V2_H_ */
