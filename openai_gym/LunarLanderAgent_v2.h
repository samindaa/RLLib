/*
 * LunarLanderAgent_v2.h
 *
 *  Created on: Jul 18, 2016
 *      Author: sabeyruw
 */

#ifndef OPENAI_GYM_LUNARLANDERAGENT_V2_H_
#define OPENAI_GYM_LUNARLANDERAGENT_V2_H_

#include "RLLibOpenAiGymAgent.h"

//Env
class LunarLander_v2: public OpenAiGymRLProblem
{
  protected:
    // Global variables:
    std::vector<RLLib::Range<double>*> ranges;
    std::vector<std::pair<double, double>> stats;

    size_t numObservationUpdates;

  public:
    LunarLander_v2() :
        OpenAiGymRLProblem(8, 4, 1/*NA*/), numObservationUpdates(0)
    {

      /*Preliminary experiments*/
      ranges.push_back(new RLLib::Range<double>(-1.01422, 1.01184)); // x
      ranges.push_back(new RLLib::Range<double>(-0.161061, 1.16219)); // y
      ranges.push_back(new RLLib::Range<double>(-1.77318, 1.62135)); // xdot
      ranges.push_back(new RLLib::Range<double>(-1.92527, 0.513943)); // ydot
      ranges.push_back(new RLLib::Range<double>(-3.73458, 3.44188)); // angle
      ranges.push_back(new RLLib::Range<double>(-6.39456, 6.53424)); // angleDot
      ranges.push_back(new RLLib::Range<double>(0.0f, 1.0f)); // touch 1
      ranges.push_back(new RLLib::Range<double>(0.0f, 1.0f)); // touch 2

      for (int i = 0; i < discreteActions->dimension(); ++i)
      {
        discreteActions->push_back(i, i);
      }

      // subject to change
      continuousActions->push_back(0, 0.0);

      for (std::vector<RLLib::Range<double>*>::iterator iter = ranges.begin(); iter != ranges.end();
          ++iter)
      {
        observationRanges->push_back(*iter);
      }

      for (size_t i = 0; i < ranges.size(); ++i)
      {
        stats.push_back(std::make_pair(ranges[i]->min(), ranges[i]->max()));
      }

    }

    virtual ~LunarLander_v2()
    {

      for (std::vector<RLLib::Range<double>*>::iterator iter = ranges.begin(); iter != ranges.end();
          ++iter)
      {
        delete *iter;
      }

    }

    void updateTRStep()
    {

      for (int i = 0; i < output->observation_tp1->dimension(); ++i)
      {
        output->o_tp1->setEntry(i, ranges[i]->toUnit(step_tp1->observation_tp1.at(i)));
        output->observation_tp1->setEntry(i, step_tp1->observation_tp1.at(i));

        // calculateMin and max
        stats[i].first = std::min(stats[i].first, output->observation_tp1->getEntry(i)); // min
        stats[i].second = std::max(stats[i].second, output->observation_tp1->getEntry(i)); // max

      }

      // Collect statistics
      if (++numObservationUpdates % 101 == 0)
      {
        for (std::vector<RLLib::Range<double>*>::iterator iter = ranges.begin();
            iter != ranges.end(); ++iter)
        {
          delete *iter;
        }

        ranges.clear();

        if (observationRanges)
        {
          delete observationRanges;
          observationRanges = new RLLib::Ranges<double>();

          for (int i = 0; i < output->observation_tp1->dimension(); ++i)
          {
            ranges.push_back(new RLLib::Range<double>(stats[i].first, stats[i].second));
            observationRanges->push_back(ranges[i]);
          }

        }

        numObservationUpdates = 0;

      }
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
      ASSERT(x->dimension() == 8);
      vector->clear();
      if (x->empty())
      {
        return vector;
      }

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

// Agent
class LunarLanderAgent_v2: public RLLibOpenAiGymAgent
{
  private:
    // RLLib
    RLLib::Random<double>* random;
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
