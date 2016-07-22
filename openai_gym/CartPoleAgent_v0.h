/*
 * CartPoleAgent.h
 *
 *  Created on: Jun 27, 2016
 *      Author: sabeyruw
 */

#ifndef OPENAI_GYM_CARTPOLEAGENT_V0_H_
#define OPENAI_GYM_CARTPOLEAGENT_V0_H_

#include "RLLibOpenAiGymAgentMacro.h"

//Env
class CartPole_v0: public OpenAiGymRLProblem
{
  protected:
    // Global variables:
    RLLib::Range<double>* thetaRange;
    RLLib::Range<double>* theta1DotRange;
    RLLib::Range<double>* theta2DotRange;

  public:
    CartPole_v0() :
        OpenAiGymRLProblem(4, 2, 1),  //
        thetaRange(new RLLib::Range<double>(-M_PI, M_PI)), //
        theta1DotRange(new RLLib::Range<double>(-4.0 * M_PI, 4.0 * M_PI)), //
        theta2DotRange(new RLLib::Range<double>(-9.0 * M_PI, 9.0 * M_PI))
    {

      for (int i = 0; i < discreteActions->dimension(); ++i)
      {
        discreteActions->push_back(i, i);
      }

      // subject to change
      continuousActions->push_back(0, 0.0);

      observationRanges->push_back(thetaRange);
      observationRanges->push_back(theta1DotRange);
      observationRanges->push_back(theta2DotRange);
    }

    virtual ~CartPole_v0()
    {
      delete thetaRange;
      delete theta1DotRange;
      delete theta2DotRange;
    }

    void updateTRStep()
    {
      output->o_tp1->setEntry(0, thetaRange->toUnit(step_tp1->observation_tp1.at(0)));
      output->o_tp1->setEntry(1, thetaRange->toUnit(step_tp1->observation_tp1.at(1)));
      output->o_tp1->setEntry(2, theta1DotRange->toUnit(step_tp1->observation_tp1.at(2)));
      output->o_tp1->setEntry(3, theta2DotRange->toUnit(step_tp1->observation_tp1.at(3)));

      output->observation_tp1->setEntry(0, step_tp1->observation_tp1.at(0));
      output->observation_tp1->setEntry(1, step_tp1->observation_tp1.at(1));
      output->observation_tp1->setEntry(2, theta1DotRange->bound(step_tp1->observation_tp1.at(2)));
      output->observation_tp1->setEntry(1, theta2DotRange->bound(step_tp1->observation_tp1.at(3)));

    }
};

class CartPoleProjector_v0: public RLLib::Projector<double>
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
    CartPoleProjector_v0(RLLib::Random<double>* random) :
        gridResolution(6), totalTilings(0)
    {
      /**
       * Agent configuration according to:
       * https://webdocs.cs.ualberta.ca/~sutton/papers/sutton-96.pdf
       */

      int memory = 0;
      memory += (12 * std::pow(gridResolution, 4));
      memory += (4 * 3 * std::pow(gridResolution, 3));
      memory += (6 * 2 * std::pow(gridResolution, 2));
      memory += (4 * 3 * std::pow(gridResolution, 1));
      memory *= 1;

      std::cout << "memory: " << memory << std::endl;

      hashing = new RLLib::MurmurHashing<double>(random, memory);
      tiles = new RLLib::Tiles<double>(hashing);
      vector = new RLLib::SVector<double>(hashing->getMemorySize() + 1/*bias*/);

      for (int i = 1; i <= 4; ++i)
      {
        x_tvec.push_back(new RLLib::PVector<double>(i));
      }

      numTilings.push_back(3);
      numTilings.push_back(2);
      numTilings.push_back(3);
      numTilings.push_back(12);

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

    virtual ~CartPoleProjector_v0()
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

      ASSERT(x->dimension() == 4);

      int h1 = 0;
      int size = 0;
      RLLib::Vector<double>* x_t = NULL;
      for (size_t i = 0; i < events.size(); ++i)
      {
        size = events[i].size();
        x_t = x_tvec.at(size - 1);
        x_t->clear();

        ASSERT(x_t->dimension() == size);

        for (size_t j = 0; j < size; ++j)
        {
          x_t->setEntry(j, x->getEntry(events[i][j]) * gridResolution);
        }

        tiles->tiles(vector, numTilings[size - 1], x_t, h1++, h2);

      }

      vector->setEntry(vector->dimension() - 1, 1.0);

      return vector;
    }

    const RLLib::Vector<double>* project(const RLLib::Vector<double>* x)
    {
      return project(x, 0);
    }

    double vectorNorm() const
    {
      return totalTilings + 1;
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
      for (int i = 0; i < 4; ++i)
      {
        indexVec.push_back(i);
      }

      calculateEvents(indexVec, currVec, 0);
    }

    void calculateEvents(const std::vector<int>& indexVec, std::vector<int>& currVec,
        const size_t& i)
    {
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

OPENAI_AGENT(CartPoleAgent_v0, CartPole-v0)
class CartPoleAgent_v0: public CartPoleAgent_v0Base
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
    CartPoleAgent_v0();
    virtual ~CartPoleAgent_v0();
    const RLLib::Action<double>* step();
};

#endif /* OPENAI_GYM_CARTPOLEAGENT_V0_H_ */
