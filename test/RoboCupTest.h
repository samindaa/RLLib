/*
 * RoboCupTest.h
 *
 *  Created on: Mar 12, 2015
 *      Author: sam
 */

#ifndef TEST_ROBOCUPTEST_H_
#define TEST_ROBOCUPTEST_H_

#include "Test.h"

RLLIB_TEST(RoboCupTest)
class RoboCupTest: public RoboCupTestBase
{
  public:
    RoboCupTest();
    virtual ~RoboCupTest();
    void run();

  private:
    // test cases
    void testSarsaMountainCar();
    void testSarsaTrueMountainCar();
    void testOffPACMountainCar();
    void testGreedyGQMountainCar3D();
    void testSwingPendulumActorCriticWithEligiblity();
    void testOffPACContinuousGridworld();
    void testTemperature();

  protected:
    // RLAgent
    template<class T>
    class RLRAgent: public RLAgent<T>
    {
        typedef RLAgent<T> Base;

      private:
        GQ<T>* gq;
        const Action<T>* a_t;
        Vector<T>* x_t;
        double gamma_tp1;

      public:
        RLRAgent(GQ<T>*gq, Control<T>* control) :
            RLAgent<T>(control), gq(gq), a_t(0), x_t(0), gamma_tp1(0)
        {
        }

        virtual ~RLRAgent()
        {
          if (x_t)
            delete x_t;
        }

        const Action<T>* initialize(const TRStep<T>* step)
        {
          a_t = Base::control->initialize(step->o_tp1);
          Vectors<T>::bufferedCopy(step->o_tp1, x_t);
          return a_t;
        }

        const Action<T>* getAtp1(const TRStep<T>* step)
        {
          gamma_tp1 = (step->observation_tp1->getEntry(0) > 80.0f) ? 0.0f : 1.0f;
          gq->set_gamma_tp1(gamma_tp1);
          const Action<T>* a_tp1 = Base::control->step(x_t, a_t, step->o_tp1, step->r_tp1,
              step->z_tp1);
          a_t = a_tp1;
          Vectors<T>::bufferedCopy(step->o_tp1, x_t);
          return a_t;
        }

        void reset()
        {
          Base::control->reset();
        }
    };

    // RLRProblem
    template<class T>
    class RLRProblem: public RLLib::RLProblem<T>
    {
        typedef RLLib::RLProblem<T> Base;
      protected:
        // Global variables:
        RLLib::Range<T>* temperatureRange;
        T temperature;
        std::vector<double> LKneePitchVec;
        std::vector<double> remainingTimeVec;
        size_t ptr;

      public:
        RLRProblem(RLLib::Random<T>* random = 0) :
            RLLib::RLProblem<T>(random, 1, 1, 1), //
            temperatureRange(new RLLib::Range<T>(47.0f, 91.0f)/*LKneePitch*/), temperature(0.0f), //
            ptr(0)
        {
          Base::discreteActions->push_back(0, 0.0f);
          // subject to change
          Base::continuousActions->push_back(0, 0.0);
          Base::observationRanges->push_back(temperatureRange);

          // Load data
          const std::string fdata =
              "/home/sam/School/conf_papers/papers/RC15-Symposium/RLLibPaper/LKneePitch_RKneePitch_data.txt";
          std::ifstream in(fdata.c_str());
          int LKneePitch;
          int RKneePitch;
          int cuttoffTime = 0;
          if (in.is_open())
          {
            std::string line;
            while (std::getline(in, line))
            {
              if (line.length() == 0)
                continue;
              std::istringstream iss(line);
              if (!(iss >> LKneePitch >> RKneePitch))
              {
                std::cerr << "ERROR! parsing: " << line << std::endl;
                exit(EXIT_FAILURE);
              } // error

              LKneePitchVec.push_back(LKneePitch);
              if (LKneePitch < 80.0f)
                ++cuttoffTime;
            }
            in.close();
          }
          else
          {
            std::cerr << "ERROR! fdata file is missing" << std::endl;
            exit(EXIT_FAILURE);
          }

          for (size_t i = 0; i < LKneePitchVec.size(); i++)
          {
            if (LKneePitchVec[i] < 80.0f)
              remainingTimeVec.push_back(((cuttoffTime - (int) i)) * 10.0f / 1000.0f);
            else
              remainingTimeVec.push_back(0.0f);
          }

#ifdef ROBOCUP_DEBUG
          // debug
          std::cout << "cuttoffTime: " << cuttoffTime << " size: " << LKneePitchVec.size()
          << std::endl;
          const std::string fddata =
          "/home/sam/School/conf_papers/papers/RC15-Symposium/RLLibPaper/debug.txt";
          std::ofstream out(fddata.c_str());
          for (size_t i = 0; i < LKneePitchVec.size(); i++)
          {
            if (LKneePitchVec[i] < 80.0f)
            out << (cuttoffTime - (int) i) << std::endl;
            else
            out << 0 << std::endl;
          }
          out.close();
#endif
        }

        virtual ~RLRProblem()
        {
          delete temperatureRange;
        }

        void updateTRStep()
        {
          Base::output->o_tp1->setEntry(0, temperatureRange->toUnit(temperature));
          Base::output->observation_tp1->setEntry(0, temperature);
        }

        // Profiles
        void initialize()
        {
          ptr = 0;
          temperature = LKneePitchVec.at(ptr);
        }

        void step(const RLLib::Action<T>* a)
        {
          temperature = LKneePitchVec.at(ptr);
          ++ptr;
        }

        bool endOfEpisode() const
        {
          //return temperature > 80.0f;
          return ptr >= LKneePitchVec.size();
        }

        T r() const
        {
          return 0.01f; // 10 ms
        }

        T z() const
        {
          return 0.0f;
        }

        const std::vector<double>& getLKneePitchVec() const
        {
          return LKneePitchVec;
        }

        const std::vector<double>& getRemainingTimeVec() const
        {
          return remainingTimeVec;
        }

        RLLib::Range<T>* getTemperatureRange() const
        {
          return temperatureRange;
        }
    };

    class CombinationGenerator
    {
      public:
        typedef std::map<int, vector<int> > Combinations;

      private:
        int n;
        int k;
        Combinations combinations;
        vector<int> input;
        vector<int> combination;

      public:
        CombinationGenerator(const int& n, const int& k, const int& nbVars) :
            n(n), k(k)
        {
          for (int i = 0; i < nbVars; i++)
            input.push_back(i);
        }

      private:
        void addCombination(const vector<int>& v)
        {
          combinations.insert(make_pair(combinations.size(), v));
        }

        void nextCombination(int offset, int k)
        {
          if (k == 0)
          {
            addCombination(combination);
            return;
          }
          for (int i = offset; i <= n - k; ++i)
          {
            combination.push_back(input[i]);
            nextCombination(i + 1, k - 1);
            combination.pop_back();
          }
        }

      public:
        Combinations& getCombinations()
        {
          nextCombination(0, k);
          return combinations;
        }
    };

    // ====================== Advanced projector ===================================
    template<class T>
    class MountainCar3DTilesProjector: public Projector<T>
    {
      protected:
        Hashing<T>* hashing;
        Tiles<T>* tiles;
        Vector<T>* vector;
        T gridResolution;

      public:
        MountainCar3DTilesProjector(Random<T>* random) :
            hashing(new MurmurHashing<T>(random, 1000000)), tiles(new Tiles<T>(hashing)), vector(
                new SVector<T>(hashing->getMemorySize() + 1)), gridResolution(6)
        {
        }

        virtual ~MountainCar3DTilesProjector()
        {
          delete hashing;
          delete tiles;
          delete vector;
        }

      public:
        const Vector<T>* project(const Vector<T>* x, const int& h2)
        {
          vector->clear();
          if (x->empty())
            return vector;
          int h1 = 0;
          static PVector<T> x4(4);
          x4.set(x)->mapMultiplyToSelf(gridResolution);
          // all 4
          tiles->tiles(vector, 12, &x4, h1++, h2);
          // 3 of 4
          static CombinationGenerator cg43(4, 3, 4); // We know x.dimension() == 4
          static CombinationGenerator::Combinations& c43 = cg43.getCombinations();
          static PVector<T> x3(3);
          for (int i = 0; i < (int) c43.size(); i++)
          {
            for (int j = 0; j < (int) c43[i].size(); j++)
              x3[j] = x->getEntry(c43[i][j]) * gridResolution;
            tiles->tiles(vector, 3, &x3, h1++, h2);
          }
          // 2 of 6
          static CombinationGenerator cg42(4, 2, 4);
          static CombinationGenerator::Combinations& c42 = cg42.getCombinations();
          static PVector<T> x2(2);
          for (int i = 0; i < (int) c42.size(); i++)
          {
            for (int j = 0; j < (int) c42[i].size(); j++)
              x2[j] = x->getEntry(c42[i][j]) * gridResolution;
            tiles->tiles(vector, 2, &x2, h1++, h2);
          }

          // 1 of 4
          static CombinationGenerator cg41(4, 1, 4);
          static CombinationGenerator::Combinations& c41 = cg41.getCombinations();
          static PVector<T> x1(1);
          for (int i = 0; i < (int) c41.size(); i++)
          {
            x1[0] = x->getEntry(c41[i][0]) * gridResolution; // there is only a single element
            tiles->tiles(vector, 3, &x1, h1++, h2);
          }

          // bias
          vector->setEntry(vector->dimension() - 1, 1.0);
          return vector;
        }

        const Vector<T>* project(const Vector<T>* x)
        {

          vector->clear();
          if (x->empty())
            return vector;
          int h1 = 0;
          static PVector<T> x4(4);
          x4.set(x)->mapMultiplyToSelf(gridResolution);
          // all 4
          tiles->tiles(vector, 12, &x4, h1++);
          // 3 of 4
          static CombinationGenerator cg43(4, 3, 4); // We know x.dimension() == 4
          static CombinationGenerator::Combinations& c43 = cg43.getCombinations();
          static PVector<T> x3(3);
          for (int i = 0; i < (int) c43.size(); i++)
          {
            for (int j = 0; j < (int) c43[i].size(); j++)
              x3[j] = x->getEntry(c43[i][j]) * gridResolution;
            tiles->tiles(vector, 3, &x3, h1++);
          }
          // 2 of 6
          static CombinationGenerator cg42(4, 2, 4);
          static CombinationGenerator::Combinations& c42 = cg42.getCombinations();
          static PVector<T> x2(2);
          for (int i = 0; i < (int) c42.size(); i++)
          {
            for (int j = 0; j < (int) c42[i].size(); j++)
              x2[j] = x->getEntry(c42[i][j]) * gridResolution;
            tiles->tiles(vector, 2, &x2, h1++);
          }

          // 1 of 4
          static CombinationGenerator cg41(4, 1, 4);
          static CombinationGenerator::Combinations& c41 = cg41.getCombinations();
          static PVector<T> x1(1);
          for (int i = 0; i < (int) c41.size(); i++)
          {
            x1[0] = x->getEntry(c41[i][0]) * gridResolution; // there is only a single element
            tiles->tiles(vector, 3, &x1, h1++);
          }

          // bias
          vector->setEntry(vector->dimension() - 1, 1.0);
          return vector;
        }

        double vectorNorm() const
        {
          return 48 + 1;
        }
        int dimension() const
        {
          return vector->dimension();
        }
    };

};

#endif /* TEST_ROBOCUPTEST_H_ */
