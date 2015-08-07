/*
 * HordTest.h
 *
 *  Created on: Jul 20, 2015
 *      Author: sam
 */

#ifndef TEST_HORDTEST_H_
#define TEST_HORDTEST_H_

#include "Test.h"
#include "util/Eigen/Dense"

class HordeProblem;
class HordeProjector;

RLLIB_TEST(HordTest)
class HordTest: public HordTestBase
{
  private:
    // RLLib:
    RLLib::Random<double>* random;
    HordeProblem* problem;
    RLLib::Hashing<double>* hashing;
    RLLib::Projector<double>* projector;
    RLLib::StateToStateAction<double>* toStateAction;
    RLLib::Trace<double>* e;
    double alpha;
    double gamma;
    double lambda;

    Sarsa<double>* sarsa;
    Policy<double>* target;

    OnPolicyControlLearner<double>* control;
    RLLib::RLAgent<double>* agent;
    RLLib::RLRunner<double>* runner;

  public:
    HordTest();
    ~HordTest();

    void run();
};

// RLLib
#define HordeProblem_HIS_LEN 5
#define HordeProblem_NUM_VAR 2

class HordeProblem: public RLLib::RLProblem<double>
{
  private:
    RLLib::Range<double>* sonarRange;
    bool m_endOfEpisode;
    double m_r;
    int nextLine;

    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix_t;
    Matrix_t M_data;
    Eigen::VectorXd m_x;

  public:
    HordeProblem(RLLib::Random<double>* random);
    virtual ~HordeProblem();
    void initialize();
    void step(const RLLib::Action<double>* action);
    void updateTRStep();
    bool endOfEpisode() const;
    double r() const;
    double z() const;

    void setNextLine(const int& nextLine)
    {
      this->nextLine = nextLine;
    }

  private:
    void updateStateVariables();
};

class HordeProjector: public RLLib::Projector<double>
{
  protected:
    RLLib::Vector<double>* vector;
    RLLib::Random<double>* random;
    RLLib::Tiles<double>* tiles;
    RLLib::Vector<double>* inputs;
    RLLib::Vector<double>* gridResolutions;
    int numTilings;
    double gridResolution;

  public:
    HordeProjector(RLLib::Hashing<double>* hashing);
    virtual ~HordeProjector();
    const RLLib::Vector<double>* project(const RLLib::Vector<double>* x, const int& h1);
    const RLLib::Vector<double>* project(const RLLib::Vector<double>* x);
    double vectorNorm() const;
    int dimension() const;

  private:
    const RLLib::Vector<double>* project(const RLLib::Vector<double>* x, const int& h1,
        const int& h2);
};

class HordePolicy: public DiscreteActionPolicy<double>
{
  protected:
    Actions<double>* actions;
    Predictor<double>* predictor;
    std::ofstream m_ofstream;
    double m_prediction;

  public:
    HordePolicy(Actions<double>* actions, Predictor<double>* predictor);
    void update(const Representations<double>* phi_tp1);
    double pi(const Action<double>* a);
    const Action<double>* sampleAction();
    const Action<double>* sampleBestAction();
};

#endif /* TEST_HORDTEST_H_ */
