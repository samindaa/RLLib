/*
 * NextingTest.h
 *
 *  Created on: Nov 25, 2013
 *      Author: sam
 */

#ifndef NEXTINGTEST_H_
#define NEXTINGTEST_H_

#include "Test.h"
#include <pthread.h>

/**
 * Based on the papers:
 * Multi-timescale Nexting in a Reinforcement Learning Robot
 *  and
 * Scaling Life-long Off-policy Learning
 *
 */
RLLIB_TEST(NextingTest)
class NextingTest: public NextingTestBase
{
  public:
    void run();

  private:
    void testNexting();
};

template<class T>
class Thread
{
  protected:
    pthread_t tid;
    int running;
    int detached;

  public:
    Thread() :
        tid(0), running(0), detached(0)
    {
    }

    virtual ~Thread()
    {
      if (running == 1 && detached == 0)
        pthread_detach(tid);
      if (running == 1)
        pthread_cancel(tid);
    }

    int join()
    {
      int result = -1;
      if (running == 1)
      {
        result = pthread_join(tid, 0);
        if (result == 0)
          detached = 0;
      }
      return result;
    }

    int detach()
    {
      int result = -1;
      if (running == 1 && detached == 0)
      {
        result = pthread_detach(tid);
        if (result == 0)
          detached = 1;
      }
      return result;
    }

    pthread_t self()
    {
      return tid;
    }

    int start();
    virtual void* run() =0;
};

// Simple Thread
template<class T>
static void* threadCallback(void* arg)
{
  return ((Thread<T>*) arg)->run();
}

template<class T> int Thread<T>::start()
{
  int result = pthread_create(&tid, 0, threadCallback<T>, this);
  if (result == 0)
    running = 1;
  return result;
}

class NextingProblem: public RLProblem<double>
{
  public:
    NextingProblem(Random<double>* random);
    virtual ~NextingProblem();
    void initialize();
    void step(const Action<double>* action);
    void updateTRStep();
    bool endOfEpisode() const;
    double r() const;
    double z() const;
};

class NextingProjector: public Projector<double>
{
  protected:
    int nbTiles;
    int memory;
    Vector<double>* vector;
    Random<double>* random;
    Hashing<double>* hashing;
    Tiles<double>* tiles;
  public:
    NextingProjector();
    virtual ~NextingProjector();
    const Vector<double>* project(const Vector<double>* x, const int& h1);
    const Vector<double>* project(const Vector<double>* x);
    double vectorNorm() const;
    int dimension() const;
};

// Horde simulation
template<class T>
class Demon
{
  public:
    virtual ~Demon()
    {
    }

    virtual void initialize() =0;
    virtual void update(const Vector<T>* x_t, const Action<T>* a_t, const Vector<T>* x_tp1) =0;
};

template<class T>
class PredictionDemon: public Demon<T>
{
  protected:
    OnPolicyTD<T>* td;
    RewardFunction<T>* reward;

  public:
    PredictionDemon(OnPolicyTD<T>* td, RewardFunction<T>* reward) :
        td(td), reward(reward)
    {
    }

    virtual ~PredictionDemon()
    {
    }

    void initialize()
    {
      ((LinearLearner<T>*) td)->initialize();
    }

    void update(const Vector<T>* x_t, const Action<T>* a_t, const Vector<T>* x_tp1)
    {
      td->update(x_t, x_tp1, reward->reward());
    }
};

template<class T>
class Horde
{
  protected:
    std::vector<Trace<T>*>* traces;
    std::vector<OnPolicyTD<T>*>* predictions;
    std::vector<RewardFunction<T>*>* rewards;
    std::vector<Demon<T>*>* demons;

    Projector<T>* projector;
    Vector<T>* x_t;

  public:
    Horde(Projector<T>* projector) :
        traces(new std::vector<Trace<T>*>), predictions(new std::vector<OnPolicyTD<T>*>), rewards(
            new std::vector<RewardFunction<T>*>), demons(new std::vector<Demon<T>*>), projector(
            projector), x_t(0)
    {
    }

    virtual ~Horde()
    {
      for (typename std::vector<Trace<T>*>::iterator iter = traces->begin(); iter != traces->end();
          ++iter)
        delete *iter;
      for (typename std::vector<OnPolicyTD<T>*>::iterator iter = predictions->begin();
          iter != predictions->end(); ++iter)
        delete *iter;
      for (typename std::vector<RewardFunction<T>*>::iterator iter = rewards->begin();
          iter != rewards->end(); ++iter)
        delete *iter;
      for (typename std::vector<Demon<T>*>::iterator iter = demons->begin(); iter != demons->end();
          ++iter)
        delete *iter;

      delete traces;
      delete predictions;
      delete rewards;
      delete demons;
      if (x_t)
        delete x_t;
    }

    void initialize(const TRStep<T>* step)
    {
      Vectors<T>::bufferedCopy(projector->project(step->o_tp1), x_t);
      // FixMe: threading
      for (typename std::vector<Demon<T>*>::iterator iter = demons->begin(); iter != demons->end();
          ++iter)
        (*iter)->initialize();
    }

    void update(const Vector<T>* o_t, const Action<T>* a_t, const TRStep<T>* step)
    {
      const Vector<T>* x_tp1 = projector->project(step->o_tp1);
      // FixMe: threading
      for (typename std::vector<Demon<T>*>::iterator iter = demons->begin(); iter != demons->end();
          ++iter)
        (*iter)->update(x_t, a_t, x_tp1);
      Vectors<T>::bufferedCopy(x_tp1, x_t);
    }

    void addOnPolicyDemon(Trace<T>* e, RewardFunction<T>* reward, OnPolicyTD<T>* td)
    {
      traces->push_back(e);
      rewards->push_back(reward);
      predictions->push_back(td);
      demons->push_back(new PredictionDemon<T>(td, reward));
    }
};

template<class T>
class HordeAgent: public RLAgent<T>
{
    typedef RLAgent<T> Base;
  protected:
    Horde<T>* horde;
    Vector<T>* o_t;
    const Action<T>* a_t;

  public:
    HordeAgent(Control<T>* control, Horde<T>* horde) :
        RLAgent<T>(control), horde(horde), o_t(0), a_t(0)
    {
    }

    virtual ~HordeAgent()
    {
      if (o_t)
        delete o_t;
    }

    const Action<T>* initialize(const TRStep<T>* step)
    {
      horde->initialize(step);
      Vectors<T>::bufferedCopy(step->o_tp1, o_t);
      a_t = Base::control->initialize(step->o_tp1);
      return a_t;
    }

    const Action<T>* getAtp1(const TRStep<T>* step)
    {
      horde->update(o_t, a_t, step);
      Vectors<T>::bufferedCopy(step->o_tp1, o_t);
      a_t = Base::control->step(o_t, a_t, step->o_tp1, step->r_tp1, step->z_tp1);
      return a_t;
    }

    void reset()
    {
    }
};

//==
template<class T>
class FakeRewardFunction: public RewardFunction<T>
{
  public:
    T reward()
    {
      return 0.0f;
    }
};

// Simple control algorithm
template<class T>
class PolicyBasedControl: public OnPolicyControlLearner<T>
{
  protected:
    Policy<T>* policy;
    StateToStateAction<T>* toStateAction;

  public:
    PolicyBasedControl(Policy<T>* policy, StateToStateAction<T>* toStateAction) :
        policy(policy), toStateAction(toStateAction)
    {
    }

    virtual ~PolicyBasedControl()
    {
    }

    const Action<T>* initialize(const Vector<T>* x)
    {
      const Representations<T>* phi = toStateAction->stateActions(x);
      return Policies::sampleAction(policy, phi);
    }

    const Action<T>* step(const Vector<T>* x_t, const Action<T>* a_t, const Vector<T>* x_tp1,
        const double& r_tp1, const double& z_tp1)
    {
      const Representations<T>* phi_tp1 = toStateAction->stateActions(x_tp1);
      return Policies::sampleAction(policy, phi_tp1);
    }

    void reset()
    {
    }

    const Action<T>* proposeAction(const Vector<T>* x)
    {
      return Policies::sampleBestAction(policy, toStateAction->stateActions(x));
    }

    double computeValueFunction(const Vector<T>* x) const
    {
      return 0.0f;
    }

    const Predictor<T>* predictor() const
    {
      return 0;
    }

    void persist(const char* f) const
    {
    }

    void resurrect(const char* f)
    {
    }

};

#endif /* NEXTINGTEST_H_ */
