/*
 * RL.h
 *
 *  Created on: Nov 13, 2013
 *      Author: sam
 */

#ifndef RL_H_
#define RL_H_

#include <typeinfo>

#include "Vector.h"
#include "Action.h"
#include "Math.h"
#include "Control.h"
#include "Timer.h"

namespace RLLib
{

template<class T>
class TRStep
{
  public:
    DenseVector<T>* o_tp1;
    T r_tp1;
    T z_tp1;
    bool endOfEpisode;
    TRStep(const int& nbVars) :
        o_tp1(new PVector<T>(nbVars)), r_tp1(0.0f), z_tp1(0.0f), endOfEpisode(false)
    {
    }

    void updateRTStep(const T& r_tp1, const T& z_tp1, const bool& endOfEpisode)
    {
      this->r_tp1 = r_tp1;
      this->z_tp1 = z_tp1;
      this->endOfEpisode = endOfEpisode;
    }

    ~TRStep()
    {
      delete o_tp1;
    }

    void setForcedEndOfEpisode(const bool& endOfEpisode)
    {
      this->endOfEpisode = endOfEpisode;
    }
};

template<class T>
class RLAgent
{
  protected:
    Control<T>* control;
    RLAgent(Control<T>* control) :
        control(control)
    {
    }

  public:
    virtual ~RLAgent()
    {
    }

    virtual const Action<T>* initialize(const TRStep<T>* step) =0;
    virtual const Action<T>* getAtp1(const TRStep<T>* step) =0;
    virtual void reset() =0;

    virtual Control<T>* getRLAgent() const
    {
      return control;
    }

    virtual T computeValueFunction(const Vector<T>* x) const
    {
      return control->computeValueFunction(x);
    }
};

template<class T>
class LearnerAgent: public RLAgent<T>
{
    typedef RLAgent<T> Base;
  private:
    const Action<T>* a_t;
    Vector<T>* absorbingState; // << this is the terminal state (absorbing state)
    Vector<T>* x_t;

  public:
    LearnerAgent(Control<T>* control) :
        RLAgent<T>(control), a_t(0), absorbingState(new PVector<T>(0)), x_t(0)
    {
    }

    virtual ~LearnerAgent()
    {
      delete absorbingState;
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
      const Action<T>* a_tp1 = Base::control->step(x_t, a_t,
          (step->endOfEpisode ? absorbingState : step->o_tp1), step->r_tp1, step->z_tp1);
      a_t = a_tp1;
      Vectors<T>::bufferedCopy(step->o_tp1, x_t);
      return a_t;
    }

    void reset()
    {
      Base::control->reset();
    }
};

template<class T>
class ControlAgent: public RLAgent<T>
{
    typedef RLAgent<T> Base;
  public:
    ControlAgent(Control<T>* control) :
        RLAgent<T>(control)
    {
    }

    virtual ~ControlAgent()
    {
    }

    const Action<T>* initialize(const TRStep<T>* step)
    {
      return Base::control->proposeAction(step->o_tp1);
    }

    const Action<T>* getAtp1(const TRStep<T>* step)
    {
      return Base::control->proposeAction(step->o_tp1);
    }

    void reset()
    {/*ControlAgent does not reset*/
    }
};

template<class T>
class RLProblem
{
  protected:
    DenseVector<T>* observations;
    TRStep<T>* output;
    Actions<T>* discreteActions;
    Actions<T>* continuousActions;
    Ranges<T>* observationRanges;
  public:
    RLProblem(int nbVars, int nbDiscreteActions, int nbContinuousActions) :
        observations(new PVector<T>(nbVars)), output(new TRStep<T>(nbVars)), discreteActions(
            new ActionArray<T>(nbDiscreteActions)), continuousActions(
            new ActionArray<T>(nbContinuousActions)), observationRanges(new Ranges<T>())
    {
    }

    virtual ~RLProblem()
    {
      delete observations;
      delete output;
      delete discreteActions;
      delete continuousActions;
      delete observationRanges;
    }

  public:
    virtual void initialize() =0;
    virtual void step(const Action<T>* action) =0;
    virtual void updateRTStep() =0;
    virtual bool endOfEpisode() const =0;
    virtual T r() const =0;
    virtual T z() const =0;

    virtual void draw() const
    {/*To output useful information*/
    }

    virtual Actions<T>* getDiscreteActions() const
    {
      return discreteActions;
    }

    virtual Actions<T>* getContinuousActions() const
    {
      return continuousActions;
    }

    virtual const DenseVector<T>* getObservations() const
    {
      return observations;
    }

    virtual TRStep<T>* getTRStep() const
    {
      return output;
    }

    virtual int dimension() const
    {
      return observations->dimension();
    }

    virtual const Ranges<T>* getObservationRanges() const
    {
      return observationRanges;
    }
};

template<class T>
class Simulator
{
  public:
    class Event
    {
        friend class Simulator;
      protected:
        int nbTotalTimeSteps;
        int nbEpisodeDone;
        T averageTimePerStep;
        T episodeR;
        T episodeZ;

      public:
        Event() :
            nbTotalTimeSteps(0), nbEpisodeDone(0), averageTimePerStep(0), episodeR(0), episodeZ(0)
        {
        }

        virtual ~Event()
        {
        }

        virtual void update() const=0;

    };
  protected:
    RLAgent<T>* agent;
    RLProblem<T>* problem;
    const Action<T>* agentAction;

    int maxEpisodeTimeSteps;
    int nbEpisodes;
    int nbRuns;
    int nbEpisodeDone;
    bool endingOfEpisode;
    bool verbose;

    Timer timer;
    T totalTimeInMilliseconds;

    std::vector<T> statistics;
    bool enableStatistics;

    bool enableTestEpisodesAfterEachRun;
    int maxTestEpisodesAfterEachRun;
  public:
    int timeStep;
    T episodeR;
    T episodeZ;
    std::vector<Event*> onEpisodeEnd;

    Simulator(RLAgent<T>* agent, RLProblem<T>* problem, const int& maxEpisodeTimeSteps,
        const int nbEpisodes = -1, const int nbRuns = -1) :
        agent(agent), problem(problem), agentAction(0), maxEpisodeTimeSteps(maxEpisodeTimeSteps), nbEpisodes(
            nbEpisodes), nbRuns(nbRuns), nbEpisodeDone(0), endingOfEpisode(false), verbose(true), totalTimeInMilliseconds(
            0), enableStatistics(false), enableTestEpisodesAfterEachRun(false), maxTestEpisodesAfterEachRun(
            20), timeStep(0), episodeR(0), episodeZ(0)
    {
    }

    ~Simulator()
    {
      onEpisodeEnd.clear();
    }

    void setVerbose(const bool& verbose)
    {
      this->verbose = verbose;
    }

    void setRuns(const int& nbRuns)
    {
      this->nbRuns = nbRuns;
    }

    void setEpisodes(const int& nbEpisodes)
    {
      this->nbEpisodes = nbEpisodes;
    }

    void setEnableStatistics(const bool& enableStatistics)
    {
      this->enableStatistics = enableStatistics;
    }

    void setTestEpisodesAfterEachRun(const bool& enableTestEpisodesAfterEachRun)
    {
      this->enableTestEpisodesAfterEachRun = enableTestEpisodesAfterEachRun;
    }

    void benchmark()
    {
      T xbar = std::accumulate(statistics.begin(), statistics.end(), 0.0f) / (T(statistics.size()));
      std::cout << std::endl;
      std::cout << "## Average: length=" << xbar;
      std::cout << std::endl;
      T sigmabar = T(0);
      for (typename std::vector<T>::const_iterator x = statistics.begin(); x != statistics.end();
          ++x)
        sigmabar += pow((*x - xbar), 2);
      sigmabar = sqrt(sigmabar) / T(statistics.size());
      T se/*standard error*/= sigmabar / sqrt(T(statistics.size()));
      std::cout << "## (+- 95%) =" << (se * 2);
      std::cout << std::endl;
      statistics.clear();
    }

    void step()
    {
      if (!agentAction)
      {
        /*Initialize the problem*/
        problem->initialize();
        /*Statistic variables*/
        timeStep = 0;
        episodeR = 0;
        episodeZ = 0;
        totalTimeInMilliseconds = 0;
        /*The episode is just started*/
        endingOfEpisode = false;
        problem->getTRStep()->setForcedEndOfEpisode(endingOfEpisode);
      }
      else
      {
        /*Step through the problem*/
        problem->step(agentAction);
      }

      if (!agentAction)
      {
        /*Initialize the control agent and get the first action*/
        agentAction = agent->initialize(problem->getTRStep());
      }
      else
      {
        TRStep<T>* step = problem->getTRStep();
        ++timeStep;
        episodeR += step->r_tp1;
        episodeZ += step->z_tp1;
        endingOfEpisode = step->endOfEpisode || (timeStep == maxEpisodeTimeSteps);
        //step->setForcedEndOfEpisode(endingOfEpisode);
        timer.start();
        agentAction = agent->getAtp1(step);
        timer.stop();
        totalTimeInMilliseconds += timer.getElapsedTimeInMilliSec();
      }

      if (endingOfEpisode/*The episode is just ended*/)
      {
        if (verbose)
        {
          T averageTimePerStep = totalTimeInMilliseconds / timeStep;
          std::cout << "{" << nbEpisodeDone << " [" << timeStep << " (" << episodeR << ", "
              << episodeZ << ", " << averageTimePerStep << ")]} ";
          //std::cout << ".";
          std::cout.flush();
        }
        if (enableStatistics)
          statistics.push_back(timeStep);
        ++nbEpisodeDone;
        /*Set the initial marker*/
        agentAction = 0;
        // Fire the events
        for (typename std::vector<Event*>::iterator iter = onEpisodeEnd.begin();
            iter != onEpisodeEnd.end(); ++iter)
        {
          Event* e = *iter;
          e->nbTotalTimeSteps = timeStep;
          e->nbEpisodeDone = nbEpisodeDone;
          e->averageTimePerStep = (totalTimeInMilliseconds / timeStep);
          e->episodeR = episodeR;
          e->episodeZ = episodeZ;
          e->update();
        }
      }

    }

    void runEpisodes()
    {
      do
      {
        step();
      } while (nbEpisodeDone < nbEpisodes);
    }

    void runEvaluate(const int& nbEpisodes = 20, const int& nbRuns = 1)
    {
      std::cout << "\n@@ Evaluate=" << enableTestEpisodesAfterEachRun << std::endl;
      RLAgent<T>* evaluateAgent = new ControlAgent<T>(agent->getRLAgent());
      Simulator<T>* runner = new Simulator<T>(evaluateAgent, problem, maxEpisodeTimeSteps,
          nbEpisodes, nbRuns);
      runner->run();
      delete evaluateAgent;
      delete runner;
    }

    void run()
    {
      if (verbose)
        std::cout << "## ControlLearner=" << typeid(*agent).name() << std::endl;
      for (int run = 0; run < nbRuns; run++)
      {
        if (verbose)
          std::cout << "\n@@ Run=" << run << std::endl;
        if (enableStatistics)
          statistics.clear();
        nbEpisodeDone = 0;
        // For each run
        agent->reset();
        runEpisodes();
        if (enableStatistics)
          benchmark();

        if (enableTestEpisodesAfterEachRun)
          runEvaluate(maxTestEpisodesAfterEachRun);
      }

    }

    bool isBeginingOfEpisode() const
    {
      return agentAction == 0;
    }

    bool isEndingOfEpisode() const
    {
      return endingOfEpisode;
    }

    bool isRunning() const
    {
      return (nbEpisodeDone < nbEpisodes) || (nbEpisodes == -1);
    }

    const RLProblem<T>* getRLProblem() const
    {
      return problem;
    }

    int getMaxEpisodeTimeSteps() const
    {
      return maxEpisodeTimeSteps;
    }

    void computeValueFunction(const char* outFile = "visualization/valueFunction.txt") const
    {
      if (problem->dimension() == 2) // only for two state variables
      {
        std::ofstream out(outFile);
        PVector<T> x_t(2);
        for (T x = 0; x <= 10; x += 0.1f)
        {
          for (T y = 0; y <= 10; y += 0.1f)
          {
            x_t.at(0) = x;
            x_t.at(1) = y;
            out << agent->computeValueFunction(&x_t) << " ";
          }
          out << std::endl;
        }
        out.close();
      }

      // draw
      problem->draw();
    }
};

}  // namespace RLLib

#endif /* RL_H_ */
