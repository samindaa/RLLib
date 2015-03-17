/*
 * UnderwaterVehicle.h
 *
 *  Created on: Aug 15, 2014
 *      Author: sam
 */

#ifndef UNDERWATERVEHICLE_H_
#define UNDERWATERVEHICLE_H_

/**
 * Plant designed according to the paper
 *
 * Reinforcement learning in feedback control
 * Challenges and benchmarks from technical process control
 * Mach Learn (2011) 84:137–169
 * DOI 10.1007/s10994-011-5235-x
 */

#include "RL.h"
#include "util/RK4.h"

using namespace RLLib;

class UnderwaterVehicle: public RLProblem<double>
{
  protected:
    class Dynammic: public RK4
    {

      public:
        Dynammic(const double& dt) :
            RK4(1, dt)
        {
        }

        void f(const double& time, const Action<double>* action, const Vector<double>* x,
            Vector<double>* x_dot)
        {
          const double u = action->getEntry(0);
          const double v = x->getEntry(0);
          const double abs_v = fabs(v);
          const double c_v = 1.2f + 0.2f * sin(abs_v);
          const double m_v = 3.0f + 1.5f * sin(abs_v);
          const double k_v_u = -0.5f * tanh((fabs(c_v * v * abs_v - u) - 30.0f) * 0.1f) + 0.5f;
          x_dot->setEntry(0, (u * k_v_u - c_v * v * abs_v) / m_v);
        }

    };

  private:
    Range<double>* thrustRange;
    Range<double>* velocityRange;

    double dt;
    double C;
    double mu;
    double setPoint; // m/s
    int timeSteps;

    Dynammic* dynamic;

  public:
    UnderwaterVehicle(Random<double>* random) :
        RLProblem<double>(random, 1, 5, 1), thrustRange(new Range<double>(-30, 30)), //
        velocityRange(new Range<double>(-5, 5)), dt(0.03), C(0.01), mu(0.3), setPoint(4), //
        timeSteps(800), dynamic(new Dynammic(dt))
    {
      discreteActions->push_back(0, thrustRange->min());
      discreteActions->push_back(1, thrustRange->min() / 2.0f);
      discreteActions->push_back(2, 0.0f);
      discreteActions->push_back(3, thrustRange->max() / 2.0f);
      discreteActions->push_back(4, thrustRange->max());

      continuousActions->push_back(0, 0);

    }

    virtual ~UnderwaterVehicle()
    {
      delete thrustRange;
      delete velocityRange;
      delete dynamic;
    }

    void initialize()
    {
      dynamic->initialize();
      dynamic->vec()->setEntry(0, -4.0); // m/s // TODO random
    }

    void step(const Action<double>* action)
    {
      for (int i = 0; i < 2; i++)
        dynamic->step(action);
    }

    void updateTRStep()
    {
      output->o_tp1->setEntry(0, velocityRange->toUnit(dynamic->vec()->getEntry(0)));
      output->observation_tp1->setEntry(0, dynamic->vec()->getEntry(0));
      // TODO: only with one variable first
    }

    bool endOfEpisode() const
    {
      return dynamic->getTimeSteps() > timeSteps;
    }

    double r() const
    {
      return fabs(setPoint - dynamic->vec()->getEntry(0)) < mu ? 0.0f : -C;
    }

    double z() const
    {
      return 0;
    }
};

#endif /* UNDERWATERVEHICLE_H_ */
