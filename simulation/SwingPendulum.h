/*
 * SwingPendulum.h
 *
 *  Created on: Aug 25, 2012
 *      Author: sam
 */

#ifndef SWINGPENDULUM_H_
#define SWINGPENDULUM_H_

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include "Env.h"

class SwingPendulum: public Env<float>
{
  protected:
    float uMax, stepTime, theta, velocity, maxVelocity;

    Range<float>* actionRange;
    Range<float>* thetaRange;
    Range<float>* velocityRange;

    float mass, length, g, requiredUpTime, upRange;

    int upTime;
    std::ofstream outfile;
  public:
    SwingPendulum() :
        Env(2, 3, 1), uMax(2.0), stepTime(0.01), theta(0), velocity(0),
            maxVelocity(M_PI_4 / stepTime),
            actionRange(new Range<float>(-uMax, uMax)),
            thetaRange(new Range<float>(-M_PI, M_PI)),
            velocityRange(new Range<float>(-maxVelocity, maxVelocity)),
            mass(1.0), length(1.0), g(9.8), requiredUpTime(10.0 /*seconds*/),
            upRange(M_PI_4 /*seconds*/), upTime(0)
    {

      discreteActions->push_back(0, actionRange->min());
      discreteActions->push_back(1, 0.0);
      discreteActions->push_back(2, actionRange->max());

      // subject to change
      continuousActions->push_back(0, 0.0);

      outfile.open("swingPendulum.txt");
    }

    virtual ~SwingPendulum()
    {
      delete actionRange;
      delete thetaRange;
      delete velocityRange;
      outfile.close();
    }

  private:
    double normalize(float data)
    {
      if (data < M_PI && data >= -M_PI) return data;
      float ndata = data - ((int) (data / 2.0 * M_PI)) * 2.0 * M_PI;
      if (ndata >= M_PI) ndata -= 2.0 * M_PI;
      else if (ndata < -M_PI) ndata += 2.0 * M_PI;
      return ndata;
    }

  public:
    void update()
    {
      if (outfile.is_open() && getOn())
        outfile << (theta * 180 / M_PI) << " " << cos(theta) << std::endl;

      DenseVector<float>& vars = *__vars;
      //std::cout << (theta * 180 / M_PI) << " " << xDot << std::endl;
      vars[0] = theta / thetaRange->length() / 10;
      vars[1] = velocity / velocityRange->length() / 10;
    }
    void initialize()
    {
      upTime = 0;
      //theta = (2.0 * drand48() - 1.0) * M_PI; //M_PI_2;
      theta = M_PI_2;
      velocity = 0.0;
      normalize(theta);
      update();
    }

    void step(const Action& a)
    {
      float torque = actionRange->bound(a.at());
      float thetaAcc = (-stepTime * velocity) + mass * g * length * sin(theta)
          + torque;
      velocity = velocityRange->bound(velocity + thetaAcc);
      theta += velocity * stepTime;
      normalize(theta);
      upTime = fabs(theta) > upRange ?
          0 : upTime + 1;

      update();
    }
    bool endOfEpisode() const
    {
      return false;
      //return upTime + 1 >= requiredUpTime / stepTime; // 1000 steps
    }
    float r() const
    {
      return cos(theta);
    }
    float z() const
    {
      return 0.0;
    }

};

#endif /* SWINGPENDULUM_H_ */
