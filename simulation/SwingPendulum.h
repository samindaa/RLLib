/*
 * SwingPendulum.h
 *
 *  Created on: Aug 25, 2012
 *      Author: sam
 */

#ifndef SWINGPENDULUM_H_
#define SWINGPENDULUM_H_

#include <iostream>
#include "Env.h"

class SwingPendulum: public Env<float>
{
  protected:
    float uMax, xDot, theta, maxTheta, stepTime, maxXdot, mass, length, g,
        requiredUpTime, upRange;

    int upTime;
  public:
    SwingPendulum() :
        Env(2, 3), uMax(2.0), xDot(0), theta(0), maxTheta(M_PI), stepTime(0.01),
            maxXdot(M_PI_4 / stepTime), mass(1.0), length(1.0), g(9.8),
            requiredUpTime(10.0/*seconds*/), upRange(M_PI_4/*seconds*/),
            upTime(0)
    {
    }

    virtual ~SwingPendulum()
    {
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
      DenseVector<float>& vars = *__vars;
      //std::cout << (theta * 180 / M_PI) << " " << xDot << std::endl;
      vars[0] = theta / (2.0 * maxTheta) / 10;
      vars[1] = xDot / (2.0 * maxXdot) / 10;
    }
    void initialize()
    {
      upTime = 0;
      theta = M_PI_2;
      xDot = 0.0;
      normalize(theta);
      update();
    }

    void step(const Action& a)
    {
      float torque = uMax * (a.action() - 1.0);
      float thetaDot = -stepTime * xDot + mass * g * length * sin(theta)
          + torque;
      xDot += thetaDot;
      if (fabs(xDot) > maxXdot) xDot = sgn(xDot) * maxXdot;
      theta += xDot * stepTime;
      upTime = fabs(theta) > upRange ?
          0 : upTime + 1;
      normalize(theta);
      update();
    }
    bool endOfEpisode() const
    {
      return upTime + 1 >= requiredUpTime / stepTime; // 1000 steps
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
