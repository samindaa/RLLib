/*
 * Mcar3D.h
 *
 *  Created on: Jun 29, 2012
 *      Author: sam
 */

#ifndef MCAR3D_H_
#define MCAR3D_H_

#include "Env.h"
#include <cstdlib>
/******************************************************************************
 *  Author: Sam Abeyruwan
 *
 *  Based on MontainCar3DSym.cc, created by Matthew Taylor
 *           (Based on MountainCar.cc, created by Adam White,
 *            created on March 29 2007.)
 *
 *    Episodic Task
 *    Reward: -1 per step
 *    Actions: Discrete
 *          0 - coast
 *          1 - left
 *          2 - right
 *          3 - down
 *          4 - up
 *
 *    State: 3D Continuous
 *          car's x-position (-1.2 to .6)
 *          car's y-position (-1.2 to .6)
 *          car's x-velocity (-.07 to .07)
 *          car's y-velocity (-.07 to .07)
 *
 ******************************************************************************
 */

class MCar3D: public Env<float>
{
  protected:

    float xposition;
    float yposition;
    float xvelocity;
    float yvelocity;

    float offset;
    float targetPosition;

    Range<float>* positionRange;
    Range<float>* velocityRange;

    float xstep;
    float ystep;
    float dxstep;
    float dystep;

    std::ofstream out;

  public:
    MCar3D() :
        Env<float>(4, 5, 1), xposition(0), yposition(0), xvelocity(0),
            yvelocity(0), offset(0), targetPosition(0.5),
            positionRange(new Range<float>(-1.2, 0.5)),
            velocityRange(new Range<float>(-0.07, 0.07)),
            xstep(positionRange->length() / 10.0),
            ystep(positionRange->length() / 10.0),
            dxstep(velocityRange->length() / 10.0),
            dystep(velocityRange->length() / 10.0)
    {

      for (unsigned int a = 0; a < discreteActions->dimension(); a++)
        discreteActions->push_back(a, a);
      // not used
      continuousActions->push_back(0, 0.0);
      out.open("visualization/mcar3D.txt");
    }

    virtual ~MCar3D()
    {
      delete positionRange;
      delete velocityRange;
      out.close();
    }

  private:

    void set_initial_position_random()
    {
      xposition = positionRange->min()
          + drand48() * ((positionRange->max() - 0.2) - positionRange->min());
      yposition = positionRange->min()
          + drand48() * ((positionRange->max() - 0.2) - positionRange->min());
      xvelocity = 0.0;
      yvelocity = 0.0;
    }

    void set_initial_position_at_bottom()
    {
      xposition = 0.; //-M_PI / 6.0 + offset;
      yposition = 0.; //-M_PI / 6.0 + offset;
      xvelocity = 0.;
      yvelocity = 0.;
    }

    void update_velocity(const Action& act)
    {

      switch (act)
      {
      case 0:
        xvelocity += cos(3 * xposition) * (-0.0025);
        yvelocity += cos(3 * yposition) * (-0.0025);
        break;
      case 1:
        xvelocity += -0.001 + cos(3 * xposition) * (-0.0025);
        yvelocity += cos(3 * yposition) * (-0.0025);
        break;
      case 2:
        xvelocity += +0.001 + cos(3 * xposition) * (-0.0025);
        yvelocity += cos(3 * yposition) * (-0.0025);
        break;
      case 3:
        xvelocity += cos(3 * xposition) * (-0.0025);
        yvelocity += -0.001 + cos(3 * yposition) * (-0.0025);
        break;
      case 4:
        xvelocity += cos(3 * xposition) * (-0.0025);
        yvelocity += +0.001 + cos(3 * yposition) * (-0.0025);
        break;
      }

      //xvelocity *= get_gaussian(1.0,std_dev_eff);
      //yvelocity *= get_gaussian(1.0,std_dev_eff);
      xvelocity = velocityRange->bound(xvelocity);
      yvelocity = velocityRange->bound(yvelocity);
    }

    void update_position()
    {
      xposition += xvelocity;
      yposition += yvelocity;
      xposition = positionRange->bound(xposition);
      yposition = positionRange->bound(yposition);
    }

    void update()
    {
      DenseVector<float>& vars = *__vars;
      vars[0] = xposition / xstep;
      vars[1] = yposition / ystep;
      vars[2] = xvelocity / dxstep;
      vars[3] = yvelocity / dystep;
      if (out.is_open() && getOn())
        out << xposition << " " << yposition << std::endl;
    }

  public:
    void initialize()
    {
//      set_initial_position_at_bottom();
      set_initial_position_random();
      update();
    }

    void step(const Action& a)
    {
      update_velocity(a);
      update_position();
      update();
    }

    bool endOfEpisode() const
    {
      return ((xposition >= targetPosition) && (yposition >= targetPosition));
    }

    float r() const
    {
      return -1.0;
    }

    float z() const
    {
      return 0;
    }

};

#endif /* MCAR3D_H_ */
