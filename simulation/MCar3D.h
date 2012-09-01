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

//@@>> TODO: a whole bunch of cleaning to do ...
class MCar3D: public Env<float>
{
  protected:
    float mcar_Xposition;
    float mcar_Yposition;
    float mcar_Xvelocity;
    float mcar_Yvelocity;

    float mcar_min_position;
    float mcar_max_position;
    float mcar_goal_position;
    float mcar_max_velocity;

    float offset;

    float mcar_Xstep;
    float mcar_Ystep;
    float mcar_Dxstep;
    float mcar_Dystep;

  public:
    MCar3D() :
        Env<float>(4, 5, 1), mcar_Xposition(0), mcar_Yposition(0),
            mcar_Xvelocity(0), mcar_Yvelocity(0), mcar_min_position(-1.2),
            mcar_max_position(0.6), mcar_goal_position(0.5),
            mcar_max_velocity(0.07), offset(0), mcar_Xstep(1.7 / 10.0),
            mcar_Ystep(1.7 / 10.0), mcar_Dxstep(0.14 / 10.0),
            mcar_Dystep(0.14 / 10.0)
    {

      for (unsigned int a = 0; a < discreteActions->dimension(); a++)
        discreteActions->push_back(a, a);
      // not used
      continuousActions->push_back(0, 0.0);
    }

    virtual ~MCar3D()
    {
    }

  private:

    void set_initial_position_random()
    {
      mcar_Xposition = mcar_min_position
          + ((double) rand() / ((double) RAND_MAX + 1))
              * ((mcar_max_position - 0.2) - mcar_min_position);
      mcar_Yposition = mcar_min_position
          + ((double) rand() / ((double) RAND_MAX + 1))
              * ((mcar_max_position - 0.2) - mcar_min_position);
      mcar_Xvelocity = 0.0;
      mcar_Yvelocity = 0.0;
    }

    void set_initial_position_at_bottom()
    {
      mcar_Xposition = 0.; //-M_PI / 6.0 + offset;
      mcar_Yposition = 0.; //-M_PI / 6.0 + offset;
      mcar_Xvelocity = 0.;
      mcar_Yvelocity = 0.;
    }

    void update_velocity(const int& act)
    {

      switch (act)
      {
      case 0:
        mcar_Xvelocity += cos(3 * mcar_Xposition) * (-0.0025);
        mcar_Yvelocity += cos(3 * mcar_Yposition) * (-0.0025);
        break;
      case 1:
        mcar_Xvelocity += -0.001 + cos(3 * mcar_Xposition) * (-0.0025);
        mcar_Yvelocity += cos(3 * mcar_Yposition) * (-0.0025);
        break;
      case 2:
        mcar_Xvelocity += +0.001 + cos(3 * mcar_Xposition) * (-0.0025);
        mcar_Yvelocity += cos(3 * mcar_Yposition) * (-0.0025);
        break;
      case 3:
        mcar_Xvelocity += cos(3 * mcar_Xposition) * (-0.0025);
        mcar_Yvelocity += -0.001 + cos(3 * mcar_Yposition) * (-0.0025);
        break;
      case 4:
        mcar_Xvelocity += cos(3 * mcar_Xposition) * (-0.0025);
        mcar_Yvelocity += +0.001 + cos(3 * mcar_Yposition) * (-0.0025);
        break;
      }

      //mcar_Xvelocity *= get_gaussian(1.0,std_dev_eff);
      //mcar_Yvelocity *= get_gaussian(1.0,std_dev_eff);

      if (mcar_Xvelocity > mcar_max_velocity) mcar_Xvelocity =
          mcar_max_velocity;
      else if (mcar_Xvelocity < -mcar_max_velocity) mcar_Xvelocity =
          -mcar_max_velocity;
      if (mcar_Yvelocity > mcar_max_velocity) mcar_Yvelocity =
          mcar_max_velocity;
      else if (mcar_Yvelocity < -mcar_max_velocity) mcar_Yvelocity =
          -mcar_max_velocity;

    }

    void update_position()
    {
      mcar_Xposition += mcar_Xvelocity;
      mcar_Yposition += mcar_Yvelocity;

      if (mcar_Xposition > mcar_max_position) mcar_Xposition =
          mcar_max_position;
      if (mcar_Xposition < mcar_min_position) mcar_Xposition =
          mcar_min_position;
      if (mcar_Xposition == mcar_max_position && mcar_Xvelocity > 0)
        mcar_Xvelocity = 0;
      if (mcar_Xposition == mcar_min_position && mcar_Xvelocity < 0)
        mcar_Xvelocity = 0;

      if (mcar_Yposition > mcar_max_position) mcar_Yposition =
          mcar_max_position;
      if (mcar_Yposition < mcar_min_position) mcar_Yposition =
          mcar_min_position;
      if (mcar_Yposition == mcar_max_position && mcar_Yvelocity > 0)
        mcar_Yvelocity = 0;
      if (mcar_Yposition == mcar_min_position && mcar_Yvelocity < 0)
        mcar_Yvelocity = 0;
    }

    void update()
    {
      DenseVector<float>& vars = *__vars;
      vars[0] = mcar_Xposition / mcar_Xstep;
      vars[1] = mcar_Yposition / mcar_Ystep;
      vars[2] = mcar_Xvelocity / mcar_Dxstep;
      vars[3] = mcar_Yvelocity / mcar_Dystep;
    }

  public:
    void initialize()
    {
      set_initial_position_at_bottom();
//      set_initial_position_random();
      update();
    }

    void step(const Action& a)
    {
      update_velocity(a.at());
      update_position();
      update();
    }

    bool endOfEpisode() const
    {
      return ((mcar_Xposition >= mcar_goal_position)
          && (mcar_Yposition >= mcar_goal_position));
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
