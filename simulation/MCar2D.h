/*
 * MCar2D.h
 *
 *  Created on: Jun 29, 2012
 *      Author: sam
 */

#ifndef MCAR2D_H_
#define MCAR2D_H_

#include <iostream>
#include <fstream>
#include "Env.h"

class MCar2D: public Env<float>
{
  protected:
    // Global variables:
    float mcar_position;
    float mcar_velocity;

    float mcar_min_position;
    float mcar_max_position;
    float mcar_max_velocity; // the negative of this in the minimum velocity
    float mcar_goal_position;

    float POS_WIDTH; // the tile width for position
    float VEL_WIDTH; // the tile width for velocity

    std::ofstream outfile;

  public:
    MCar2D() :
        Env<float>(2, 3), mcar_position(0), mcar_velocity(0),
            mcar_min_position(-1.2), mcar_max_position(0.6),
            mcar_max_velocity(0.07), mcar_goal_position(0.5),
            POS_WIDTH(1.7 / 10.0), VEL_WIDTH(0.14 / 10.0)
    {
      for (unsigned int a = 0; a < actions->getNumActions(); a++)
        actions->add(a, a);
      outfile.open("mcar.txt");
    }

    virtual ~MCar2D()
    {
      outfile.close();
    }

    void update()
    {
      DenseVector<float>& vars = *__vars;
      vars[0] = mcar_position / POS_WIDTH;
      vars[1] = mcar_velocity / VEL_WIDTH;

      if (outfile.is_open() && getOn())
        outfile << mcar_position  << std::endl;
    }

    // Profiles
    void initialize()
    {
      mcar_position = -0.5;
      mcar_velocity = 0.0;
      update();
    }

    void step(const Action& a)
    {
      mcar_velocity += (a.at() - 1) * 0.001
          + ::cos(3 * mcar_position) * (-0.0025);
      if (mcar_velocity > mcar_max_velocity) mcar_velocity = mcar_max_velocity;
      if (mcar_velocity < -mcar_max_velocity) mcar_velocity =
          -mcar_max_velocity;
      mcar_position += mcar_velocity;
      if (mcar_position > mcar_max_position) mcar_position = mcar_max_position;
      if (mcar_position < mcar_min_position) mcar_position = mcar_min_position;
      if (mcar_position == mcar_min_position && mcar_velocity < 0)
        mcar_velocity = 0;

      update();
    }

    bool endOfEpisode() const
    {
      return (mcar_position >= mcar_goal_position);
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

#endif /* MCAR2D_H_ */
