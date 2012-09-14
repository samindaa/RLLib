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
    float position;
    float velocity;

    Range<float>* positionRange;
    Range<float>* velocityRange;
    Range<float>* actionRange;

    float targetPosition;
    float throttleFactor;

    float POS_WIDTH; // the tile width for position
    float VEL_WIDTH; // the tile width for velocity

    std::ofstream outfile;

  public:
    MCar2D() :
        Env<float>(2, 3, 1), position(0), velocity(0),
            positionRange(new Range<float>(-1.2, 0.6)),
            velocityRange(new Range<float>(-0.07, 0.07)),
            actionRange(new Range<float>(-1.0, 1.0)),
            targetPosition(positionRange->max()), throttleFactor(1.0),
            POS_WIDTH(positionRange->length() / 10.0),
            VEL_WIDTH(velocityRange->length() / 10.0)
    {
      discreteActions->push_back(0, actionRange->min());
      discreteActions->push_back(1, 0.0);
      discreteActions->push_back(2, actionRange->max());
      /*int i = 0;
       for (float p = actionRange->min(); p <= actionRange->max(); p += 0.02)
       discreteActions->push_back(i++, p);*/

      // subject to change
      continuousActions->push_back(0, 0.0);

      outfile.open("visualization/mcar.txt");
    }

    virtual ~MCar2D()
    {
      delete positionRange;
      delete velocityRange;
      delete actionRange;
      outfile.close();
    }

    void update()
    {
      DenseVector<float>& vars = *__vars;
      vars[0] = position / POS_WIDTH;
      vars[1] = velocity / VEL_WIDTH;

      if (outfile.is_open() && getOn()) outfile << position << std::endl;
    }

    // Profiles
    void initialize()
    {
      position = -0.5;
      velocity = 0.0;
      update();
    }

    void step(const Action& a)
    {
      float throttle = actionRange->bound(a.at()) * throttleFactor;
      velocity = velocityRange->bound(
          velocity + throttle * 0.001 + cos(3 * position) * (-0.0025));
      position += velocity;
      if (position < positionRange->min()) velocity = 0.0;
      position = positionRange->bound(position);
      update();
    }

    bool endOfEpisode() const
    {
      return (position >= targetPosition);
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
