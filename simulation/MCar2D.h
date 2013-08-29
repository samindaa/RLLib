/*
 * Copyright 2013 Saminda Abeyruwan (saminda@cs.miami.edu)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * MCar2D.h
 *
 *  Created on: Jun 29, 2012
 *      Author: sam
 */

#ifndef MCAR2D_H_
#define MCAR2D_H_

#include <iostream>
#include <fstream>
#include "Environment.h"

class MCar2D: public Environment<float>
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

    std::ofstream outfile;

  public:
    MCar2D() :
        Environment<float>(2, 3, 1), position(0), velocity(0), positionRange(
            new Range<float>(-1.2, 0.6)), velocityRange(new Range<float>(-0.07, 0.07)), actionRange(
            new Range<float>(-1.0, 1.0)), targetPosition(positionRange->max()), throttleFactor(1.0)
    {
      discreteActions->push_back(0, actionRange->min());
      discreteActions->push_back(1, 0.0);
      discreteActions->push_back(2, actionRange->max());

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

    void updateRTStep()
    {
      DenseVector<float>& vars = *output->o_tp1;
      vars[0] = (position - positionRange->min()) * 10.0 / positionRange->length();
      vars[1] = (velocity - velocityRange->min()) * 10.0 / velocityRange->length();
      output->updateRTStep(r(), z(), endOfEpisode());

      //if (outfile.is_open() && getOn())
      //  outfile << position << std::endl;
      if (outfile.is_open() && getOn())
        outfile << vars[0] << " " << vars[1] << endl;
    }

    // Profiles
    void initialize()
    {
      position = -0.5;
      velocity = 0.0;
      updateRTStep();
    }

    void step(const Action& a)
    {
      float throttle = actionRange->bound(a.at()) * throttleFactor;
      velocity = velocityRange->bound(velocity + throttle * 0.001 + cos(3 * position) * (-0.0025));
      position += velocity;
      if (position < positionRange->min())
        velocity = 0.0;
      position = positionRange->bound(position);
      updateRTStep();
    }

    bool endOfEpisode() const
    {
      return (position >= targetPosition);
    }

    float r() const
    {
      return endOfEpisode() ? 0 : -1.0;
    }

    float z() const
    {
      return 0;
    }

};

#endif /* MCAR2D_H_ */
