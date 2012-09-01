/*
 * ContinuousGridworld.h
 *
 *  Created on: Sep 1, 2012
 *      Author: sam
 */

#ifndef CONTINUOUSGRIDWORLD_H_
#define CONTINUOUSGRIDWORLD_H_

#include <fstream>
#include <iostream>
#include "Env.h"

class ContinuousGridworld: public Env<float>
{
  protected:
    Range<float>* observationRange;
    Range<float>* actionRange;
    float absoluteNoise;

  public:
    ContinuousGridworld() :
        Env<float>(2, 2 * 2 + 1, 1), observationRange(new Range<float>(0, 1.0)),
            actionRange(new Range<float>(-0.05, 0.05)),
            absoluteNoise(0.1 * 5.0 / 2.0)
    {
      // discrete actions
      for (unsigned int i = 0; i < discreteActions->dimension(); i++)
      {
        for (int k = 0; k < 2; k++)
          discreteActions->push_back(i, 0);
      }
      for (unsigned int i = 0; i < discreteActions->dimension() - 1; i++)
      {
        unsigned int dimension = i / 2;
        unsigned int dimensionAction = i % 2;
        if (dimensionAction == 0) discreteActions->update(i, dimension, -1.0);
        else discreteActions->update(i, dimension, 1.0);
      }

      // continuous actions are not setup for this problem
    }

    virtual ~ContinuousGridworld()
    {
      delete observationRange;
      delete actionRange;
    }

    void initialize()
    {
      __vars->at(0) = 0.2;
      __vars->at(1) = 0.4;
    }

    void update()
    { // nothing
    }

    void step(const Action& action)
    {
      float noise = drand48() * absoluteNoise - (absoluteNoise / 2.0);
      for (int i = 0; i < __vars->dimension(); i++)
        __vars->at(i) = observationRange->bound(
            __vars->at(i) + actionRange->bound(action.at(i) + noise));
    }

    bool endOfEpisode() const
    {
      float distance = 0;
      for (int i = 0; i < __vars->dimension(); i++)
        distance += fabs(1.0 - __vars->at(i));
      return distance < 0.1;
    }

    float N(const float& p, const float& mu, const float& sigma) const
    {
      return exp(-pow((p - mu), 2) / (2.0 * pow(sigma, 2)))
          / (sigma * sqrt(2.0 * M_PI));
    }

    float r() const
    {
      float px = __vars->at(0);
      float py = __vars->at(1);
      return -1.0
          - 2.0
              * (N(px, 0.3, 0.1) * N(py, 0.6, 0.03)
                  + N(px, 0.4, 0.03) * N(py, 0.5, 0.1)
                  + N(px, 0.8, 0.03) * N(py, 0.9, 0.1));
    }

    float z() const
    {
      return 0;
    }

    void draw() const
    {
      std::ofstream out("env.txt");
      for (float px = observationRange->min(); px <= observationRange->max();
          px += 0.01)
      {
        for (float py = observationRange->min(); py <= observationRange->max();
            py += 0.01)
          out
              << (N(px, 0.3, 0.1) * N(py, 0.6, 0.03)
                  + N(px, 0.4, 0.03) * N(py, 0.5, 0.1)
                  + N(px, 0.8, 0.03) * N(py, 0.9, 0.1)) << " ";

        out << std::endl;
      }
      out.close();
    }
};

#endif /* CONTINUOUSGRIDWORLD_H_ */
