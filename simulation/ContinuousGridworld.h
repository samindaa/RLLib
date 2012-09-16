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
    DenseVector<float>* observations;
    std::ofstream outpath;

  public:
    ContinuousGridworld() :
        Env<float>(2, 2 * 2 + 1, 1), observationRange(new Range<float>(0, 1.0)),
            actionRange(new Range<float>(-0.05, 0.05)), absoluteNoise(0.025),
            observations(new DenseVector<float>(2))
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
        if (dimensionAction == 0)
          discreteActions->update(i, dimension, -1.0);
        else
          discreteActions->update(i, dimension, 1.0);
      }

      // continuous actions are not setup for this problem
      outpath.open("visualization/continuousGridworldPath.txt");
    }

    virtual ~ContinuousGridworld()
    {
      delete observationRange;
      delete actionRange;
      delete observations;
      outpath.close();
    }

    void initialize()
    {
      observations->at(0) = 0.2;
      observations->at(1) = 0.4;
      update();
    }

    void update()
    { // nothing
      // unit generalization
      for (int i = 0; i < __vars->dimension(); i++)
        __vars->at(i) = observations->at(i) * 10.0;
      //std::cout << __vars->at(0) << " " << __vars->at(1) << " || ";
      if (getOn())
      {
        for (int i = 0; i < __vars->dimension(); i++)
          outpath << __vars->at(i) << " ";
        outpath << std::endl;
      }
    }

    void step(const Action& action)
    {
      float noise = drand48() * absoluteNoise - (absoluteNoise / 2.0);
      for (int i = 0; i < observations->dimension(); i++)
        observations->at(i) = observationRange->bound(
            observations->at(i) + actionRange->bound(action.at(i) + noise));
      update();
    }

    bool endOfEpisode() const
    {
      // L1-norm
      float distance = 0;
      for (int i = 0; i < observations->dimension(); i++)
        distance += fabs(1.0 - observations->at(i));
      return distance < 0.1;
    }

    float N(const float& p, const float& mu, const float& sigma) const
    {
      return Gaussian::probability(p, mu, sigma);
    }

    float r() const
    {
      float px = observations->at(0);
      float py = observations->at(1);
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
      std::ofstream oute("visualization/continuousGridworld.txt");
      for (float px = observationRange->min(); px <= observationRange->max();
          px += 0.01)
      {
        for (float py = observationRange->min(); py <= observationRange->max();
            py += 0.01)
          oute
              << (N(px, 0.3, 0.1) * N(py, 0.6, 0.03)
                  + N(px, 0.4, 0.03) * N(py, 0.5, 0.1)
                  + N(px, 0.8, 0.03) * N(py, 0.9, 0.1)) << " ";

        oute << std::endl;
      }
      oute.close();
    }
};

#endif /* CONTINUOUSGRIDWORLD_H_ */
