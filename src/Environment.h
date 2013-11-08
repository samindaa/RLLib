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
 * Environment.h
 *
 *  Created on: Aug 28, 2013
 *      Author: sam
 */

#ifndef ENVIRONMENT_H_
#define ENVIRONMENT_H_

#include "Vector.h"
#include "Action.h"
#include "Math.h"

namespace RLLib
{

template<class T = double>
class TRStep
{
  public:
    DenseVector<T>* o_tp1;
    double r_tp1;
    double z_tp1;
    bool endOfEpisode;
    TRStep(const int& nbVars) :
        o_tp1(new PVector<T>(nbVars)), r_tp1(0.0f), z_tp1(0.0f), endOfEpisode(false)
    {
    }

    void updateRTStep(const double& r_tp1, const double& z_tp1, const bool& endOfEpisode)
    {
      this->r_tp1 = r_tp1;
      this->z_tp1 = z_tp1;
      this->endOfEpisode = endOfEpisode;
    }

    ~TRStep()
    {
      delete o_tp1;
    }
};

template<class T = double>
class Environment
{
  protected:
    DenseVector<T>* observations;
    DenseVector<T>* resolutions;
    TRStep<T>* output;
    ActionList<T>* discreteActions;
    ActionList<T>* continuousActions;
  public:
    Environment(int nbVars, int nbDiscreteActions, int nbContinuousActions) :
        observations(new PVector<T>(nbVars)), resolutions(new PVector<T>(nbVars)), output(
            new TRStep<T>(nbVars)), discreteActions(new GeneralActionList<T>(nbDiscreteActions)), continuousActions(
            new GeneralActionList<T>(nbContinuousActions))
    {
    }

    virtual ~Environment()
    {
      delete observations;
      delete resolutions;
      delete output;
      delete discreteActions;
      delete continuousActions;
    }

  public:
    virtual void initialize() =0;
    virtual void step(const Action<T>* action) =0;
    virtual void updateRTStep() =0;
    virtual bool endOfEpisode() const =0;
    virtual float r() const =0;
    virtual float z() const =0;

    virtual void draw() const
    {/*To output useful information*/
    }

    virtual ActionList<T>* getDiscreteActionList() const
    {
      return discreteActions;
    }

    virtual ActionList<T>* getContinuousActionList() const
    {
      return continuousActions;
    }

    virtual const TRStep<T>* getTRStep() const
    {
      return output;
    }

    virtual const DenseVector<T>* getObservations() const
    {
      return observations;
    }

    virtual const DenseVector<T>* getResolutions() const
    {
      return resolutions;
    }

    virtual void setResolution(const double& resolution)
    {
      for (int i = 0; i < resolutions->dimension(); i++)
        resolutions->at(i) = resolution;
    }

    virtual int dimension() const
    {
      return observations->dimension();
    }
};

}  // namespace RLLib

#endif /* ENVIRONMENT_H_ */
