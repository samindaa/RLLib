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
 * Env.h
 *
 *  Created on: Jun 29, 2012
 *      Author: sam
 */

#ifndef ENV_H_
#define ENV_H_

#include "Vector.h"
#include "Action.h"
#include "Math.h"

using namespace RLLib;

/*
 * Represent an environment, a plant or an simulation.
 */
template<class O>
class Env
{
  protected:
    DenseVector<O>* __vars;
    ActionList* discreteActions;
    ActionList* continuousActions;
    bool itsOn;

  public:
    Env(int numVars, int numDiscreteActions, int numContinuousActions) :
        __vars(new DenseVector<O>(numVars)), discreteActions(
            new GeneralActionList(numDiscreteActions)), continuousActions(
            new GeneralActionList(numContinuousActions)), itsOn(false)
    {
    }

    virtual ~Env()
    {
      delete __vars;
      delete discreteActions;
      delete continuousActions;
    }

    virtual void initialize() =0;
    virtual void update() =0;
    virtual void step(const Action& action) =0;
    virtual bool endOfEpisode() const =0;
    virtual float r() const =0;
    virtual float z() const =0;
    virtual void draw() const
    {
    }

    virtual void setOn(const bool& itsOn)
    {
      this->itsOn = itsOn;
    }

    virtual bool getOn() const
    {
      return itsOn;
    }

    ActionList& getDiscreteActionList() const
    {
      return *discreteActions;
    }

    ActionList& getContinuousActionList() const
    {
      return *continuousActions;
    }

    const DenseVector<O>& getVars() const
    {
      return *__vars;
    }

};

#endif /* ENV_H_ */
