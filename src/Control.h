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
 * Control.h
 *
 *  Created on: Aug 19, 2012
 *      Author: sam
 */

#ifndef CONTROL_H_
#define CONTROL_H_

#include "Action.h"
#include "Vector.h"
#include "Function.h"
#include "Policy.h"
#include "Predictor.h"

namespace RLLib
{

template<class T>
class Control: public ParameterizedFunction<T>
{
  public:
    virtual ~Control()
    {
    }
    virtual const Action<T>* initialize(const Vector<T>* x_0) =0;
    virtual void reset() =0;
    virtual const Action<T>* proposeAction(const Vector<T>* x) =0;
    virtual const Action<T>* step(const Vector<T>* x_t, const Action<T>* a_t, const Vector<T>* x_tp1,
        const double& r_tp1, const double& z_tp1) =0;
    virtual const double computeValueFunction(const Vector<T>* x) const =0;
    virtual const Predictor<T>* predictor() const =0;
};

template<class T>
class OnPolicyControlLearner: public Control<T>
{
  public:
    virtual ~OnPolicyControlLearner()
    {
    }
};

template<class T>
class OffPolicyControlLearner: public Control<T>
{
  public:
    virtual ~OffPolicyControlLearner()
    {
    }
};

template<class T>
class ActorOffPolicy: public ParameterizedFunction<T>
{
  public:
    virtual ~ActorOffPolicy()
    {
    }
    virtual void initialize() =0;
    virtual void reset() =0;
    virtual void update(const Representations<T>* phi_t, const Action<T>* a_t, double const& rho_t,
        const double& delta_t) =0;
    virtual PolicyDistribution<T>* policy() const =0;
    virtual const Action<T>* proposeAction(const Representations<T>* phi) =0;
    virtual double pi(const Action<T>* a) const =0;
};

template<class T>
class ActorOnPolicy: public ParameterizedFunction<T>
{
  public:
    virtual ~ActorOnPolicy()
    {
    }
    virtual void initialize() =0;
    virtual void reset() =0;
    virtual void update(const Representations<T>* phi_t, const Action<T>* a_t, double delta) =0;
    virtual PolicyDistribution<T>* policy() const =0;
    virtual const Action<T>* proposeAction(const Representations<T>* phi) =0;
};

} // namespace RLLib

#endif /* CONTROL_H_ */
