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

namespace RLLib
{

template<class T, class O>
class Control : public ParameterizedFunction<T>
{
  public:
    virtual ~Control()
    {
    }
    virtual const Action& initialize(const DenseVector<O>& x_0) =0;
    virtual void reset() =0;
    virtual const Action& proposeAction(const DenseVector<O>& x) =0;
    virtual const Action& step(const DenseVector<O>& x_t, const Action& a_t,
        const DenseVector<O>& x_tp1, const double& r_tp1, const double& z_tp1) =0;
    virtual const double computeValueFunction(const DenseVector<O>& x) const =0;
};

template<class T, class O>
class OnPolicyControlLearner: public Control<T, O>
{
  public:
    virtual ~OnPolicyControlLearner()
    {
    }
};

template<class T, class O>
class OffPolicyControlLearner: public Control<T, O>
{
  public:
    virtual ~OffPolicyControlLearner()
    {
    }
};

template<class T, class O>
class ActorOffPolicy: public ParameterizedFunction<T>
{
  public:
    virtual ~ActorOffPolicy()
    {
    }
    virtual void initialize() =0;
    virtual void reset() =0;
    virtual void update(const Representations<T>& xas_t, const Action& a_t, double const& rho_t,
        double const& gamma_t, double delta_t) =0;
    virtual PolicyDistribution<T>& policy() const =0;
    virtual const Action& proposeAction(const Representations<T>& xas) =0;
    virtual double pi(const Action& a) const =0;
};

template<class T, class O>
class ActorOnPolicy: public ParameterizedFunction<T>
{
  public:
    virtual ~ActorOnPolicy()
    {
    }
    virtual void initialize() =0;
    virtual void reset() =0;
    virtual void update(const Representations<T>& xas_t, const Action& a_t, double delta) =0;
    virtual PolicyDistribution<T>& policy() const =0;
    virtual const Action& proposeAction(const Representations<T>& xas) =0;
};

} // namespace RLLib

#endif /* CONTROL_H_ */
