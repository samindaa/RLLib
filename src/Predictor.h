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
 * Predictor.h
 *
 *  Created on: Aug 19, 2012
 *      Author: sam
 */

#ifndef PREDICTOR_H_
#define PREDICTOR_H_

#include "Vector.h"

namespace RLLib
{

template<class T>
class Predictor
{
  public:
    virtual ~Predictor()
    {
    }
    virtual int dimension() const=0;
    virtual double predict(const SparseVector<T>& x) const =0;
    virtual double initialize() =0;
    virtual void reset() =0;

    virtual void persist(const std::string& f) const =0;
    virtual void resurrect(const std::string& f) =0;
};

template<class T>
class OnPolicyTD: public Predictor<T>
{
  public:
    virtual ~OnPolicyTD()
    {
    }
    virtual double update(const SparseVector<T>& x_t, const SparseVector<T>& x_tp1,
        double r_tp1) =0;
};

template<class T>
class GVF: public Predictor<T>
{
  public:
    virtual ~GVF()
    {
    }
    virtual double update(const SparseVector<T>& x_t, const SparseVector<T>& x_tp1,
        const double& gamma_tp1, const double& lambda_tp1, const double& rho_t, const double& r_tp1,
        const double& z_tp1) =0;
};

} // namespace RLLib

#endif /* PREDICTOR_H_ */
