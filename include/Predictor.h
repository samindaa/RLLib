/*
 * Copyright 2015 Saminda Abeyruwan (saminda@cs.miami.edu)
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
#include "Function.h"

namespace RLLib
{

  template<typename T>
  class Predictor
  {
    public:
      virtual ~Predictor()
      {
      }
      virtual T predict(const Vector<T>* x) const =0;
      virtual Vector<T>* weights() const =0;
  };

  template<typename T>
  class OnPolicyTD: public virtual Predictor<T>, public virtual LinearLearner<T>
  {
    public:
      virtual ~OnPolicyTD()
      {
      }
      virtual T update(const Vector<T>* x_t, const Vector<T>* x_tp1, const T& r_tp1) =0;
  };

  template<typename T>
  class OffPolicyTD: public virtual Predictor<T>, public virtual LinearLearner<T>
  {
    public:
      virtual ~OffPolicyTD()
      {
      }
      virtual T update(const Vector<T>* x_t, const Vector<T>* x_tp1, const T& rho_t, const T& r_tp1,
          const T& z_tp1) =0;

  };

  template<typename T>
  class GVF: public OffPolicyTD<T>
  {
    public:
      virtual ~GVF()
      {
      }
      virtual T update(const Vector<T>* x_t, const Vector<T>* x_tp1, const T& gamma_tp1,
          const T& lambda_tp1, const T& rho_t, const T& r_tp1, const T& z_tp1) =0;
  };

} // namespace RLLib

#endif /* PREDICTOR_H_ */
