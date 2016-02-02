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
 * Function.h
 *
 *  Created on: Sep 1, 2013
 *      Author: sam
 */

#ifndef FUNCTION_H_
#define FUNCTION_H_

#include "Vector.h"

namespace RLLib
{

  template<typename T>
  class ParameterizedFunction
  {
    public:
      virtual ~ParameterizedFunction()
      {
      }
      virtual void persist(const char* f) const =0;
      virtual void resurrect(const char* f) =0;
  };

  template<typename T>
  class LinearLearner: public ParameterizedFunction<T>
  {
    public:
      virtual ~LinearLearner()
      {
      }
      virtual T initialize() =0;
      virtual void reset() =0;
  };

  template<typename T>
  class RewardFunction
  {
    public:
      virtual ~RewardFunction()
      {
      }
      virtual T reward() =0;
  };

  template<typename T>
  class OutcomeFunction
  {
    public:
      virtual ~OutcomeFunction()
      {
      }
      virtual T outcome() =0;
  };

}  // namespace RLLib

#endif /* FUNCTION_H_ */
