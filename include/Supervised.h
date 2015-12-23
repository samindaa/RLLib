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
 * Supervised.h
 *
 *  Created on: Sep 7, 2012
 *      Author: sam
 */

#ifndef SUPERVISED_H_
#define SUPERVISED_H_

#include "Predictor.h"

namespace RLLib
{

  template<typename T>
  class LearningAlgorithm: public Predictor<T>
  {
    public:
      virtual ~LearningAlgorithm()
      {
      }
      virtual T learn(const Vector<T>* x_t, const T& y_tp1) = 0;
  };

} // namespace RLLib

#endif /* SUPERVISED_H_ */
