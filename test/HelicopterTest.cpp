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
 * HelicopterTest.cpp
 *
 *  Created on: May 13, 2013
 *      Author: sam
 */

#include "HelicopterTest.h"

RLLIB_TEST_MAKE(HelicopterTest)

void HelicopterTest::run()
{
  testEndEpisode();
  testCrash();
}

