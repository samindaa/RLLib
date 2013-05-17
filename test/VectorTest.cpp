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
 * SparseVectorTest.cpp
 *
 * Created on: Dec 17, 2012
 *     Author: sam
 *
 * Tests for SparseVector
 */

#include "VectorTest.h"

RLLIB_TEST_MAKE(SparseVectorTest)

void SparseVectorTest::run()
{
  testActiveIndices();
  testSparseVectorSet();
  testRandomVectors();
  testSetEntry();
  testSum();
  testDotProduct();
  testPlus();
  testMinus();
  testMapTimes();
  testMaxNorm();
  testFullVector();
  testSparseVector();
  testEbeMultiply();
}

