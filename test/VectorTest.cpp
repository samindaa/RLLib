/*
 * SparseVectorTest.cpp
 *
 * Created on: Dec 17, 2012
 *     Author: sam
 *
 * Tests for SparseVector
 */

#include "VectorTest.h"

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

