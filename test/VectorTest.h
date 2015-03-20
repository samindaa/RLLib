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
 * VectorTest.h
 *
 *  Created on: May 12, 2013
 *      Author: sam
 */

#ifndef VECTORTEST_H_
#define VECTORTEST_H_

#include "Test.h"

class VectorTest
{
  protected:
    Vector<double>* a;
    Vector<double>* b;

    std::vector<Vector<double>*> vectors;
    Random<double>* random;

  public:
    VectorTest();
    virtual ~VectorTest();

    virtual void initialize() =0;
    virtual Vector<double>* newPrototypeVector(const int& size) =0;

    virtual void run();

    virtual Vector<double>* newVector(const int& size);
    virtual Vector<double>* newVector(const double* values, const int& size);
    virtual Vector<double>* newVector(const Vector<double>* prototype);

    virtual Vector<double>* newSVector(const int& size);
    virtual Vector<double>* newSVector(const double* values, const int& size);
    virtual Vector<double>* newSVector(const Vector<double>* prototype);

    virtual Vector<double>* newPVector(const int& size);
    virtual Vector<double>* newPVector(const double* values, const int& size);
    virtual Vector<double>* newPVector(const Vector<double>* prototype);

    virtual void testAfter();
    virtual void testVectorVector();
    virtual void testSetEntry();
    virtual void testSum();
    virtual void testDotProductPVector();
    virtual void testDotProductSVector();
    virtual void testCopy();
    virtual void testPlus();
    virtual void testMinus();
    virtual void testPlusSVector();
    virtual void testMinusSVector();
    virtual void testMapTimes();
    virtual void testSubtractToSelf();
    virtual void testAddToSelf();
    virtual void testEbeMultiplySelf();
    virtual void testEbeMultiplySelf2();
    virtual void testEbeDivideSelf();
    virtual void testMax();
    virtual void testPositiveMax();
    virtual void testCheckValue();
    virtual void testAddToSelfWithFactor();
    virtual void testMultiplySelfByExponential();
    virtual void testMultiplySelfByExponentialBounded();
    virtual void testAbs();
    virtual void testSetPVector();
    virtual void testL1Norm();
    virtual void testOffset();
};

class PVectorTest: public VectorTest
{
  public:
    void initialize();
    Vector<double>* newPrototypeVector(const int& size);

};

RLLIB_TEST(PVectorTests)
class PVectorTests: public PVectorTestsBase
{
  protected:
    VectorTest* vectorTest;
  public:
    PVectorTests();
    virtual ~PVectorTests();
    void run();
};

class SVectorTest: public VectorTest
{
  private:
    Random<double>* random;

  public:
    SVectorTest();
    virtual ~SVectorTest();
    void initialize();
    Vector<double>* newPrototypeVector(const int& size);

  private:
    SVector<double>* createRandomSVector(const int& maxActive, const int& size);
    void checkVectorOperation(SVector<double>* a, Vector<double>* b);

  public:
    void testActiveIndices();
    void testSVectorSet();
    void testRandomVectors();
    void testMaxNorm();
    void testFullVector();
    void testSVector();
};

RLLIB_TEST(SVectorTests)
class SVectorTests: public SVectorTestsBase
{
  protected:
    SVectorTest* vectorTest;
  public:
    SVectorTests();
    virtual ~SVectorTests();
    void run();
};

#endif /* VECTORTEST_H_ */
