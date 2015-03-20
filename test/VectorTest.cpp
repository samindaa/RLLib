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
 * SparseVectorTest.cpp
 *
 * Created on: Dec 17, 2012
 *     Author: sam
 *
 * Tests for SparseVector
 */

#include "VectorTest.h"

VectorTest::VectorTest() :
    a(0), b(0), random(new Random<double>)
{
}

VectorTest::~VectorTest()
{
  for (vector<Vector<double>*>::iterator iter = vectors.begin(); iter != vectors.end(); ++iter)
    delete *iter;
  vectors.clear();
  delete random;
}

void VectorTest::run()
{
  testAfter();
  testVectorVector();
  testSetEntry();
  testSum();
  testDotProductPVector();
  testDotProductSVector();
  testCopy();
  testPlus();
  testMinus();
  testPlusSVector();
  testMinusSVector();
  testMapTimes();
  testSubtractToSelf();
  testAddToSelf();
  testEbeMultiplySelf();
  testEbeMultiplySelf2();
  testEbeDivideSelf();
  testMax();
  testPositiveMax();
  testCheckValue();
  testAddToSelfWithFactor();
  testMultiplySelfByExponential();
  testMultiplySelfByExponentialBounded();
  testAbs();
  testSetPVector();
  testL1Norm();
  testOffset();
}

Vector<double>* VectorTest::newVector(const int& size)
{
  Vector<double>* v = newPrototypeVector(size);
  vectors.push_back(v);
  return v;
}

Vector<double>* VectorTest::newVector(const double* values, const int& size)
{
  Vector<double>* v = newVector(size);
  for (int i = 0; i < size; i++)
    v->setEntry(i, values[i]);
  return v;
}

Vector<double>* VectorTest::newVector(const Vector<double>* prototype)
{
  Vector<double>* v = newVector(prototype->dimension());
  v->set(prototype);
  return v;
}

Vector<double>* VectorTest::newSVector(const int& size)
{
  Vector<double>* v = new SVector<double>(size, 2);
  vectors.push_back(v);
  return v;
}

Vector<double>* VectorTest::newSVector(const double* values, const int& size)
{
  Vector<double>* v = newSVector(size);
  for (int i = 0; i < size; i++)
    v->setEntry(i, values[i]);
  return v;
}

Vector<double>* VectorTest::newSVector(const Vector<double>* prototype)
{
  Vector<double>* v = newSVector(prototype->dimension());
  v->set(prototype);
  return v;
}

Vector<double>* VectorTest::newPVector(const int& size)
{
  Vector<double>* v = new PVector<double>(size);
  vectors.push_back(v);
  return v;
}

Vector<double>* VectorTest::newPVector(const double* values, const int& size)
{
  Vector<double>* v = newPVector(size);
  for (int i = 0; i < size; i++)
    v->setEntry(i, values[i]);
  return v;
}

Vector<double>* VectorTest::newPVector(const Vector<double>* prototype)
{
  Vector<double>* v = newPVector(prototype->dimension());
  v->set(prototype);
  return v;
}

// -- tests
void VectorTest::testAfter()
{
  const double aValues[] = { 0.0, 3.0, 2.0, 0.0, 1.0 };
  const double bValues[] = { 3.0, 4.0, 0.0, 0.0, 4.0 };
  Assert::assertEquals(a, newVector(aValues, Arrays::length(aValues)));
  Assert::assertEquals(b, newVector(bValues, Arrays::length(bValues)));
}

void VectorTest::testVectorVector()
{
  Vector<double>* c = newVector(a);
  Assert::assertEquals(a, c);
}

void VectorTest::testSetEntry()
{
  Vector<double>* v = newVector(a);
  v->setEntry(1, 3);
  const double bValues[] = { 0.0, 3.0, 2.0, 0.0, 1.0 };
  const double cValues[] = { 0.0, 0.0, 2.0, 0.0, 1.0 };
  Vector<double>* b = newVector(bValues, Arrays::length(bValues));
  Assert::assertEquals(b, v);
  v->setEntry(0, 0);
  v->setEntry(1, 0);
  Vector<double>* c = newVector(cValues, Arrays::length(cValues));
  Assert::assertEquals(c, v);

}

void VectorTest::testSum()
{
  Assert::assertObjectEquals(6.0, a->sum(), 0.0);
  Assert::assertObjectEquals(11.0, b->sum(), 0.0);
}

void VectorTest::testDotProductPVector()
{
  Assert::assertObjectEquals(16.0, a->dot(b), 0.0);
  Assert::assertObjectEquals(16.0, b->dot(a), 0.0);

  SVector<double> c(*(const SVector<double>*) newSVector(b));
  Assert::assertObjectEquals(16.0, a->dot(&c));
  Assert::assertObjectEquals(16.0, c.dot(a));

  PVector<double> d(*(const PVector<double>*) newPVector(b));
  Assert::assertObjectEquals(16.0, a->dot(&d));
  Assert::assertObjectEquals(16.0, d.dot(a));

}

void VectorTest::testDotProductSVector()
{
  Assert::assertObjectEquals(16.0, a->dot(newSVector(b)), 0.0);
}

void VectorTest::testCopy()
{
  Vector<double>* ca = a->copy();
  Assert::assertNotSame(ca, a);
  Assert::assertEquals(ca, a);
  delete ca;
}

void VectorTest::testPlus()
{
  Vector<double>* c = newVector(a);
  Vector<double>* d = newVector(b);
  const double cValues[] = { 3.0, 7.0, 2.0, 0.0, 5.0 };
  Vector<double>* r = newVector(cValues, Arrays::length(cValues));
  Assert::assertEquals(r, c->addToSelf(d));
}

void VectorTest::testMinus()
{
  Vector<double>* _a = newVector(a);
  Vector<double>* _b = newVector(b);

  const double cValues[] = { -3.0, -1.0, 2.0, 0.0, -3.0 };
  const double dValues[] = { -3.0, 2.0, 4.0, 0.0, -2.0 };
  Vector<double>* c = newVector(cValues, Arrays::length(cValues));
  Vector<double>* d = newVector(dValues, Arrays::length(dValues));
  Assert::assertEquals(c, _a->subtractToSelf(_b));
  Vector<double>* e = newVector(a);
  Assert::assertEquals(d, e->mapMultiplyToSelf(2.0)->subtractToSelf(b));

}

void VectorTest::testPlusSVector()
{
  Vector<double>* _a = newVector(a);
  Vector<double>* _b = newSVector(b);
  const double cValues[] = { 3.0, 7.0, 2.0, 0.0, 5.0 };
  Assert::assertEquals(newVector(cValues, Arrays::length(cValues)), _a->addToSelf(_b));
}

void VectorTest::testMinusSVector()
{
  Vector<double>* _a = newVector(a);
  Vector<double>* _b = newSVector(b);
  const double cValues[] = { -3.0, -1.0, 2.0, 0.0, -3.0 };
  Assert::assertEquals(newVector(cValues, Arrays::length(cValues)), _a->subtractToSelf(_b));
}

void VectorTest::testMapTimes()
{
  Vector<double>* _a = newVector(a);
  Vector<double>* _b = newVector(b);
  const double cValues[] = { 0.0, 15.0, 10.0, 0.0, 5.0 };
  const double dValues[] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
  Vector<double>* c = newVector(cValues, Arrays::length(cValues));
  Vector<double>* d = newVector(dValues, Arrays::length(dValues));
  Assert::assertEquals(c, _a->mapMultiplyToSelf(5.0));
  Assert::assertEquals(d, _b->mapMultiplyToSelf(0.0));
}

void VectorTest::testSubtractToSelf()
{
  {
    const double dValues[] = { 0.0, 2.0, 2.0, 0.0, 1.0 };
    const double eValues[] = { 0.0, 1.0, 0.0, 0.0, 0.0 };

    Assert::assertEquals(newVector(dValues, Arrays::length(dValues)),
        newVector(a)->subtractToSelf(newPVector(eValues, Arrays::length(eValues))));
  }

  {
    const double fValues[] = { 0.0, 2.0, 2.0, 0.0, 1.0 };
    const double gValues[] = { 0.0, 1.0, 0.0, 0.0, 0.0 };

    Assert::assertEquals(newVector(fValues, Arrays::length(fValues)),
        newVector(a)->subtractToSelf(newSVector(gValues, Arrays::length(gValues))));
  }

  {
    const double hValues[] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
    const double iValues[] = { 0.0, 3.0, 2.0, 0.0, 1.0 };

    Assert::assertEquals(newVector(hValues, Arrays::length(hValues)),
        newVector(a)->subtractToSelf(newSVector(iValues, Arrays::length(iValues))));
  }

  {
    const double jValues[] = { 0.0, 0.0, 0.0, 0.0, 0.0 };

    Assert::assertEquals(newVector(jValues, Arrays::length(jValues)),
        newVector(a)->subtractToSelf(newVector(a)));
  }
}

void VectorTest::testAddToSelf()
{
  {
    const double dValues[] = { 0.0, 2.0, 2.0, 0.0, 1.0 };
    const double eValues[] = { 0.0, -1.0, 0.0, 0.0, 0.0 };

    Assert::assertEquals(newVector(dValues, Arrays::length(dValues)),
        newVector(a)->addToSelf(newPVector(eValues, Arrays::length(eValues))));
  }

  {
    const double fValues[] = { 0.0, 3.0, 2.0, 0.0, 0.0 };
    const double gValues[] = { 0.0, 0.0, 0.0, 0.0, -1.0 };

    Assert::assertEquals(newVector(fValues, Arrays::length(fValues)),
        newVector(a)->addToSelf(newPVector(gValues, Arrays::length(gValues))));
  }

  {
    const double hValues[] = { 0.0, 2.0, 2.0, 0.0, 1.0 };
    const double iValues[] = { 0.0, -1.0, 0.0, 0.0, 0.0 };

    Assert::assertEquals(newVector(hValues, Arrays::length(hValues)),
        newVector(a)->addToSelf(newSVector(iValues, Arrays::length(iValues))));
  }

  {
    const double jValues[] = { 0.0, 0.0, 0.0, 0.0, 0.0 };

    Assert::assertEquals(newVector(jValues, Arrays::length(jValues)),
        newVector(a)->addToSelf(newVector(a)->mapMultiplyToSelf(-1.0f)));
  }
}

void VectorTest::testEbeMultiplySelf()
{
  const double a1Values[] = { -1, 1, 2 };
  {
    const double a2Values[] = { 3, 4, 5 };
    const double rrValues[] = { -3, 4, 10 };
    Assert::assertEquals(newVector(rrValues, Arrays::length(rrValues)),
        newVector(a1Values, Arrays::length(a1Values))->ebeMultiplyToSelf(
            newVector(a2Values, Arrays::length(a2Values))));
  }
  {
    const double a1pValues[] = { 1.0, 2.0, 3.0 };
    const double zzValues[] = { -1, 2, 6 };

    Assert::assertEquals(newVector(zzValues, Arrays::length(zzValues)),
        newVector(a1Values, Arrays::length(a1Values))->ebeMultiplyToSelf(
            newPVector(a1pValues, Arrays::length(a1pValues))));
  }
  {
    const double a3Values[] = { 0, 1, 0 };
    const double a4Values[] = { -1, 0, 2 };
    const double ooValues[] = { 0, 0, 0 };
    Assert::assertEquals(newVector(ooValues, Arrays::length(ooValues)),
        newVector(a3Values, Arrays::length(a3Values))->ebeMultiplyToSelf(
            newVector(a4Values, Arrays::length(a4Values))));
  }
  {
    const double p1Values[] = { 2.0, 2.0, 2.0, 2.0, 2.0 };
    const double rrValues[] = { 0, 6.0, 4.0, 0.0, 2.0 };
    Vector<double>* p1 = newPVector(p1Values, Arrays::length(p1Values));
    Assert::assertEquals(newPVector(rrValues, Arrays::length(rrValues)), p1->ebeMultiplyToSelf(a));
  }
}

void VectorTest::testEbeMultiplySelf2()
{
  const double _aValues[] = { 0, 1, 0, -2, 0, 3, 0 };
  const double rrValues[] = { 0, 1, 0, 4, 0, 9, 0 };
  Vector<double>* _a = newVector(_aValues, Arrays::length(_aValues));
  Assert::assertEquals(newPVector(rrValues, Arrays::length(rrValues)), _a->ebeMultiplyToSelf(_a));
}

void VectorTest::testEbeDivideSelf()
{
  const double a2Values[] = { 3, 4, 0, -6 };
  const double a1Values[] = { -1, -2, 2, 3 };
  const double expectedValues[] = { -3, -2, 0.0, -2.0 };

  Assert::assertEquals(newPVector(expectedValues, Arrays::length(expectedValues)),
      newSVector(a2Values, Arrays::length(a2Values))->ebeDivideToSelf(
          newVector(a1Values, Arrays::length(a1Values))));

  const double a3Values[] = { 3, 4, 0, -6 };
  Assert::assertEquals(newPVector(expectedValues, Arrays::length(expectedValues)),
      newPVector(a3Values, Arrays::length(a3Values))->ebeDivideToSelf(
          newVector(a1Values, Arrays::length(a1Values))));
}

void VectorTest::testMax()
{
  const double aValues[] = { 1.0, -2.0, -3.0, 0.0, 2.0 };
  Vector<double>* a = newVector(aValues, Arrays::length(aValues));
  Assert::assertObjectEquals(3.0, a->maxNorm());
}

void VectorTest::testPositiveMax()
{
  const double a1Values[] = { 2, -1, 1, 3, 1 };
  const double sa2Values[] = { 1, 0, -2, 0, -3 };
  Vector<double>* a1 = newVector(a1Values, Arrays::length(a1Values));
  Vector<double>* sa2 = newSVector(sa2Values, Arrays::length(sa2Values));
  Vector<double>* pa2 = newPVector(sa2Values, Arrays::length(sa2Values));
  const double expectedValues[] = { 2, 0, 1, 3, 1 };
  Vector<double>* expected = newPVector(expectedValues, Arrays::length(expectedValues));
  Vectors<double>::positiveMaxToSelf(sa2, a1);
  Vectors<double>::positiveMaxToSelf(pa2, a1);
  Assert::assertEquals(expected, sa2);
  Assert::assertEquals(expected, pa2);
}

void VectorTest::testCheckValue()
{
  const double aValues[] = { 1.0, 1.0 };
  Assert::assertPasses(VectorsTestsUtils::checkValues(newVector(aValues, Arrays::length(aValues))));
  const double bValues[] = { 1.0, std::numeric_limits<double>::quiet_NaN() };
  Assert::assertFails(VectorsTestsUtils::checkValues(newVector(bValues, Arrays::length(bValues))));
  const double cValues[] = { 1.0, std::numeric_limits<double>::infinity() };
  Assert::assertFails(VectorsTestsUtils::checkValues(newVector(cValues, Arrays::length(cValues))));
}

void VectorTest::testAddToSelfWithFactor()
{
  const double vValues[] = { 1.0, 1.0, 1.0, 1.0, 1.0 };
  Vector<double>* v = newPVector(vValues, Arrays::length(vValues));
  v->addToSelf(2.0, a);
  const double rValues[] = { 1.0, 7.0, 5.0, 1.0, 3.0 };
  Assert::assertEquals(newPVector(rValues, Arrays::length(rValues)), v);
  v->addToSelf(-1.0, b);
  const double sValues[] = { -2.0, 3.0, 5.0, 1.0, -1.0 };
  Assert::assertEquals(newPVector(sValues, Arrays::length(sValues)), v);
}

void VectorTest::testMultiplySelfByExponential()
{
  const double v1Values[] = { 0.0, 1.0, 2.0, 3.0, 4.0 };
  const double v2Values[] = { 1.0, -2.0, -3.0, 0.0, 2.0 };
  DenseVector<double>* v1 = (DenseVector<double>*) newPVector(v1Values, Arrays::length(v1Values));
  Vector<double>* v2 = newVector(v2Values, Arrays::length(v2Values));
  Vectors<double>::multiplySelfByExponential(v1, 1.0, v2);

  const double rValues[] = { 0.0, std::exp(-2.0), 2.0 * std::exp(-3.0), 3.0, 4.0 * std::exp(2.0) };
  Assert::assertEquals(newVector(rValues, Arrays::length(rValues)), v1);

  const double v3Values[] = { -10000.0, -10000.0, -10000.0, -100000.0, -10000.0 };
  Vectors<double>::multiplySelfByExponential(v1, 1.0,
      newVector(v3Values, Arrays::length(v3Values)));
  ASSERT(Vectors<double>::isNull(v1));
}

void VectorTest::testMultiplySelfByExponentialBounded()
{
  const double v1Values[] = { 0.0, 1.0, 2.0, 3.0, 4.0 };
  const double v2Values[] = { 1.0, -2.0, -3.0, 0.0, 2.0 };
  DenseVector<double>* v1 = (DenseVector<double>*) newPVector(v1Values, Arrays::length(v1Values));
  Vector<double>* v2 = newVector(v2Values, Arrays::length(v2Values));
  Vectors<double>::multiplySelfByExponential(v1, 1.0, v2);

  const double rValues[] = { 0.0, std::exp(-2.0), 2.0 * std::exp(-3.0), 3.0, 4 * std::exp(2.0) };
  Assert::assertEquals(newVector(rValues, Arrays::length(rValues)), v1);

  const double v3Values[] = { -10000.0, -10000.0, -10000.0, -100000.0, -10000.0 };
  Vectors<double>::multiplySelfByExponential(v1, 1.0, newVector(v3Values, Arrays::length(v3Values)),
      0.1);
  ASSERT(!Vectors<double>::isNull(v1));
}

void VectorTest::testAbs()
{
  const double vValues[] = { 1.0, -2.0, -3.0, 0.0, 2.0 };
  const double rValues[] = { 1.0, 2.0, 3.0, 0.0, 2.0 };
  Vector<double>* v = newVector(vValues, Arrays::length(vValues));
  Assert::assertEquals(newPVector(rValues, Arrays::length(rValues)), Vectors<double>::absToSelf(v));
}

void VectorTest::testSetPVector()
{
  const double vValues[] = { 1.0, -2.0, -3.0, 0.0, 2.0 };
  Vector<double>* v = newVector(vValues, Arrays::length(vValues));
  Vector<double>* v1 = newPVector(5);
  Assert::assertEquals(v, v1->set(v));
}

void VectorTest::testL1Norm()
{
  const double vValues[] = { 1.0, -2.0, -3.0, 0.0, 2.0 };
  Vector<double>* v = newVector(vValues, Arrays::length(vValues));
  Assert::assertObjectEquals(8.0, v->l1Norm(), 1e-8);
}

void VectorTest::testOffset()
{
  const int nbStrips = 3;
  const int stripSize = 10;
  Vector<double>* a = newVector(nbStrips * stripSize);

  Vector<double>* b = newPVector(stripSize);
  Vector<double>* c = newSVector(stripSize);
  b->set(1.0f);
  c->set(1.0f);

  Vector<double>* d = newPVector(stripSize);
  Vector<double>* e = newSVector(stripSize);
  for (int i = 0; i < stripSize; i++)
  {
    d->setEntry(i, random->nextNormalGaussian());
    e->setEntry(i, random->nextNormalGaussian());
  }

  for (int i = 0; i < nbStrips; i++)
  {
    a->set(b, stripSize * i);
    std::cout << a << std::endl;
    double value = 0;
    for (int j = 0; j < stripSize; j++)
      value += a->getEntry(i * stripSize + j);
    Assert::assertObjectEquals(value, b->sum());
  }

  for (int i = 0; i < nbStrips; i++)
  {
    a->set(c, stripSize * i);
    double value = 0;
    for (int j = 0; j < stripSize; j++)
      value += a->getEntry(i * stripSize + j);
    Assert::assertObjectEquals(value, c->sum());
  }

  for (int i = 0; i < nbStrips; i++)
  {
    a->set(d, stripSize * i);
    double value = 0;
    for (int j = 0; j < stripSize; j++)
      value += a->getEntry(i * stripSize + j);
    Assert::assertObjectEquals(value, d->sum(), 0.0001);
  }

  for (int i = 0; i < nbStrips; i++)
  {
    a->set(e, stripSize * i);
    double value = 0;
    for (int j = 0; j < stripSize; j++)
      value += a->getEntry(i * stripSize + j);
    Assert::assertObjectEquals(value, e->sum(), 0.0001);
  }
}

void PVectorTest::initialize()
{
  const double aValues[] = { 0.0, 3.0, 2.0, 0.0, 1.0 };
  const double bValues[] = { 3.0, 4.0, 0.0, 0.0, 4.0 };
  a = newVector(aValues, Arrays::length(aValues));
  b = newVector(bValues, Arrays::length(bValues));
}

Vector<double>* PVectorTest::newPrototypeVector(const int& size)
{
  return new PVector<double>(size);
}

// -- tests
RLLIB_TEST_MAKE(PVectorTests)

PVectorTests::PVectorTests() :
    vectorTest(new PVectorTest)
{
}

PVectorTests::~PVectorTests()
{
  delete vectorTest;
}

void PVectorTests::run()
{
  vectorTest->initialize();
  vectorTest->run();
}

RLLIB_TEST_MAKE(SVectorTests)

SVectorTests::SVectorTests() :
    vectorTest(new SVectorTest)
{
}

SVectorTests::~SVectorTests()
{
  delete vectorTest;
}

void SVectorTests::run()
{
  vectorTest->initialize();
  vectorTest->run();

  vectorTest->testActiveIndices();
  vectorTest->testSVectorSet();
  vectorTest->testRandomVectors();
  vectorTest->testMaxNorm();
  vectorTest->testFullVector();
  vectorTest->testSVector();
}

SVectorTest::SVectorTest() :
    random(new Random<double>())
{
}

SVectorTest::~SVectorTest()
{
  delete random;
}

void SVectorTest::initialize()
{
  const double aValues[] = { 0.0, 3.0, 2.0, 0.0, 1.0 };
  const double bValues[] = { 3.0, 4.0, 0.0, 0.0, 4.0 };
  a = newVector(aValues, Arrays::length(aValues));
  b = newVector(bValues, Arrays::length(bValues));
}

Vector<double>* SVectorTest::newPrototypeVector(const int& size)
{
  return new SVector<double>(size);
}

SVector<double>* SVectorTest::createRandomSVector(const int& maxActive, const int& size)
{
  SVector<double>* result = (SVector<double>*) newVector(size);
  int nbActive = random->nextInt(maxActive);
  for (int i = 0; i < nbActive; i++)
    result->setEntry(random->nextInt(size), random->nextReal() * 2 - 1);
  Assert::checkConsistency(result);
  return result;
}

void SVectorTest::checkVectorOperation(SVector<double>* a, Vector<double>* b)
{
  PVector<double>* pa = (PVector<double>*) newPVector(a);
  PVector<double>* pb = (PVector<double>*) newPVector(b);
  Assert::assertEquals(pa, a);
  Assert::assertEquals(pb, b);
  Assert::assertEquals(pa->addToSelf(pb), a->addToSelf(b));
  Assert::assertEquals(pa->subtractToSelf(pb), a->subtractToSelf(b));
  Assert::assertEquals(pa->ebeMultiplyToSelf(pb), a->ebeMultiplyToSelf(b));
  float factor = random->nextReal();
  Assert::assertEquals(pa->addToSelf(factor, pb), a->addToSelf(factor, b));
}

void SVectorTest::testActiveIndices()
{
  const double aValues[] = { 0.0, 3.0, 2.0, 0.0, 1.0 };
  const double bValues[] = { 3.0, 4.0, 0.0, 0.0, 4.0 };
  SparseVector<double>* a = (SparseVector<double>*) newSVector(aValues, Arrays::length(aValues));
  SparseVector<double>* b = (SparseVector<double>*) newSVector(bValues, Arrays::length(bValues));

  const int aGroundTruthActiveIndices[] = { 1, 2, 4 };
  const int bGroundTruthActiveIndices[] = { 0, 1, 4 };
  const int* aActiveIndices = a->nonZeroIndexes();
  const int* bActiveIndices = b->nonZeroIndexes();

  Assert::assertObjectEquals(3, a->nonZeroElements());
  Assert::assertObjectEquals(3, b->nonZeroElements());
  for (int i = 0; i < a->nonZeroElements(); i++)
    Assert::assertObjectEquals(aGroundTruthActiveIndices[i], aActiveIndices[i]);
  for (int i = 0; i < b->nonZeroElements(); i++)
    Assert::assertObjectEquals(bGroundTruthActiveIndices[i], bActiveIndices[i]);

}

void SVectorTest::testSVectorSet()
{
  const double aValues[] = { 1.0, 2.0, 2.0, 4.0 };
  const double bValues[] = { 0.0, 1.0, 0.0, 0.0 };
  SparseVector<double>* s = (SparseVector<double>*) newSVector(aValues, Arrays::length(aValues));
  SparseVector<double>* b = (SparseVector<double>*) newSVector(bValues, Arrays::length(bValues));
  s->set(b);
  Assert::checkConsistency(s);
  DenseVector<double>* d = (DenseVector<double>*) newPVector(bValues, Arrays::length(bValues));
  Assert::assertEquals(d, s);
}

void SVectorTest::testRandomVectors()
{
  int size = 10;
  int active = 4;

  for (int i = 0; i < 10000; i++)
  {
    SVector<double>* a = createRandomSVector(active, size);
    SVector<double>* b = createRandomSVector(active, size);
    checkVectorOperation(a, b);
  }

}

void SVectorTest::testMaxNorm()
{
  const double aValues[] = { 1.0, -2.0, -3.0, 0.0, 2.0 };
  Vector<double>* a = newVector(aValues, Arrays::length(aValues));
  Assert::assertObjectEquals(3.0, a->maxNorm());
}

void SVectorTest::testFullVector()
{
  PVector<float> v(10);
  cout << v << endl;
  for (int i = 0; i < v.dimension(); i++)
  {
    double k = random->nextReal();
    v[i] = k;
    cout << k << " ";
  }
  cout << endl;
  cout << v << endl;
  PVector<float> d;
  cout << d << endl;
  d = v;
  d * 100;
  cout << d << endl;
  cout << d.maxNorm() << endl;

  PVector<float> i(5);
  i[0] = 1.0;
  cout << i << endl;
  cout << i.maxNorm() << endl;
  cout << i.l2Norm() << endl;
}

void SVectorTest::testSVector()
{
  SVector<double> a(20, 2);
  SVector<double> b(20, 2);
  for (int i = 0; i < 5; i++)
  {
    a.insertEntry(i, i + 1);
    b.insertEntry(i, i + 11);
  }

//cout << a << endl;
//cout << b << endl;
//cout << a.nbActiveEntries() << " " << b.nbActiveEntries() << endl;
  Assert::assertObjectEquals(5, a.nonZeroElements());
  Assert::assertObjectEquals(5, b.nonZeroElements());
  Assert::assertObjectEquals(205, (int) a.dot(&b));
  b.removeEntry(2);
//cout << a.nbActiveEntries() << " " << b.nbActiveEntries() << endl;
  Assert::assertObjectEquals(4, b.nonZeroElements());
//cout << a << endl;
//cout << b << endl;
  Assert::assertObjectEquals(166, (int) a.dot(&b));
//cout << "dot=" << a.dot(b) << endl;
  cout << a.addToSelf(&b) << endl;
  a.clear();
  b.clear();
  Assert::assertObjectEquals(0, (int) a.dot(&b));
//cout << a << endl;
//cout << b << endl;
}
