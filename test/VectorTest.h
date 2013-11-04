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
 * VectorTest.h
 *
 *  Created on: May 12, 2013
 *      Author: sam
 */

#ifndef VECTORTEST_H_
#define VECTORTEST_H_

#include "HeaderTest.h"

RLLIB_TEST(SparseVectorTest)

class SparseVectorTest: public SparseVectorTestBase
{
  protected:
    vector<SparseVector<double>*> vectors;

    SparseVector<double>* _a;
    SparseVector<double>* _b;

  public:
    SparseVectorTest() :
        _a(0), _b(0)
    {
      srand(time(0));
    }
    virtual ~SparseVectorTest()
    {
      for (vector<SparseVector<double>*>::iterator iter = vectors.begin(); iter != vectors.end();
          ++iter)
        delete *iter;
      vectors.clear();
    }

  private:

    SparseVector<double>* newVector(const int& size)
    {
      SparseVector<double>* type = new SVector<double>(size, 2);
      vectors.push_back(type);
      return type;
    }

    SparseVector<double>* newVector(const double* values, const int& size)
    {
      SparseVector<double>* type = newVector(size);
      for (int i = 0; i < size; i++)
        type->setEntry(i, values[i]);
      return type;
    }

    SparseVector<double>* createRandomSparseVector(const int& maxActive, const int& size)
    {
      SparseVector<double>* type = newVector(size);
      int nbActive = rand() % maxActive;
      for (int i = 0; i < nbActive; i++)
        type->setEntry(rand() % maxActive, Probabilistic::nextDouble() * 2 - 1);
      Assert::checkConsistency(type);
      return type;
    }

    void checkVectorOperations(SparseVector<double>* a, SparseVector<double>* b)
    {
      SVector<double> pa(a);
      SVector<double> pb(b);
      Assert::checkConsistency(&pa);
      Assert::checkConsistency(&pb);
      Assert::checkVectorEquals(&pa, a);
      Assert::checkVectorEquals(&pb, b);
      Assert::checkVectorEquals(pa.addToSelf(&pb), a->addToSelf(b));
      Assert::checkVectorEquals(pa.subtractToSelf(&pb), a->subtractToSelf(b));
      float factor = Probabilistic::nextFloat();
      Assert::checkVectorEquals(pa.addToSelf(factor, &pb), a->addToSelf(factor, b));
    }

  protected:
    void testActiveIndices()
    {
      const double aValues[] = { 0.0, 3.0, 2.0, 0.0, 1.0 };
      const double bValues[] = { 3.0, 4.0, 0.0, 0.0, 4.0 };
      SparseVector<double>* a = newVector(aValues, Arrays::length(aValues));
      SparseVector<double>* b = newVector(bValues, Arrays::length(bValues));

      const int aGroundTruthActiveIndices[] = { 1, 2, 4 };
      const int bGroundTruthActiveIndices[] = { 0, 1, 4 };
      const int* aActiveIndices = a->nonZeroIndexes();
      const int* bActiveIndices = b->nonZeroIndexes();

      Assert::equals(3, a->nonZeroElements());
      Assert::equals(3, b->nonZeroElements());
      for (int i = 0; i < a->nonZeroElements(); i++)
        Assert::equals(aGroundTruthActiveIndices[i], aActiveIndices[i]);
      for (int i = 0; i < b->nonZeroElements(); i++)
        Assert::equals(bGroundTruthActiveIndices[i], bActiveIndices[i]);

      // Now we are ready to use them
      _a = a;
      _b = b;

    }

    void testSparseVectorSet()
    {
      const double aValues[] = { 1.0, 2.0, 2.0, 4.0 };
      const double bValues[] = { 0.0, 1.0, 0.0, 0.0 };
      SparseVector<double>* a = newVector(aValues, Arrays::length(aValues));
      SparseVector<double>* b = newVector(bValues, Arrays::length(bValues));
      a->set(b);
      Assert::checkConsistency(a);
    }

    void testRandomVectors()
    {
      int size = 20;
      int active = 8;

      for (int i = 0; i < 10000; i++)
      {
        SparseVector<double>* a = createRandomSparseVector(active, size);
        SparseVector<double>* b = createRandomSparseVector(active, size);
        checkVectorOperations(a, b);
      }

    }

    void testSetEntry()
    {
      SVector<double> v(_a);
      v.setEntry(1, 3);
      const double bValues[] = { 0.0, 3.0, 2.0, 0.0, 1.0 };
      const double cValues[] = { 0.0, 0.0, 2.0, 0.0, 1.0 };
      SparseVector<double>* b = newVector(bValues, Arrays::length(bValues));
      Assert::checkVectorEquals(b, &v);
      v.setEntry(0, 0);
      v.setEntry(1, 0);
      SparseVector<double>* c = newVector(cValues, Arrays::length(cValues));
      Assert::checkVectorEquals(c, &v);
    }

    void testSum()
    {
      Assert::equals(6.0, _a->sum());
      Assert::equals(11.0, _b->sum());
    }

    void testDotProduct()
    {
      Assert::equals(16.0, _a->dot(_b));
      Assert::equals(16.0, _b->dot(_a));

      SVector<double> c(_b);
      Assert::equals(16.0, _a->dot(&c));
      Assert::equals(16.0, c.dot(_a));
    }

    void testPlus()
    {
      SVector<double> a(_a);
      SVector<double> b(_b);
      const double cValues[] = { 3.0, 7.0, 2.0, 0.0, 5.0 };
      SparseVector<double>* c = newVector(cValues, Arrays::length(cValues));
      Assert::checkVectorEquals(c, a.addToSelf(&b));
    }

    void testMinus()
    {
      SVector<double> a(_a);
      SVector<double> b(_b);

      const double cValues[] = { -3.0, -1.0, 2.0, 0.0, -3.0 };
      const double dValues[] = { -3.0, 2.0, 4.0, 0.0, -2.0 };
      SparseVector<double>* c = newVector(cValues, Arrays::length(cValues));
      SparseVector<double>* d = newVector(dValues, Arrays::length(dValues));
      Assert::checkVectorEquals(c, a.subtractToSelf(&b));
      SVector<double> e(_a);
      Assert::checkVectorEquals(d, e.mapMultiplyToSelf(2.0)->subtractToSelf(&b));
    }

    void testMapTimes()
    {
      SVector<double> a(_a);
      SVector<double> b(_b);
      const double cValues[] = { 0.0, 15.0, 10.0, 0.0, 5.0 };
      const double dValues[] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
      SparseVector<double>* c = newVector(cValues, Arrays::length(cValues));
      SparseVector<double>* d = newVector(dValues, Arrays::length(dValues));
      Assert::checkVectorEquals(c, a.mapMultiplyToSelf(5.0));
      Assert::checkVectorEquals(d, b.mapMultiplyToSelf(0.0));
    }

    void testMaxNorm()
    {
      const double aValues[] = { 1.0, -2.0, -3.0, 0.0, 2.0 };
      SparseVector<double>* a = newVector(aValues, Arrays::length(aValues));
      Assert::equals(3.0, a->maxNorm());
    }

    void testFullVector()
    {
      PVector<float> v(10);
      cout << v << endl;
      for (int i = 0; i < v.dimension(); i++)
      {
        double k = Probabilistic::nextDouble();
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
      cout << i.euclideanNorm() << endl;
    }

    void testSparseVector()
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
      Assert::equals(5, a.nonZeroElements());
      Assert::equals(5, b.nonZeroElements());
      Assert::equals(205, (int) a.dot(&b));
      b.removeEntry(2);
      //cout << a.nbActiveEntries() << " " << b.nbActiveEntries() << endl;
      Assert::equals(4, b.nonZeroElements());
      //cout << a << endl;
      //cout << b << endl;
      Assert::equals(166, (int) a.dot(&b));
      //cout << "dot=" << a.dot(b) << endl;
      cout << a.addToSelf(&b) << endl;
      a.clear();
      b.clear();
      Assert::equals(0, (int) a.dot(&b));
      //cout << a << endl;
      //cout << b << endl;
    }

    void testEbeMultiply()
    {
      const double a2Values[] = { 3, 4, 5 };
      const double a1Values[] = { -1, 1, 2 };
      const double cValues[] = { -3, 4, 10 };
      SparseVector<double>* a2 = newVector(a2Values, Arrays::length(a2Values));
      SparseVector<double>* a1 = newVector(a1Values, Arrays::length(a1Values));

      SparseVector<double>* c = newVector(cValues, Arrays::length(cValues));
      Assert::checkVectorEquals(c, a2->ebeMultiplyToSelf(a1));
    }

  public:
    void run();
};

#endif /* VECTORTEST_H_ */
