/*
 * SparseVectorTest.cpp
 *
 * Created on: Dec 17, 2012
 *     Author: sam
 *
 * Tests for SparseVector
 */

#include "HeaderTest.h"

class SparseVectorTest
{
  protected:
    typedef void (SparseVectorTest::*testMethod)();
    vector<SVecDoubleType*> vectors;
    vector<testMethod> testMethods;

    SVecDoubleType* _a;
    SVecDoubleType* _b;

  public:
    SparseVectorTest() :
        _a(0), _b(0)
    {
      srand(time(0));
    }

    ~SparseVectorTest()
    {
      for (vector<SVecDoubleType*>::iterator iter = vectors.begin();
          iter != vectors.end(); ++iter)
        delete *iter;
      vectors.clear();
      testMethods.clear();
    }

  private:

    SVecDoubleType* newVector(const int& size)
    {
      SVecDoubleType* type = new SVecDoubleType(size);
      vectors.push_back(type);
      return type;
    }

    SVecDoubleType* newVector(const double* values, const int& size)
    {
      SVecDoubleType* type = newVector(size);
      for (int i = 0; i < size; i++)
        type->setEntry(i, values[i]);
      return type;
    }

    SVecDoubleType* createRandomSparseVector(const int& maxActive,
        const int& size)
    {
      SVecDoubleType* type = newVector(size);
      int nbActive = rand() % maxActive;
      for (int i = 0; i < nbActive; i++)
        type->setEntry(rand() % maxActive, Random::nextDouble() * 2 - 1);
      assert(checkSparseVectorConsistency(*type));
      return type;
    }

    void checkVectorOperations(SVecDoubleType& a, SVecDoubleType& b)
    {
      SVecDoubleType pa(a);
      SVecDoubleType pb(b);
      assert(checkSparseVectorConsistency(pa));
      assert(checkSparseVectorConsistency(pb));
      checkVectorEquals(pa, a);
      checkVectorEquals(pb, b);
      checkVectorEquals(pa.addToSelf(pb), a.addToSelf(b));
      checkVectorEquals(pa.substractToSelf(pb), a.substractToSelf(b));
      float factor = Random::nextFloat();
      checkVectorEquals(pa.addToSelf(factor, pb), a.addToSelf(factor, b));
    }

  protected:
    void testActiveIndices()
    {
      SVecDoubleType* a = newVector((double[]
          )
          { 0.0, 3.0, 2.0, 0.0, 1.0 }, 5);
      SVecDoubleType* b = newVector((double[]
          )
          { 3.0, 4.0, 0.0, 0.0, 4.0 }, 5);

      const int aGroundTruthActiveIndices[] =
      { 1, 2, 4 };
      const int bGroundTruthActiveIndices[] =
      { 0, 1, 4 };
      const int* aActiveIndices = a->getActiveIndexes();
      const int* bActiveIndices = b->getActiveIndexes();

      assert(3 == a->nbActiveEntries());
      assert(3 == b->nbActiveEntries());
      for (int i = 0; i < a->nbActiveEntries(); i++)
        assert(aGroundTruthActiveIndices[i] == aActiveIndices[i]);
      for (int i = 0; i < b->nbActiveEntries(); i++)
        assert(bGroundTruthActiveIndices[i] == bActiveIndices[i]);

      // Now we are ready to use them
      _a = a;
      _b = b;

    }

    void testSparseVectorSet()
    {
      SVecDoubleType* a = newVector((double[]
          )
          { 1.0, 2.0, 2.0, 4.0 }, 4);

      SVecDoubleType* b = newVector((double[]
          )
          { 0.0, 1.0, 0.0, 0.0 }, 4);
      a->set(*b);
      assert(checkSparseVectorConsistency(*a));
    }

    void testRandomVectors()
    {
      int size = 20;
      int active = 8;

      for (int i = 0; i < 10000; i++)
      {
        SVecDoubleType* a = createRandomSparseVector(active, size);
        SVecDoubleType* b = createRandomSparseVector(active, size);
        checkVectorOperations(*a, *b);
      }

    }

    void testSetEntry()
    {
      SVecDoubleType v(*_a);
      v.setEntry(1, 3);
      SVecDoubleType* b = newVector((double[]
          )
          { 0.0, 3.0, 2.0, 0.0, 1.0 }, 5);
      checkVectorEquals(*b, v);
      v.setEntry(0, 0);
      v.setEntry(1, 0);
      SVecDoubleType* c = newVector((double[]
          )
          { 0.0, 0.0, 2.0, 0.0, 1.0 }, 5);
      checkVectorEquals(*c, v);
    }

    void testSum()
    {
      assert(6.0 == _a->sum());
      assert(11.0 == _b->sum());
    }

    void testDotProduct()
    {
      assert(16.0 == _a->dot(*_b));
      assert(16.0 == _b->dot(*_a));

      SVecDoubleType c(*_b);
      assert(16.0 == _a->dot(c));
      assert(16.0 == c.dot(*_a));
    }

    void testPlus()
    {
      SVecDoubleType a(*_a);
      SVecDoubleType b(*_b);
      SVecDoubleType* c = newVector((double[]
          )
          { 3.0, 7.0, 2.0, 0.0, 5.0 }, 5);
      checkVectorEquals(*c, a.addToSelf(b));
    }

    void testMinus()
    {
      SVecDoubleType a(*_a);
      SVecDoubleType b(*_b);
      SVecDoubleType* c = newVector((double[]
          )
          { -3.0, -1.0, 2.0, 0.0, -3.0 }, 5);
      SVecDoubleType* d = newVector((double[]
          )
          { -3.0, 2.0, 4.0, 0.0, -2.0 }, 5);
      checkVectorEquals(*c, a.substractToSelf(b));
      SVecDoubleType e(*_a);
      checkVectorEquals(*d, e.multiplyToSelf(2.0).substractToSelf(b));
    }

    void testMapTimes()
    {
      SVecDoubleType a(*_a);
      SVecDoubleType b(*_b);
      SVecDoubleType* c = newVector((double[]
          )
          { 0.0, 15.0, 10.0, 0.0, 5.0 }, 5);
      SVecDoubleType* d = newVector((double[]
          )
          { 0.0, 0.0, 0.0, 0.0, 0.0 }, 5);
      checkVectorEquals(*c, a.multiplyToSelf(5.0));
      checkVectorEquals(*d, b.multiplyToSelf(0.0));
    }

    void testMaxNorm()
    {
      SVecDoubleType* a = newVector((double[]
          )
          { 1.0, -2.0, -3.0, 0.0, 2.0 }, 5);
      assert(3.0 == a->maxNorm());
    }

  public:
    void run()
    {
      // register
      testMethods.push_back(&SparseVectorTest::testActiveIndices);
      testMethods.push_back(&SparseVectorTest::testSparseVectorSet);
      testMethods.push_back(&SparseVectorTest::testRandomVectors);
      testMethods.push_back(&SparseVectorTest::testSetEntry);
      testMethods.push_back(&SparseVectorTest::testSum);
      testMethods.push_back(&SparseVectorTest::testDotProduct);
      testMethods.push_back(&SparseVectorTest::testPlus);
      testMethods.push_back(&SparseVectorTest::testMinus);
      testMethods.push_back(&SparseVectorTest::testMapTimes);
      testMethods.push_back(&SparseVectorTest::testMaxNorm);

      int methodCounter = 0;
      cout << "*** nbTests=" << testMethods.size() << endl;
      for (vector<testMethod>::iterator f = testMethods.begin();
          f != testMethods.end(); ++f)
      {
        cout << "*** running=" << methodCounter++ << endl;
        (this->*(*f))();
      }
    }
};

int main(int argc, char** argv)
{
  cout << "*** VectorTest starts ... " << endl;
  SparseVectorTest sparseVectorTest;
  sparseVectorTest.run();
  cout << "*** VectorTest ends ... " << endl;
  return 0;
}

