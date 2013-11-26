/*
 * IDBDTest.h
 *
 *  Created on: Nov 18, 2013
 *      Author: sam
 */

#ifndef IDBDTEST_H_
#define IDBDTEST_H_

#include "Test.h"

RLLIB_TEST(IDBDTest)
class IDBDTest: public IDBDTestBase
{
  public:
    void run();

  private:
    void testIDBD();
    void testK1();
    void testAutostep();
};

#endif /* IDBDTEST_H_ */
