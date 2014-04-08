/*
 * MurmurHash3Test.h
 *
 *  Created on: Apr 8, 2014
 *      Author: sam
 */

#ifndef MURMURHASH3TEST_H_
#define MURMURHASH3TEST_H_

#include "Test.h"

RLLIB_TEST(MurmurHash3Test)

class MurmurHash3Test : public MurmurHash3TestBase
{
  public:
    void verificationTest();
    void run();
};

#endif /* MURMURHASH3TEST_H_ */
