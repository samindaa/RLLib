/*
 * MurmurHash2Test.h
 *
 *  Created on: Sep 3, 2013
 *      Author: sam
 */

#ifndef MURMURHASH2TEST_H_
#define MURMURHASH2TEST_H_

#include "Test.h"
#include "Tiles.h"

RLLIB_TEST(MurmurHash2Test)

class MurmurHash2Test: public MurmurHash2TestBase
{
  public:
    MurmurHash2Test() {}
    ~MurmurHash2Test() {}
    void run();

  private:
    void testChangingSeed();
    void testChangingKey();
    void testChangingKeyLength();
    void setKey(unsigned char* key, unsigned int keyLength, int start);
};

#endif /* MURMURHASH2TEST_H_ */
