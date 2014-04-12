/*
 * MurmurHash2Test.cpp
 *
 *  Created on: Sep 3, 2013
 *      Author: sam
 */

#include "MurmurHash2Test.h"

void MurmurHash2Test::testChangingSeed()
{
  Random<double>* random = new Random<double>;
  MurmurHashing<double>* murmurHashing = new MurmurHashing<double>(random, 0);
  unsigned char key[] = { 0x4E, 0xE3, 0x91, 0x00, 0x10, 0x8F, 0xFF };
  unsigned int expected[] = { 0xeef8be32, 0x8109dec6, 0x9aaf4192, 0xc1bcaf1c, 0x821d2ce4,
      0xd45ed1df, 0x6c0357a7, 0x21d4e845, 0xfa97db50, 0x2f1985c8, 0x5d69782a, 0x0d6e4b85,
      0xe7d9cf6b, 0x337e6b49, 0xe1606944, 0xccc18ae8 };
  for (unsigned int i = 0; i < Arrays::length(expected); i++)
  {
    unsigned int expectedHash = expected[i];
    unsigned int hash = murmurHashing->MurmurHashNeutral2(key, Arrays::length(key), i);
    std::cout << "i=" << i << " " << expectedHash << " " << hash << std::endl;
    ASSERT(expectedHash == hash);
  }

  delete random;
  delete murmurHashing;
}

void MurmurHash2Test::testChangingKey()
{
  Random<double>* random = new Random<double>;
  MurmurHashing<double>* murmurHashing = new MurmurHashing<double>(random, 0);
  unsigned char key[133] = { 0 };
  unsigned int expected[] = { 0xd743ae0b, 0xf1b461c6, 0xa45a6ceb, 0xdb15e003, 0x877721a4,
      0xc30465f1, 0xfb658ba4, 0x1adf93b2, 0xe40a7931, 0x3da52db0, 0xbf523511, 0x1efaf273,
      0xe628c1dd, 0x9a0344df, 0x901c99fc, 0x5ae1aa44 };

  for (unsigned int i = 0; i < 16; i++)
  {
    setKey(key, Arrays::length(key), i);
    unsigned int expectedHash = expected[i];
    unsigned int hash = murmurHashing->MurmurHashNeutral2(key, Arrays::length(key), 0x1234ABCD);
    std::cout << "i=" << i << " " << expectedHash << " " << hash << std::endl;
    ASSERT(expectedHash == hash);
  }
  delete random;
  delete murmurHashing;
}

void MurmurHash2Test::testChangingKeyLength()
{
  Random<double>* random = new Random<double>;
  MurmurHashing<double>* murmurHashing = new MurmurHashing<double>(random, 0);
  unsigned int expected[] = { 0xa0c72f8e, 0x29c2f97e, 0x00ca8bba, 0x88387876, 0xe203ce49,
      0x58d75952, 0xab84febe, 0x98153c65, 0xcbb38375, 0x6ea1a28b, 0x9afa8f55, 0xfb890eb6,
      0x9516cc49, 0x6408a8eb, 0xbb12d3e6, 0x00fb7519 };
  for (unsigned int i = 0; i < 16; i++)
  {
    unsigned char* key = new unsigned char[i];
    std::fill(key, key + i, 0);
    setKey(key, i, i);
    unsigned int expectedHash = expected[i];
    unsigned int hash = murmurHashing->MurmurHashNeutral2(key, i, 0x7870AAFF);
    std::cout << "i=" << i << " " << expectedHash << " " << hash << std::endl;
    ASSERT(expectedHash == hash);
  }
  delete random;
  delete murmurHashing;
}

void MurmurHash2Test::setKey(unsigned char* key, unsigned int keyLength, int start)
{
  for (unsigned int i = 0; i < keyLength; i++)
    key[i] = (unsigned char) (start + (i & 0xFF));
}

void MurmurHash2Test::run()
{
  testChangingSeed();
  testChangingKey();
  testChangingKeyLength();
}

RLLIB_TEST_MAKE(MurmurHash2Test)

