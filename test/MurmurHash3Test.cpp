/*
 * MurmurHash3Test.cpp
 *
 *  Created on: Apr 8, 2014
 *      Author: sam
 */

#include "MurmurHash3Test.h"

// Based on https://code.google.com/p/smhasher/
void MurmurHash3Test::verificationTest()
{
  MurmurHashing* hash = new MurmurHashing(0);

  const int hashbits = 32;
  uint32_t expected = 0xB0F57EE3;

  const int hashbytes = hashbits / 8;

  uint8_t * key = new uint8_t[256];
  uint8_t * hashes = new uint8_t[hashbytes * 256];
  uint32_t verification = 0;

  memset(key, 0, 256);
  memset(hashes, 0, hashbytes * 256);

  // Hash keys of the form {0}, {0,1}, {0,1,2}... up to N=255,using 256-N as
  // the seed

  for (int i = 0; i < 256; i++)
  {
    key[i] = (uint8_t) i;
    uint32_t val = hash->MurmurHash3_x86_32(key, i, 256 - i);
    // unpack
    hashes[i * hashbytes + 3] = (val & 0xff000000) >> 24;
    hashes[i * hashbytes + 2] = (val & 0x00ff0000) >> 16;
    hashes[i * hashbytes + 1] = (val & 0x0000ff00) >> 8;
    hashes[i * hashbytes + 0] = (val & 0x000000ff);
  }

  // Then hash the result array

  verification = hash->MurmurHash3_x86_32(hashes, hashbytes * 256, 0);

  // The first four bytes of that hash, interpreted as a little-endian integer, is our
  // verification value

  delete hash;
  delete[] key;
  delete[] hashes;

  //----------

  if (expected != verification)
  {
    printf("Verification value 0x%08X : Failed! (Expected 0x%08x)\n", verification, expected);
    ASSERT(false);
  }
  else
  {
    printf("Verification value 0x%08X : Passed!\n", verification);
    ASSERT(true);
  }
}

void MurmurHash3Test::run()
{
  verificationTest();
}

RLLIB_TEST_MAKE(MurmurHash3Test)

