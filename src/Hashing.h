/*
 * Copyright 2014 Saminda Abeyruwan (saminda@cs.miami.edu)
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
 * Hashing.h
 *
 *  Created on: Mar 30, 2014
 *      Author: sam
 */

#ifndef HASHING_H_
#define HASHING_H_

#include <limits.h>
#include "Assert.h"

namespace RLLib
{

class Hashing
{
  public:
    enum
    {
      MAX_NUM_VARS = 20          // Maximum number of variables in a grid-tiling
    };
    virtual ~Hashing()
    {
    }
    virtual int hash(int* ints/*coordinates*/, int num_ints) =0;
    virtual int getMemorySize() const =0;
};

class AbstractHashing: public Hashing
{
  protected:
    int memorySize;

  public:
    AbstractHashing(const int& memorySize) :
        memorySize(memorySize)
    {
    }

    virtual ~AbstractHashing()
    {
    }

    int getMemorySize() const
    {
      return memorySize;
    }
};

class UNH: public AbstractHashing
{
  protected:
    int increment;
    unsigned int rndseq[16384]; // 2^14 (16384)  {old: 2048}

  public:
    UNH(const int& memorySize) :
        AbstractHashing(memorySize), increment(470)
    {
      /*First call to hashing, initialize table of random numbers */
      //printf("inside tiles \n");
      //srand(0);
      for (int k = 0; k < 16384/*2048*/; k++)
      {
        rndseq[k] = 0;
        for (int i = 0; i < int(sizeof(int)); ++i)
          rndseq[k] = (rndseq[k] << 8) | (rand() & 0xff);
      }
      //srand(time(0));
    }

    /** hash_UNH
     *  Takes an array of integers and returns the corresponding tile after hashing
     */
    int hash(int* ints/*coordinates*/, int num_ints)
    {
      long index;
      long sum = 0;

      for (int i = 0; i < num_ints; i++)
      {
        /* add random table offset for this dimension and wrap around */
        index = ints[i];
        index += (increment * i);
        /* index %= 2048; */
        index = index & 16383/*2047*/;
        while (index < 0)
          index += 16384/*2048*/;

        /* add selected random number to sum */
        sum += (long) rndseq[(int) index];
      }
      index = (int) (sum % memorySize);
      while (index < 0)
        index += memorySize;

      /* printf("index is %d \n", index); */
      return (index);
    }
};

class MurmurHashing: public AbstractHashing
{
  protected:
    unsigned int seed;
    uint8_t* key;
  public:
    MurmurHashing(const int& memorySize) :
        AbstractHashing(memorySize)
    {
      // Constant seed
      //srand(0);
      seed = (unsigned int) rand();
      //srand(time(0));
      key = new uint8_t[(MAX_NUM_VARS * 2 + 1) * 4]; //<< Arbitrary
      ::memset(key, 0, (MAX_NUM_VARS * 2 + 1) * 4);
    }

    virtual ~MurmurHashing()
    {
      delete[] key;
    }

  public:
    /**
     * MurmurHashNeutral2, by Austin Appleby
     * https://sites.google.com/site/murmurhash/
     * https://sites.google.com/site/murmurhash/MurmurHashNeutral2.cpp?attredirects=0
     * Same as MurmurHash2, but endian- and alignment-neutral.
     * Half the speed though, alas.
     */
    unsigned int murmurHashNeutral2(const void* key, int len, unsigned int seed)
    {
      // 'm' and 'r' are mixing constants generated off-line.
      // They're not really 'magic', they just happen to work well.

      const static unsigned int m = 0x5bd1e995;
      const static int r = 24;

      unsigned int h = seed ^ len;

      const unsigned char* data = (const unsigned char*) key;

      while (len >= 4)
      {
        unsigned int k;

        k = data[0];
        k |= data[1] << 8;
        k |= data[2] << 16;
        k |= data[3] << 24;

        k *= m;
        k ^= k >> r;
        k *= m;

        h *= m;
        h ^= k;

        data += 4;
        len -= 4;
      }

      switch (len)
      {
      case 3:
        h ^= data[2] << 16;
      case 2:
        h ^= data[1] << 8;
      case 1:
        h ^= data[0];
        h *= m;
      };

      h ^= h >> 13;
      h *= m;
      h ^= h >> 15;

      return h;
    }

  private:
    void pack(const uint32_t& val, uint8_t* dest)
    {
      dest[0] = (val & 0xff000000) >> 24;
      dest[1] = (val & 0x00ff0000) >> 16;
      dest[2] = (val & 0x0000ff00) >> 8;
      dest[3] = (val & 0x000000ff);
    }
  public:
    int hash(int* ints/*coordinates*/, int num_ints)
    {
      for (int i = 0; i < num_ints; i++)
        pack((uint32_t) ints[i], &key[i * 4]);
      return (int) (murmurHashNeutral2(key, (num_ints * 4), seed) % memorySize);
    }

};

class ColisionDetection: public Hashing
{
  protected:
    Hashing* hashing;
    Hashing* referenceHashing;
    Hashing* referenceHashing2;
    int safe;
    long calls;
    long clearhits;
    long collisions;
    long *data;
    long m;

  public:
    ColisionDetection(Hashing* hashing, const int& size, const int& safety) :
        hashing(hashing), referenceHashing(new UNH(INT_MAX)), referenceHashing2(
            new UNH(INT_MAX / 4)), safe(safe), calls(0), clearhits(0), collisions(0), m(size)
    {
      if (size % 2 != 0)
      {
        std::cerr << "Size of collision table must be power of 2 " << size << std::endl;
        exit(0);
      }
      data = new long[size];
      for (long* i = data; i < data + size; ++i)
        *i = -1;
    }

    ~ColisionDetection()
    {
      delete[] data;
      delete referenceHashing;
      delete referenceHashing2;
    }

    int usage()
    {
      int count = 0;
      for (int i = 0; i < m; i++)
      {
        if (data[i] != -1)
          count++;
      }
      return count;
    }

    void print()
    {
      printf("Collision table: Safety : %d Usage : %d Size : %ld Calls : %ld Collisions : %ld\n",
          this->safe, this->usage(), this->m, this->calls, this->collisions);
    }

    void save(int file)
    {
      ASSERT(write(file, (char * ) &m, sizeof(long)));
      ASSERT(write(file, (char * ) &safe, sizeof(int)));
      ASSERT(write(file, (char * ) &calls, sizeof(long)));
      ASSERT(write(file, (char * ) &clearhits, sizeof(long)));
      ASSERT(write(file, (char * ) &collisions, sizeof(long)));
      ASSERT(write(file, (char * ) data, m * sizeof(long)));
    }

    void restore(int file)
    {
      ASSERT(read(file, (char * ) &m, sizeof(long)));
      ASSERT(read(file, (char * ) &safe, sizeof(int)));
      ASSERT(read(file, (char * ) &calls, sizeof(long)));
      ASSERT(read(file, (char * ) &clearhits, sizeof(long)));
      ASSERT(read(file, (char * ) &collisions, sizeof(long)));
      ASSERT(read(file, (char * ) data, m * sizeof(long)));
    }

    void save(const char *filename)
    {
      //write(open(filename, O_BINARY | O_CREAT | O_WRONLY);
    }

    void restore(const char *filename)
    {
      //read(open(filename, O_BINARY | O_CREAT | O_WRONLY);
    }

    int hash(int* ints/*coordinates*/, int num_ints)
    {
      int j;
      long ccheck;

      calls++;
      j = hashing->hash(ints, num_ints);
      ccheck = referenceHashing->hash(ints, num_ints);
      if (ccheck == data[j])
        clearhits++;
      else if (data[j] == -1)
      {
        clearhits++;
        data[j] = ccheck;
      }
      else if (safe == 0)
        collisions++;
      else
      {
        long h2 = 1 + 2 * referenceHashing2->hash(ints, num_ints);
        int i = 0;
        while (++i)
        {
          collisions++;
          j = (j + h2) % (this->m);
          /*printf("collision (%d) \n",j);*/
          if (i > this->m)
          {
            printf("\nTiles: Collision table out of Memory");
            break/*exit(0) <<@ Sam Abeyruwan*/;
          }
          if (ccheck == data[j])
            break;
          if (data[j] == -1)
          {
            data[j] = ccheck;
            break;
          }
        }
      }
      return j;
    }
};

}  // namespace RLLib

#endif /* HASHING_H_ */
