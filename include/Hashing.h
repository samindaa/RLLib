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
 * Hashing.h
 *
 *  Created on: Mar 30, 2014
 *      Author: sam
 */

#ifndef HASHING_H_
#define HASHING_H_

#include <limits.h>

#include "Mathema.h"

namespace RLLib
{

  template<typename T>
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

  template<typename T>
  class AbstractHashing: public Hashing<T>
  {
    protected:
      Random<T>* random;
      int memorySize;

    public:
      AbstractHashing(Random<T>* random, const int& memorySize) :
          random(random), memorySize(memorySize)
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

  template<typename T>
  class UNH: public AbstractHashing<T>
  {
    private:
      typedef AbstractHashing<T> Base;

    protected:
      int increment;
      unsigned int rndseq[16384]; // 2^14 (16384)  {old: 2048}

    public:
      UNH(Random<T>* random, const int& memorySize) :
          AbstractHashing<T>(random, memorySize), increment(470)
      {
        /*First call to hashing, initialize table of random numbers */
        //printf("inside tiles \n");
        //srand(0);
        for (int k = 0; k < 16384/*2048*/; k++)
        {
          rndseq[k] = 0;
          for (int i = 0; i < int(sizeof(int)); ++i)
            rndseq[k] = (rndseq[k] << 8) | (random->randu32() & 0xff);
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
        index = (int) (sum % Base::memorySize);
        while (index < 0)
          index += Base::memorySize;

        /* printf("index is %d \n", index); */
        return (index);
      }
  };

  template<typename T>
  class MurmurHashing: public AbstractHashing<T>
  {
    private:
      typedef AbstractHashing<T> Base;

    protected:
      uint32_t seed;
      uint32_t out;

    public:
      MurmurHashing(Random<T>* random, const int& memorySize) :
          AbstractHashing<T>(random, memorySize), seed(random->randu32()), out(0)
      {
      }

      virtual ~MurmurHashing()
      {
      }

    public:
      /**
       * https://code.google.com/p/smhasher/
       *
       * The main implementations of Murmur are written to be as clear as possible at the expense
       * of some cross-platform compatibility. Shane Day offered to put together an implementation
       * of Murmur3_x86_32 that should compile on virtually any platform and which passes the Murmur3
       * verification test, and I've now merged his code into the repository.
       */

      // Block read - if your platform needs to do endian-swapping or can only
      // handle aligned reads, do the conversion here
      uint32_t getblock32(const uint32_t * p, int i)
      {
        return p[i];
      }

      uint32_t ROTL32(uint32_t x, int8_t r)
      {
        return (x << r) | (x >> (32 - r));
      }

      // Finalization mix - force all bits of a hash block to avalanche
      uint32_t fmix32(uint32_t h)
      {
        h ^= h >> 16;
        h *= 0x85ebca6b;
        h ^= h >> 13;
        h *= 0xc2b2ae35;
        h ^= h >> 16;

        return h;
      }

      void MurmurHash3_x86_32(const void * key, int len, uint32_t seed, void * out)
      {
        const uint8_t * data = (const uint8_t*) key;
        const int nblocks = len / 4;

        uint32_t h1 = seed;

        const uint32_t c1 = 0xcc9e2d51;
        const uint32_t c2 = 0x1b873593;

        //----------
        // body

        const uint32_t * blocks = (const uint32_t *) (data + nblocks * 4);

        for (int i = -nblocks; i; i++)
        {
          uint32_t k1 = getblock32(blocks, i);

          k1 *= c1;
          k1 = ROTL32(k1, 15);
          k1 *= c2;

          h1 ^= k1;
          h1 = ROTL32(h1, 13);
          h1 = h1 * 5 + 0xe6546b64;
        }

        //----------
        // tail

        const uint8_t * tail = (const uint8_t*) (data + nblocks * 4);

        uint32_t k1 = 0;

        switch (len & 3)
        {
          case 3:
            k1 ^= tail[2] << 16;
          case 2:
            k1 ^= tail[1] << 8;
          case 1:
            k1 ^= tail[0];
            k1 *= c1;
            k1 = ROTL32(k1, 15);
            k1 *= c2;
            h1 ^= k1;
        };

        //----------
        // finalization

        h1 ^= len;

        h1 = fmix32(h1);

        *(uint32_t*) out = h1;
      }

      //-----------------------------------------------------------------------------
      // MurmurHashNeutral2, by Austin Appleby

      // Same as MurmurHash2, but endian- and alignment-neutral.
      // Half the speed though, alas.

      uint32_t MurmurHashNeutral2(const void * key, int len, uint32_t seed)
      {
        const uint32_t m = 0x5bd1e995;
        const int r = 24;

        uint32_t h = seed ^ len;

        const unsigned char * data = (const unsigned char *) key;

        while (len >= 4)
        {
          uint32_t k;

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

    public:
      int hash(int* ints/*coordinates*/, int num_ints)
      {
        MurmurHash3_x86_32(((const uint8_t*) ints), (num_ints * 4), seed, &out);
        return out % Base::memorySize;
      }

  };

  template<typename T>
  class ColisionDetection: public Hashing<T>
  {
    protected:
      Hashing<T>* hashing;
      Hashing<T>* referenceHashing;
      Hashing<T>* referenceHashing2;
      int safe;
      long calls;
      long clearhits;
      long collisions;
      long *data;
      long m;

    public:
      ColisionDetection(Hashing<T>* hashing, const int& size, const int& safety) :
          hashing(hashing), referenceHashing(new UNH<T>(INT_MAX)), //
          referenceHashing2(new UNH<T>(INT_MAX / 4)), safe(safe), calls(0), clearhits(0), //
          collisions(0), m(size)
      {
        if (size % 2 != 0)
        {
#if !defined(EMBEDDED_MODE)
          std::cerr << "Size of collision table must be power of 2 " << size << std::endl;
          exit(0);
#endif
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
#if !defined(EMBEDDED_MODE)
        printf("Collision table: Safety : %d Usage : %d Size : %ld Calls : %ld Collisions : %ld\n",
            this->safe, this->usage(), this->m, this->calls, this->collisions);
#endif
      }

      void save(int file)
      {
#if !defined(EMBEDDED_MODE)
        ASSERT(write(file, (char * ) &m, sizeof(long)));
        ASSERT(write(file, (char * ) &safe, sizeof(int)));
        ASSERT(write(file, (char * ) &calls, sizeof(long)));
        ASSERT(write(file, (char * ) &clearhits, sizeof(long)));
        ASSERT(write(file, (char * ) &collisions, sizeof(long)));
        ASSERT(write(file, (char * ) data, m * sizeof(long)));
#endif
      }

      void restore(int file)
      {
#if !defined(EMBEDDED_MODE)
        ASSERT(read(file, (char * ) &m, sizeof(long)));
        ASSERT(read(file, (char * ) &safe, sizeof(int)));
        ASSERT(read(file, (char * ) &calls, sizeof(long)));
        ASSERT(read(file, (char * ) &clearhits, sizeof(long)));
        ASSERT(read(file, (char * ) &collisions, sizeof(long)));
        ASSERT(read(file, (char * ) data, m * sizeof(long)));
#endif
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
#if !defined(EMBEDDED_MODE)
              printf("\nTiles: Collision table out of Memory");
#endif
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
