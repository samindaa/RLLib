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
 * External documentation and recommendations on the use of this code is
 * available at http://www.cs.umass.edu/~rich/tiles.html.
 *
 * This is an implementation of grid-style tile codings, based originally on
 * the UNH CMAC code (see http://www.ece.unh.edu/robots/cmac.htm).
 * Here we provide a procedure, "GetTiles", that maps floating-point and integer
 * variables to a list of tiles. This function is memoryless and requires no
 * setup. We assume that hashing colisions are to be ignored. There may be
 * duplicates in the list of tiles, but this is unlikely if memory-size is
 * large.
 *
 * The floating-point input variables will be gridded at unit intervals, so generalization
 * will be by 1 in each direction, and any scaling will have
 * to be done externally before calling tiles.  There is no generalization
 * across integer values.
 *
 * It is recommended by the UNH folks that num-tilings be a power of 2, e.g., 16.
 *
 * We assume the existence of a function "rand()" that produces successive
 * random integers, of which we use only the low-order bytes.
 *
 * Modified by: Saminda Abeyruwan 
 * To be used as a single header file.
 */

#ifndef _TILES_H_
#define _TILES_H_

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>

#include "Vector.h"

namespace RLLib
{

// ============================================================================
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
      assert(write(file, (char * ) &m, sizeof(long)));
      assert(write(file, (char * ) &safe, sizeof(int)));
      assert(write(file, (char * ) &calls, sizeof(long)));
      assert(write(file, (char * ) &clearhits, sizeof(long)));
      assert(write(file, (char * ) &collisions, sizeof(long)));
      assert(write(file, (char * ) data, m * sizeof(long)));
    }

    void restore(int file)
    {
      assert(read(file, (char * ) &m, sizeof(long)));
      assert(read(file, (char * ) &safe, sizeof(int)));
      assert(read(file, (char * ) &calls, sizeof(long)));
      assert(read(file, (char * ) &clearhits, sizeof(long)));
      assert(read(file, (char * ) &collisions, sizeof(long)));
      assert(read(file, (char * ) data, m * sizeof(long)));
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

// ============================================================================
template<class T>
class Tiles
{
  protected:
    int qstate[Hashing::MAX_NUM_VARS];
    int base[Hashing::MAX_NUM_VARS];
    int wrap_widths_times_num_tilings[Hashing::MAX_NUM_VARS];
    int coordinates[Hashing::MAX_NUM_VARS * 2 + 1]; /* one interval number per relevant dimension */

    Hashing* hashing; /*The has function*/

    Vector<int>* i_tmp_arr;
    Vector<T>* f_tmp_arr;

  public:
    Tiles(Hashing* hashing) :
        hashing(hashing), i_tmp_arr(new PVector<int>(Hashing::MAX_NUM_VARS)), f_tmp_arr(
            new PVector<T>(Hashing::MAX_NUM_VARS))
    {
    }

    ~Tiles()
    {
      delete i_tmp_arr;
      delete f_tmp_arr;
    }

    void tiles(Vector<T>* the_tiles,        // provided array contains returned tiles (tile indices)
        int num_tilings,           // number of tile indices to be returned in tiles
        const Vector<T>* floats,            // array of floating point variables
        int num_floats, // number of active floating point variables
        const Vector<int>* ints,                   // array of integer variables
        int num_ints)              // number of integer variables
    {
      int i, j;
      int num_coordinates = num_floats + num_ints + 1;
      for (int i = 0; i < num_ints; i++)
        coordinates[num_floats + 1 + i] = ints->getEntry(i);

      /* quantize state to integers (henceforth, tile widths == num_tilings) */
      for (i = 0; i < num_floats; i++)
      {
        qstate[i] = (int) floor(floats->getEntry(i) * num_tilings);
        base[i] = 0;
      }

      /*compute the tile numbers */
      for (j = 0; j < num_tilings; j++)
      {

        /* loop over each relevant dimension */
        for (i = 0; i < num_floats; i++)
        {

          /* find coordinates of activated tile in tiling space */
          if (qstate[i] >= base[i])
            coordinates[i] = qstate[i] - ((qstate[i] - base[i]) % num_tilings);
          else
            coordinates[i] = qstate[i] + 1 + ((base[i] - qstate[i] - 1) % num_tilings)
                - num_tilings;

          /* compute displacement of next tiling in quantized space */
          base[i] += 1 + (2 * i);
        }
        /* add additional indices for tiling and hashing_set so they hash differently */
        coordinates[i] = j;

        the_tiles->setEntry(hashing->hash(coordinates, num_coordinates), 1.0f);
      }
    }

    void tiles(Vector<T>* the_tiles,        // provided array contains returned tiles (tile indices)
        int num_tilings,           // number of tile indices to be returned in tiles
        const Vector<T>* floats,            // array of floating point variables
        const Vector<int>* ints,                   // array of integer variables
        int num_ints)              // number of integer variables
    {
      tiles(the_tiles, num_tilings, floats, floats->dimension(), ints, num_ints);
    }

// No ints
    void tiles(Vector<T>* the_tiles, int nt, const Vector<T>* floats)
    {
      tiles(the_tiles, nt, floats, i_tmp_arr, 0);
    }

//one int
    void tiles(Vector<T>* the_tiles, int nt, const Vector<T>* floats, int h1)
    {
      i_tmp_arr->setEntry(0, h1);
      tiles(the_tiles, nt, floats, i_tmp_arr, 1);
    }

// two ints
    void tiles(Vector<T>* the_tiles, int nt, const Vector<T>* floats, int h1, int h2)
    {
      i_tmp_arr->setEntry(0, h1);
      i_tmp_arr->setEntry(1, h2);
      tiles(the_tiles, nt, floats, i_tmp_arr, 2);
    }

// three ints
    void tiles(Vector<T>* the_tiles, int nt, const Vector<T>* floats, int h1, int h2, int h3)
    {
      i_tmp_arr->setEntry(0, h1);
      i_tmp_arr->setEntry(1, h2);
      i_tmp_arr->setEntry(2, h3);
      tiles(the_tiles, nt, floats, i_tmp_arr, 3);
    }

// one float, No ints
    void tiles1(Vector<T>* the_tiles, int nt, const T& f1)
    {
      f_tmp_arr->setEntry(0, f1);
      tiles(the_tiles, nt, f_tmp_arr, 1, i_tmp_arr, 0);
    }

// one float, one int
    void tiles1(Vector<T>* the_tiles, int nt, const T& f1, int h1)
    {
      f_tmp_arr->setEntry(0, f1);
      i_tmp_arr->setEntry(0, h1);
      tiles(the_tiles, nt, f_tmp_arr, 1, i_tmp_arr, 1);
    }

// one float, two ints
    void tiles1(Vector<T>* the_tiles, int nt, const T& f1, int h1, int h2)
    {
      f_tmp_arr->setEntry(0, f1);
      i_tmp_arr->setEntry(0, h1);
      i_tmp_arr->setEntry(1, h2);
      tiles(the_tiles, nt, f_tmp_arr, 1, i_tmp_arr, 2);
    }

// one float, three ints
    void tiles1(Vector<T>* the_tiles, int nt, const T& f1, int h1, int h2, int h3)
    {
      f_tmp_arr->setEntry(0, f1);
      i_tmp_arr->setEntry(0, h1);
      i_tmp_arr->setEntry(1, h2);
      i_tmp_arr->setEntry(2, h3);
      tiles(the_tiles, nt, f_tmp_arr, 1, i_tmp_arr, 3);
    }

// two floats, No ints
    void tiles2(Vector<T>* the_tiles, int nt, const T& f1, const T& f2)
    {
      f_tmp_arr->setEntry(0, f1);
      f_tmp_arr->setEntry(1, f2);
      tiles(the_tiles, nt, f_tmp_arr, 2, i_tmp_arr, 0);
    }

// two floats, one int
    void tiles2(Vector<T>* the_tiles, int nt, const T& f1, const T& f2, int h1)
    {
      f_tmp_arr->setEntry(0, f1);
      f_tmp_arr->setEntry(1, f2);
      i_tmp_arr->setEntry(0, h1);
      tiles(the_tiles, nt, f_tmp_arr, 2, i_tmp_arr, 1);
    }

// two floats, two ints
    void tiles2(Vector<T>* the_tiles, int nt, const T& f1, const T& f2, int h1, int h2)
    {
      f_tmp_arr->setEntry(0, f1);
      f_tmp_arr->setEntry(1, f2);
      i_tmp_arr->setEntry(0, h1);
      i_tmp_arr->setEntry(1, h2);
      tiles(the_tiles, nt, f_tmp_arr, 2, i_tmp_arr, 2);
    }

// two floats, three ints
    void tiles2(Vector<T>* the_tiles, int nt, const T& f1, const T& f2, int h1, int h2, int h3)
    {
      f_tmp_arr->setEntry(0, f1);
      f_tmp_arr->setEntry(1, f2);
      i_tmp_arr->setEntry(0, h1);
      i_tmp_arr->setEntry(1, h2);
      i_tmp_arr->setEntry(2, h3);
      tiles(the_tiles, nt, f_tmp_arr, 2, i_tmp_arr, 3);
    }

    void tileswrap(Vector<T>* the_tiles,    // provided array contains returned tiles (tile indices)
        int num_tilings,           // number of tile indices to be returned in tiles
        const Vector<T>* floats,            // array of floating point variables
        int num_floats, // number of active floating point variables
        int wrap_widths[],         // array of widths (length and units as in floats)
        const Vector<int>* ints,                  // array of integer variables
        int num_ints)             // number of integer variables
    {
      int i, j;
      int num_coordinates = num_floats + num_ints + 1;

      for (int i = 0; i < num_ints; i++)
        coordinates[num_floats + 1 + i] = ints->getEntry(i);

      /* quantize state to integers (henceforth, tile widths == num_tilings) */
      for (i = 0; i < num_floats; i++)
      {
        qstate[i] = (int) floor(floats->getEntry(i) * num_tilings);
        base[i] = 0;
        wrap_widths_times_num_tilings[i] = wrap_widths[i] * num_tilings;
      }

      /*compute the tile numbers */
      for (j = 0; j < num_tilings; j++)
      {

        /* loop over each relevant dimension */
        for (i = 0; i < num_floats; i++)
        {

          /* find coordinates of activated tile in tiling space */
          if (qstate[i] >= base[i])
            coordinates[i] = qstate[i] - ((qstate[i] - base[i]) % num_tilings);
          else
            coordinates[i] = qstate[i] + 1 + ((base[i] - qstate[i] - 1) % num_tilings)
                - num_tilings;
          if (wrap_widths[i] != 0)
            coordinates[i] = coordinates[i] % wrap_widths_times_num_tilings[i];
          if (coordinates[i] < 0)
          {
            while (coordinates[i] < 0)
              coordinates[i] += wrap_widths_times_num_tilings[i];
          }
          /* compute displacement of next tiling in quantized space */
          base[i] += 1 + (2 * i);
        }
        /* add additional indices for tiling and hashing_set so they hash differently */
        coordinates[i] = j;

        the_tiles->setEntry(hashing->hash(coordinates, num_coordinates), 1.0f);
      }
      return;
    }
};

} // namespace RLLib

#endif
