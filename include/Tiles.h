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

#include <fcntl.h>
// Visual Studio 2013
#ifndef _MSC_VER
#include <unistd.h>
#endif
#include <stdint.h>
#include <string.h>

#include "Vector.h"
#include "Hashing.h"

namespace RLLib
{

  template<typename T>
  class Tiles
  {
    protected:
      int qstate[Hashing<T>::MAX_NUM_VARS];
      int base[Hashing<T>::MAX_NUM_VARS];
      int wrap_widths_times_num_tilings[Hashing<T>::MAX_NUM_VARS];
      int coordinates[Hashing<T>::MAX_NUM_VARS * 2 + 1]; /* one interval number per relevant dimension */

      Hashing<T>* hashing; /*The has function*/

      Vector<int>* i_tmp_arr;
      Vector<T>* f_tmp_arr;

    public:
      Tiles(Hashing<T>* hashing) :
          hashing(hashing), i_tmp_arr(new PVector<int>(Hashing<T>::MAX_NUM_VARS)), //
          f_tmp_arr(new PVector<T>(Hashing<T>::MAX_NUM_VARS))
      {
      }

      ~Tiles()
      {
        delete i_tmp_arr;
        delete f_tmp_arr;
      }

      void tiles(Vector<T>* the_tiles,      // provided array contains returned tiles (tile indices)
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

      void tiles(Vector<T>* the_tiles,      // provided array contains returned tiles (tile indices)
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

      void tileswrap(Vector<T>* the_tiles,  // provided array contains returned tiles (tile indices)
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
