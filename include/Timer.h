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
 * Timer.h
 *
 *  Created on: Aug 5, 2013
 *      Author: sam
 */

#ifndef TIMER_H_
#define TIMER_H_

// Visual Studio 2013
#ifdef _MSC_VER
#include <windows.h>
#else
#include <sys/time.h>
#endif
#include <stdlib.h>

namespace RLLib
{
  class Timer
  {
    public:
      Timer()
      {
#ifdef _MSC_VER
        QueryPerformanceFrequency(&frequency);
        startCount.QuadPart = 0;
        endCount.QuadPart = 0;
#else
        startCount.tv_sec = startCount.tv_usec = 0;
        endCount.tv_sec = endCount.tv_usec = 0;
#endif

        stopped = 0;
        startTimeInMicroSec = 0;
        endTimeInMicroSec = 0;
      }

      ~Timer()
      {
      }

      void start()
      {
        stopped = 0; // reset stop flag
#ifdef _MSC_VER
            QueryPerformanceCounter(&startCount);
#else
        gettimeofday(&startCount, NULL);
#endif
      }

      void stop()
      {
        stopped = 1; // set timer stopped flag

#ifdef _MSC_VER
        QueryPerformanceCounter(&endCount);
#else
        gettimeofday(&endCount, NULL);
#endif
      }

      double getElapsedTimeInMicroSec()
      {
#ifdef _MSC_VER
        if(!stopped)
        QueryPerformanceCounter(&endCount);

        startTimeInMicroSec = startCount.QuadPart * (1000000.0 / frequency.QuadPart);
        endTimeInMicroSec = endCount.QuadPart * (1000000.0 / frequency.QuadPart);
#else
        if (!stopped)
          gettimeofday(&endCount, NULL);

        startTimeInMicroSec = (startCount.tv_sec * 1000000.0) + startCount.tv_usec;
        endTimeInMicroSec = (endCount.tv_sec * 1000000.0) + endCount.tv_usec;
#endif

        return endTimeInMicroSec - startTimeInMicroSec;
      }

      double getElapsedTimeInSec()
      {
        return this->getElapsedTimeInMicroSec() * 0.000001;
      }

      double getElapsedTime()
      {
        return this->getElapsedTimeInSec();
      }

      double getElapsedTimeInMilliSec()
      {
        return this->getElapsedTimeInMicroSec() * 0.001;
      }

    protected:

    private:
      double startTimeInMicroSec;
      double endTimeInMicroSec;
      int stopped;
#ifdef _MSC_VER
      LARGE_INTEGER frequency;                    // ticks per second
      LARGE_INTEGER startCount;//
      LARGE_INTEGER endCount;//
#else
      timeval startCount;                         //
      timeval endCount;                           //
#endif
  };

}  // namespace RLLib

#endif /* TIMER_H_ */
