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
 * Assert.h
 *
 *  Created on: Mar 30, 2014
 *      Author: sam
 */

#ifndef RLLIB_ASSERT_H_
#define RLLIB_ASSERT_H_

//*****************************************************************************
//
// The ASSERT macro, which does the actual assertion checking.  Typically, this
// will be for procedure arguments.
//
//*****************************************************************************
#include <cstdio>
#include <cstdlib>

#if defined(NDEBUG)
#define ASSERT(expr)   ((void)0)
#else
#define ASSERT(expr) do                                                       \
                     {                                                        \
                         if(!(expr))                                          \
                         {                                                    \
                             __ASSERT(__func__, __FILE__, __LINE__, #expr);   \
                         }                                                    \
                     }                                                        \
                     while(0)
#endif /* NDEBUG */


#define __ASSERT(__func, __file, __lineno, __sexp)                                                              \
  printf("Failed assertion! __func=%s __file=%s  __lineno=%u __sexp=%s \n", __func, __file, __lineno, __sexp);  \
  abort();

#endif /* RLLIB_ASSERT_H_ */
