/*
 * Assert.h
 *
 *  Created on: Mar 25, 2014
 *      Author: sam
 */

#ifndef CUSTOM_ASSERT_H_
#define CUSTOM_ASSERT_H_

//*****************************************************************************
//
// The ASSERT macro, which does the actual assertion checking.  Typically, this
// will be for procedure arguments.
//
//*****************************************************************************

// Energia binding
#if defined(ENERGIA)
#define EMBEDDED_MODE
#endif

#if defined(EMBEDDED_MODE)
#include "Energia.h"
#else
#include <cstdio>
#include <cstdlib>
#endif

// Visual Studio 2013
#ifdef _MSC_VER
#pragma warning(disable:4996)
#define _USE_MATH_DEFINES // for C++
#define __func__ __FUNCTION__
#endif

#if defined(NDEBUG)
#define ASSERT(expr)   ((void)0)
#else
#define ASSERT(expr) do                                                       \
                     {                                                        \
                         if(!(expr))                                          \
                         {                                                    \
                             __ASSERT(__func__, __FILE__, __LINE__, #expr)    \
                         }                                                    \
                     }                                                        \
                     while(0) /*I am forcing the user to put the ";"*/
#endif /* NDEBUG */

#if defined(EMBEDDED_MODE)
#define __ASSERT(__func, __file, __lineno, __sexp)                                         \
  do                                                                                       \
  {                                                                                        \
      /*TODO: FixMe*/                                                                      \
  }                                                                                        \
  while(0);
#else

#define __ASSERT(__func, __file, __lineno, __sexp)                                                              \
  printf("Failed assertion! __func=%s __file=%s  __lineno=%u __sexp=%s \n", __func, __file, __lineno, __sexp);  \
  abort();
#endif

#endif /* CUSTOM_ASSERT_H_ */
