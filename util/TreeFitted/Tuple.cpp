/************************************************************************
 Tuple.h.cpp - Copyright marcello

 Here you can write a license for your code, some comments or any other
 information you want to have in your generated code. To to this simply
 configure the "headings" directory in uml to point to a directory
 where you have your heading files.

 or you can just replace the contents of this file with your own.
 If you want to do this, this file is located at

 /usr/share/apps/umbrello/headings/heading.cpp

 -->Code Generators searches for heading files based on the file extension
 i.e. it will look for a file name ending in ".h" to include in C++ header
 files, and for a file name ending in ".java" to include in all generated
 java code.
 If you name the file "heading.<extension>", Code Generator will always
 choose this file even if there are other files with the same extension in the
 directory. If you name the file something else, it must be the only one with that
 extension in the directory to guarantee that Code Generator will choose it.

 you can use variables in your heading files which are replaced at generation
 time. possible variables are : author, date, time, filename and filepath.
 just write %variable_name%

 This file was generated on Sat Nov 10 2007 at 15:05:38
 The original location of this file is /home/marcello/Projects/fitted/Developing/Tuple.cpp
 **************************************************************************/

#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <cstdarg>

#include "Tuple.h"

//<< Sam
using namespace PoliFitted;

/*
Tuple* Tuple::CONVERT(unsigned int count, ...)
{
  Tuple* t = new Tuple(count);
  std::vector<double>::const_iterator it;
  std::vector<double>::const_iterator end;
  unsigned int p = 0;

  va_list vl;
  va_start(vl, count);

  while (p < count)
  {
    it = va_arg(vl, std::vector<double>::const_iterator);
    end = va_arg(vl, std::vector<double>::const_iterator);
    for (; it != end; ++it)
    {
      (*t)[p++] = *it;
    }
  }

  va_end(vl);
  return t;
}
*/

// Constructors/Destructors
//

Tuple::Tuple() :
    mValues(0)
{
}

Tuple::Tuple(unsigned int size) :
    mValues(new float[size])
{

}

Tuple::~Tuple()
{
  delete[] mValues;
}

Tuple* Tuple::Clone(unsigned int input_size)
{
  Tuple* values = new Tuple(input_size);
  for (unsigned int i = 0; i < input_size; i++)
  {
    (*values)[i] = mValues[i];
  }
  return values;
}

void Tuple::Resize(unsigned int new_size)
{
  mValues = (float*) realloc(mValues, new_size * sizeof(float));
}

//
// Methods
//

float Tuple::Distance(Tuple* t, unsigned int size, NormType type)
{
  float distance = 0.0;
  for (unsigned int i = 0; i < size; i++)
  {
    float d_i = mValues[i] - (*t)[i];
    switch (type)
    {
    case L1:
      distance += fabs(d_i);
      break;
    case L2:
      distance += d_i * d_i;
      break;
    case Linfty:
      distance = (d_i > distance) ? d_i : distance;
      break;
    default:
      cerr << "ERROR: Undefined norm type!" << endl;
      exit(0);
    }
  }

  if (L2 == type)
  {
    distance = sqrt(distance);
  }
  return distance;
}

// Accessor methods
//

