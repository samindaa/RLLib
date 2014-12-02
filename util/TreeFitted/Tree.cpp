/************************************************************************
 Tree.h.cpp - Copyright marcello

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
 The original location of this file is /home/marcello/Projects/fitted/Developing/Tree.cpp
 **************************************************************************/

#include <cmath>
#include <iostream>

#include "Tree.h"

//<< Sam
using namespace PoliFitted;

// Constructors/Destructors
//

Tree::Tree(string type, unsigned int input_size, unsigned int output_size) :
    Regressor(type, input_size, output_size), root(0)
{
}

Tree::~Tree()
{
  if (!root)
  {
    delete root;
  }
}

//
// Methods
//

// Other methods
//

