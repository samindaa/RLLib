/**************************************************************************
 *   File:                        rtnode.h                                 *
 *   Description:   Basic classes for Tree based algorithms                *
 *   Copyright (C) 2007 by  Walter Corno & Daniele Dell'Aglio              *
 ***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
#ifndef RTLEAFSAMPLE_H
#define RTLEAFSAMPLE_H
#include <cstdlib>
#include <string>
#include <fstream>
#include <iostream>

#include "rtLeaf.h"
#include "Dataset.h"

using namespace std;

//<< Sam
namespace PoliFitted
{

/**************************************************************************
 *   rtLeaf is a template class that rapresents a leaf of a regression     *
 *   tree.                                                                 *
 *   This class extends rtANode and contains methods to set/get the value  *
 *   saved in the node, this value is of type T                            *
 ***************************************************************************/
class rtLeafSample: public rtLeaf
{
  public:

    /**
     * Empty Constructor
     */
    rtLeafSample() :
        rtLeaf(), mpSample(NULL)
    {
    }

    /**
     * Basic constructor
     * @param val the value to store in the node
     */
    rtLeafSample(Dataset* cloned_dataset) :
        rtLeaf(cloned_dataset)
    {
      mpSample = cloned_dataset;
    }

    /**
     *
     */
    virtual void WriteOnStream(ofstream& out)
    {
      out << "LS" << endl;
      out << mValue << endl;
      out << *mpSample;
    }

    /**
     *
     */
    virtual void ReadFromStream(ifstream& in)
    {
      mpSample = new Dataset();
      in >> mValue;
      in >> *mpSample;
    }

    void SetSample(Dataset* ds)
    {
      mpSample = ds;
    }

    Dataset* GetSample()
    {
      return mpSample;
    }

  protected:

  private:
    Dataset* mpSample;
};

}

#endif // RTNODE_H
