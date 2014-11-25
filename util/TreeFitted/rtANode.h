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
#ifndef RTANODE_H
#define RTANODE_H
#include <cstdlib>
#include <string>
#include <fstream>
#include <iostream>

#include "Tuple.h"

using namespace std;

//<< Sam
namespace PoliFitted
{

/**************************************************************************
 *   rtANode is a class that rapresents an abstract node of a regression   *
 *   tree. The method isLeaf() is used to determine if it is a leaf or an  *
 *   internal node.                                                        *
 ***************************************************************************/
class rtANode
{
  public:

    /**
     * Empty Constructor
     */
    rtANode()
    {
    }

    /**
     * Empty Destructor
     */
    virtual ~rtANode()
    {
    }

    /**
     * This method is used to determine if the object is a leaf or an
     * internal node
     * @return true if it is a leaf, false otherwise
     */
    virtual bool isLeaf();

    /**
     * Get axis, axis is the index of the split
     * @return the axis
     */
    virtual int getAxis()
    {
      return -1;
    }

    /**
     * Get Split
     * @return the split value
     */
    virtual float getValue(Tuple* input = 0)
    {
      if (input == 0)
        return -1;
      return -1;
    }

    /**
     * Get Split
     * @return the split value
     */
    virtual float getSplit()
    {
      return -1;
    }

    /**
     * Get Left Child
     * @return a pointer to the left chid node
     */
    virtual rtANode* getLeft()
    {
      return NULL;
    }

    /**
     * Get Right Child
     * @return a pointer to the right child node
     */
    virtual rtANode* getRight()
    {
      return NULL;
    }

    /**
     * Get Child
     * @return a pointer to the right child node
     */
    virtual rtANode* getChild(float* values = 0)
    {
      if (values == 0)
        return NULL;
      return NULL;
    }

    /**
     *
     */
    virtual void WriteOnStream(ofstream& out) = 0;

    /**
     *
     */
    virtual void ReadFromStream(ifstream& in) = 0;

    /**
     *
     */
    friend ofstream& operator<<(ofstream& out, rtANode& n)
    {
      n.WriteOnStream(out);
      return out;
    }

    /**
     *
     */
    friend ifstream& operator>>(ifstream& in, rtANode& n)
    {
      n.ReadFromStream(in);
      return in;
    }

  protected:

  private:

};

inline bool rtANode::isLeaf()
{
  return false;
}

}

#endif // RTANODE_H
