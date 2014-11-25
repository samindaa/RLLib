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
#ifndef RTINODE_H
#define RTINODE_H
#include <cstdlib>
#include <string>
#include <fstream>
#include <iostream>

#include "rtANode.h"
#include "rtLeaf.h"
#include "rtLeafSample.h"
#include "rtLeafLinearInterp.h"

using namespace std;

//<< Sam
namespace PoliFitted
{

/**************************************************************************
 *   rtINode is a template class that rapresents an internal node of a     *
 *   regression tree.                                                      *
 *   This class extends rtANode and contains methods to set/get the index  *
 *   used to split the tree, the split value and the pointers to the left  *
 *   and right childs of the node (binary trees).                          *
 *   The splitting value is of type T.                                     *
 ***************************************************************************/
class rtINode: public rtANode
{
  public:

    /**
     * Empty constructor
     */
    rtINode() :
        axis(-1), split(0), left( NULL), right( NULL)
    {
    }

    /**
     * Basic contructor
     * @param a the index of splitting
     * @param s the split value
     * @param l the pointer to left child
     * @param r the pointer to right child
     */
    rtINode(int a, float s, rtANode* l, rtANode* r) :
        axis(a), split(s), left(l), right(r)
    {
    }

    /**
     * Get axis, axis is the index of the split
     * @return the axis
     */
    virtual int getAxis()
    {
      return axis;
    }

    /**
     * Get Split
     * @return the split value
     */
    virtual float getSplit()
    {
      return split;
    }

    /**
     * Get Left Child
     * @return a pointer to the left chid node
     */
    virtual rtANode* getLeft()
    {
      return left;
    }

    /**
     * Get Right Child
     * @return a pointer to the right child node
     */
    virtual rtANode* getRight()
    {
      return right;
    }

    /**
     * Get Child
     * @return a pointer to the right child node
     */
    virtual rtANode* getChild(float* values)
    {
      if (values[axis] < split)
      {
        return left;
      }
      else
      {
        return right;
      }
    }

    /**
     * Set te axis
     * @param a the axis
     */
    void setAxis(int a)
    {
      axis = a;
    }

    /**
     * Set the split
     * @param s the split value
     */
    void setSplit(float s)
    {
      split = s;
    }

    /**
     * Set the left child
     * @param l a pointer to the left child node
     */
    void setLeft(rtANode* l)
    {
      left = l;
    }

    /**
     * Set the right child
     * @param r a pointer to the right child node
     */
    void setRight(rtANode* r)
    {
      right = r;
    }

    /**
     * This method is used to determine if the object is a leaf or an
     * internal node
     * @return true if it is a leaf, false otherwise
     */
    virtual bool isLeaf();

    /**
     * Empty decostructor
     */
    virtual ~rtINode()
    {
      if (left != NULL)
      {
        delete left;
      }
      if (right != NULL)
      {
        delete right;
      }
    }

    /**
     *
     */
    virtual void WriteOnStream(ofstream& out)
    {
      out << "N" << endl;
      out << axis << " " << split;
      out << endl;
      if (left)
      {
        out << *left;
      }
      else
      {
        out << "EMPTYLEAF" << endl;
      }
      if (right)
      {
        out << *right;
      }
      else
      {
        out << "EMPTYLEAF" << endl;
      }
    }

    /**
     *
     */
    virtual void ReadFromStream(ifstream& in)
    {
      string type;
      rtANode* children[2];
      in >> axis >> split;
      for (unsigned int i = 0; i < 2; i++)
      {
        in >> type;
        if ("L" == type)
        {
          children[i] = new rtLeaf();
        }
        else if ("LS" == type)
        {
          children[i] = new rtLeafSample();
        }
        else if ("LLI" == type)
        {
          children[i] = new rtLeafLinearInterp();
        }
        else if ("N" == type)
        {
          children[i] = new rtINode();
        }
        else
        {
          children[i] = 0;
        }

        if (children[i])
        {
          in >> *children[i];
        }
      }
      left = children[0];
      right = children[1];
    }

  protected:

  private:
    int axis;  // the axis of split
    float split;  // the value of split
    rtANode* left;  // pointer to right child
    rtANode* right;  // pointer to left child
};

inline bool rtINode::isLeaf()
{
  return false;
}

}
#endif // RTINODE_H
