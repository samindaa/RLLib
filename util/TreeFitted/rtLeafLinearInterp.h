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
#ifndef RTLEAFLINEARINTERP_H
#define RTLEAFLINEARINTERP_H
#include <cstdlib>
#include <string>
#include <fstream>
#include <iostream>
#include <cassert>

#include <Eigen/Dense>//<< Eigen 3 (Sam)

#include "rtLeaf.h"
#include "Dataset.h"

using namespace std;

//<< Sam
namespace PoliFitted
{
/**************************************************************************
 *   rtLeafLinearInterp is a template class that rapresents a leaf of a    *
 *   regression tree, where the samples are approximated bymeans of linear *
 *   interpolation.                                                        *
 *   This class extends rtANode and contains methods to set/get the value  *
 *   saved in the node, this value is of type T                            *
 ***************************************************************************/
class rtLeafLinearInterp: public rtLeaf
{
  public:

    /**
     * Empty Constructor
     */
    rtLeafLinearInterp();

    /**
     * Basic constructor
     * @param data the dataset used to fit the linear model
     */
    rtLeafLinearInterp(Dataset* data);
    /**
     *
     */
    virtual ~rtLeafLinearInterp();

    /**
     * Set the value
     * @param val the value
     */
    virtual float Fit(Dataset* data);

    /**
     * Get the value
     * @return the value
     */
    virtual float getValue(Tuple* input = 0);

    /**
     * This method is used to determine if the object is a leaf or an
     * internal node
     * @return true if it is a leaf, false otherwise
     */
    virtual bool isLeaf();

    /**
     *
     */
    virtual void WriteOnStream(ofstream& out);

    /**
     *
     */
    virtual void ReadFromStream(ifstream& in);

  protected:
    Eigen::VectorXd mCoeffs;

};

inline bool rtLeafLinearInterp::isLeaf()
{
  return true;
}

}

#endif // RTNODE_H
