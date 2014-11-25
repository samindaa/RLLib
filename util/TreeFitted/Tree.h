/**************************************************************************
 *   File:                        regtree.h                                *
 *   Description:   Basic abstract class for regression trees              *
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
#ifndef REGTREE_H
#define REGTREE_H

#include "rtnode.h"
#include "Regressor.h"

//<< Sam
namespace PoliFitted
{

class Tree: public Regressor
{
  public:
    // Constructors/Destructors
    //

    /**
     * Empty Constructor
     */
    Tree(string type, unsigned int input_size = 1, unsigned int output_size = 1);

    /**
     * Empty Destructor
     */
    virtual ~Tree();

  protected:
    rtANode* root;  // the root of the tree
};

}

#endif
