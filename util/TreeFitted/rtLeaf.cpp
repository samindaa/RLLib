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
#include "rtLeaf.h"

//<< Sam
using namespace PoliFitted;

rtLeaf::rtLeaf()
{
  mValue = 0.0;
  mVariance = 0.0;
}

/**
 * Basic constructor
 * @param val the value to store in the node
 */
rtLeaf::rtLeaf(Dataset* data)
{
  Fit(data);
}

rtLeaf::~rtLeaf()
{
}

float rtLeaf::Fit(Dataset* data)
{
  mValue = data->Mean();
#ifdef LEAF_VARIANCE
  mVariance = data->Variance();
#endif
#ifdef SPLIT_ANALYSIS
  for (unsigned int i = 0; i < data->GetInputSize(); i++)
  {
    float min, max, tmp;
    //initialize min and max with the attribute value of the first observation
    min = max = data->at(0)->GetInput(i);
    for (unsigned int c = 1; c < data->size(); c++)
    {
      tmp = data->at(c)->GetInput(i);
      if (tmp < min) min = tmp;
      else if (tmp > max) max = tmp;
    }
    char cmdf[100];
    sprintf(cmdf,"+ (x>=%f && x<=%f ? %f : 0)",min,max,(max-min));
    mPlotCutsF[i].insert(10,cmdf);
    char cmdg[100];
    sprintf(cmdg,"+ (x>=%f && x<=%f ? %f : 0)",min,max,1.0);
    mPlotCutsG[i].insert(10,cmdg);
    char cmds[100];
    sprintf(cmds,"+ (x>=%f && x<=%f ? %d : 0)",min,max,data->size());
    mPlotCutsS[i].insert(10,cmds);
  }
#endif
  return 0.0; //data->Variance();
}

float rtLeaf::getValue(Tuple* input)
{
  return mValue;
}

void rtLeaf::WriteOnStream(ofstream& out)
{
  out << "L" << endl;
  out << mValue << endl;
#ifdef LEAF_VARIANCE
  out << mVariance << endl;
#endif
}

void rtLeaf::ReadFromStream(ifstream& in)
{
  in >> mValue;
#ifdef LEAF_VARIANCE
  in >> mVariance;
#endif
}

