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
#include "rtLeafLinearInterp.h"

using namespace PoliFitted;

rtLeafLinearInterp::rtLeafLinearInterp()
{
}

/**
 * Basic constructor
 * @param val the value to store in the node
 */
rtLeafLinearInterp::rtLeafLinearInterp(Dataset* data)
{
  Fit(data);
}

rtLeafLinearInterp::~rtLeafLinearInterp()
{
}

float rtLeafLinearInterp::Fit(Dataset* data)
{
  unsigned int size = data->size();
  unsigned int input_size = data->GetInputSize() + 1;
  mCoeffs.setZero(input_size);
  // In case the samples are less than the unknowns
  if (size < input_size)
  {
    float result = 0.0;
    for (unsigned int i = 0; i < data->size(); i++)
    {
      result += data->at(i)->GetOutput();
    }
    result /= (float) data->size();
    mCoeffs(0) = result;
    return 0.0;
  }
  else
  {
    Eigen::MatrixXd X(size, input_size);
    Eigen::VectorXd y(size);

    for (unsigned int i = 0; i < size; i++)
    {
      y(i) = data->at(i)->GetOutput();
      X(i, 0) = 1.0f;
      for (unsigned int j = 1; j < input_size; j++)
        X(i, j) = data->at(i)->GetInput(j - 1);
    }

    mCoeffs = X.colPivHouseholderQr().solve(y); // Fit
    // The sum of squares of the residuals from the best-fit, \chi^2, is returned in chisq
    return (X * mCoeffs - y).array().square().sum();
  }
}

float rtLeafLinearInterp::getValue(Tuple* input)
{
  float result = mCoeffs(0);
  for (unsigned int i = 1; i < mCoeffs.rows(); i++)
  {
    result += mCoeffs(i) * (*input)[i - 1];
  }
//  if (result < 0) result = 0;
  return result;
}

void rtLeafLinearInterp::WriteOnStream(ofstream& out)
{
  out << "LLI" << endl;
  out << mCoeffs.rows() << endl;
  for (unsigned int i = 0; i < mCoeffs.rows(); i++)
  {
    out << mCoeffs(i) << " ";
  }
}

void rtLeafLinearInterp::ReadFromStream(ifstream& in)
{
  int size;
  double value;
  in >> size;
  assert(size > 0);
  mCoeffs.setZero(size);
  for (int i = 0; i < size; i++)
  {
    in >> value;
    mCoeffs(i) = value;
  }
}

