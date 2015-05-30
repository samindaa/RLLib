/*
 * Copyright 2014 Saminda Abeyruwan (saminda@cs.miami.edu)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Spline.h
 *
 *  Created on: Oct 19, 2012
 *      Author: sam
 */

#ifndef SPLINE_H_
#define SPLINE_H_

#include <vector>
#include <stdexcept>

#include "util/Eigen/Dense"

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace RLLib
{
  class Spline: public std::vector<std::pair<double, double> >
  {
    public:
      //The boundary conditions available
      enum BCType
      {
        FIXED_1ST_DERIV_BC, FIXED_2ND_DERIV_BC, PARABOLIC_RUNOUT_BC
      };

      enum SplineType
      {
        LINEAR, CUBIC
      };

      //Constructor takes the boundary conditions as arguments, this
      //sets the first derivative (gradient) at the lower and upper
      //end points
      Spline() :
          _valid(false), bcLow(FIXED_2ND_DERIV_BC), bcHigh(FIXED_2ND_DERIV_BC), bcLowVal(0), //
          bcHighVal(0), _type(CUBIC)
      {
      }

      typedef std::vector<std::pair<double, double> > base;
      typedef base::const_iterator const_iterator;

      //Standard STL read-only container stuff
      const_iterator begin() const
      {
        return base::begin();
      }
      const_iterator end() const
      {
        return base::end();
      }
      void clear()
      {
        _valid = false;
        base::clear();
        _data.clear();
      }
      size_t size() const
      {
        return base::size();
      }
      size_t max_size() const
      {
        return base::max_size();
      }
      size_t capacity() const
      {
        return base::capacity();
      }
      bool empty() const
      {
        return base::empty();
      }

      //Add a point to the spline, and invalidate it so its
      //recalculated on the next access
      inline void addPoint(double x, double y)
      {
        _valid = false;
        base::push_back(std::pair<double, double>(x, y));
      }

      //Reset the boundary conditions
      inline void setLowBC(BCType BC, double val = 0)
      {
        bcLow = BC;
        bcLowVal = val;
        _valid = false;
      }

      inline void setHighBC(BCType BC, double val = 0)
      {
        bcHigh = BC;
        bcHighVal = val;
        _valid = false;
      }

      void setType(SplineType type)
      {
        _type = type;
        _valid = false;
      }

      //Check if the spline has been calculated, then generate the
      //spline interpolated value
      double operator()(double xval)
      {
        if (!_valid)
          generate();

        //Special cases when we're outside the range of the spline points
        if (xval <= x(0))
          return lowCalc(xval);
        if (xval >= x(size() - 1))
          return highCalc(xval);

        //Check all intervals except the last one
        for (std::vector<SplineData>::const_iterator iPtr = _data.begin(); iPtr != _data.end() - 1;
            ++iPtr)
          if ((xval >= iPtr->x) && (xval <= (iPtr + 1)->x))
            return splineCalc(iPtr, xval);

        return splineCalc(_data.end() - 1, xval);
      }

    private:

      ///////PRIVATE DATA MEMBERS
      struct SplineData
      {
          double x, a, b, c, d;
      };
      //vector of calculated spline data
      std::vector<SplineData> _data;

      //Second derivative at each point
      VectorXd _ddy;
      //Tracks whether the spline parameters have been calculated for
      //the current set of points
      bool _valid;
      //The boundary conditions
      BCType bcLow, bcHigh;
      //The values of the boundary conditions
      double bcLowVal, bcHighVal;

      SplineType _type;

      ///////PRIVATE FUNCTIONS
      //Function to calculate the value of a given spline at a point xval
      inline double splineCalc(std::vector<SplineData>::const_iterator i, double xval)
      {
        const double lx = xval - i->x;
        return ((i->a * lx + i->b) * lx + i->c) * lx + i->d;
      }

      inline double lowCalc(double xval)
      {
        const double lx = xval - x(0);

        if (_type == LINEAR)
          return lx * bcHighVal + y(0);

        const double firstDeriv = (y(1) - y(0)) / h(0)
            - 2 * h(0) * (_data[0].b + 2 * _data[1].b) / 6;

        switch (bcLow)
        {
          case FIXED_1ST_DERIV_BC:
            return lx * bcLowVal + y(0);
          case FIXED_2ND_DERIV_BC:
            return lx * lx * bcLowVal + firstDeriv * lx + y(0);
          case PARABOLIC_RUNOUT_BC:
            return lx * lx * _ddy[0] + lx * firstDeriv + y(0);
        }
        throw std::runtime_error("Unknown BC");
      }

      inline double highCalc(double xval)
      {
        const double lx = xval - x(size() - 1);

        if (_type == LINEAR)
          return lx * bcHighVal + y(size() - 1);

        const double firstDeriv = 2 * h(size() - 2) * (_ddy[size() - 2] + 2 * _ddy[size() - 1]) / 6
            + (y(size() - 1) - y(size() - 2)) / h(size() - 2);

        switch (bcHigh)
        {
          case FIXED_1ST_DERIV_BC:
            return lx * bcHighVal + y(size() - 1);
          case FIXED_2ND_DERIV_BC:
            return lx * lx * bcHighVal + firstDeriv * lx + y(size() - 1);
          case PARABOLIC_RUNOUT_BC:
            return lx * lx * _ddy[size() - 1] + lx * firstDeriv + y(size() - 1);
        }
        throw std::runtime_error("Unknown BC");
      }

      //These just provide access to the point data in a clean way
      inline double x(size_t i) const
      {
        return operator[](i).first;
      }
      inline double y(size_t i) const
      {
        return operator[](i).second;
      }
      inline double h(size_t i) const
      {
        return x(i + 1) - x(i);
      }

      //This function will recalculate the spline parameters and store
      //them in _data, ready for spline interpolation
      void generate()
      {
        if (size() < 2)
          throw std::runtime_error("Spline requires at least 2 points");

        //If any spline points are at the same x location, we have to
        //just slightly seperate them
        {
          bool testPassed(false);
          while (!testPassed)
          {
            testPassed = true;
            std::sort(base::begin(), base::end());

            for (base::iterator iPtr = base::begin(); iPtr != base::end() - 1; ++iPtr)
              if (iPtr->first == (iPtr + 1)->first)
              {
                if ((iPtr + 1)->first != 0)
                  (iPtr + 1)->first += (iPtr + 1)->first * std::numeric_limits<double>::epsilon()
                      * 10;
                else
                  (iPtr + 1)->first = std::numeric_limits<double>::epsilon() * 10;
                testPassed = false;
                break;
              }
          }
        }

        const size_t e = size() - 1;

        switch (_type)
        {
          case LINEAR:
          {
            _data.resize(e);
            for (size_t i(0); i < e; ++i)
            {
              _data[i].x = x(i);
              _data[i].a = 0;
              _data[i].b = 0;
              _data[i].c = (y(i + 1) - y(i)) / (x(i + 1) - x(i));
              _data[i].d = y(i);
            }
            break;
          }
          case CUBIC:
          {

            MatrixXd A(size(), size());
            for (size_t yv(0); yv <= e; ++yv)
              for (size_t xv(0); xv <= e; ++xv)
                A(xv, yv) = 0;

            for (size_t i(1); i < e; ++i)
            {
              A(i - 1, i) = h(i - 1);
              A(i, i) = 2 * (h(i - 1) + h(i));
              A(i + 1, i) = h(i);
            }

            VectorXd C(size());
            for (size_t xv(0); xv <= e; ++xv)
              C(xv) = 0;

            for (size_t i(1); i < e; ++i)
              C(i) = 6 * ((y(i + 1) - y(i)) / h(i) - (y(i) - y(i - 1)) / h(i - 1));

            //Boundary conditions
            switch (bcLow)
            {
              case FIXED_1ST_DERIV_BC:
                C(0) = 6 * ((y(1) - y(0)) / h(0) - bcLowVal);
                A(0, 0) = 2 * h(0);
                A(1, 0) = h(0);
                break;
              case FIXED_2ND_DERIV_BC:
                C(0) = bcLowVal;
                A(0, 0) = 1;
                break;
              case PARABOLIC_RUNOUT_BC:
                C(0) = 0;
                A(0, 0) = 1;
                A(1, 0) = -1;
                break;
            }

            switch (bcHigh)
            {
              case FIXED_1ST_DERIV_BC:
                C(e) = 6 * (bcHighVal - (y(e) - y(e - 1)) / h(e - 1));
                A(e, e) = 2 * h(e - 1);
                A(e - 1, e) = h(e - 1);
                break;
              case FIXED_2ND_DERIV_BC:
                C(e) = bcHighVal;
                A(e, e) = 1;
                break;
              case PARABOLIC_RUNOUT_BC:
                C(e) = 0;
                A(e, e) = 1;
                A(e - 1, e) = -1;
                break;
            }

            MatrixXd AInv = A.inverse();
            _ddy = AInv * C;

            _data.resize(size() - 1);
            for (size_t i(0); i < e; ++i)
            {
              _data[i].x = x(i);
              _data[i].a = (_ddy(i + 1) - _ddy(i)) / (6 * h(i));
              _data[i].b = _ddy(i) / 2;
              _data[i].c = (y(i + 1) - y(i)) / h(i) - _ddy(i + 1) * h(i) / 6 - _ddy(i) * h(i) / 3;
              _data[i].d = y(i);
            }
            break;
          }
        }
        _valid = true;
      }
  };

} // namespace RLLib

#endif /* SPLINE_H_ */
