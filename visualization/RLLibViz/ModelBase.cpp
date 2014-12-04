/*
 * ModelBase.cpp
 *
 *  Created on: Oct 11, 2013
 *      Author: sam
 */
#include "ModelBase.h"

using namespace RLLibViz;

ModelBase::ModelBase() :
    valueFunction(new Matrix(100, 100))
{
}

ModelBase::~ModelBase()
{
  delete valueFunction;
}

void ModelBase::updateValueFunction(Window* window, const RLLib::Control<double>* control,
    const RLLib::Ranges<double>* ranges, const bool& isEndingOfEpisode, const int& index)
{
  // Value function
  if (isEndingOfEpisode)
  {
    RLLib::PVector<double> x_t(2);
    double maxValue = 0, minValue = 0;
    const Range<double>* positionRange = ranges->at(0);
    const Range<double>* velocityRange = ranges->at(1);

    for (int position = 0; position < valueFunction->rows(); position++)
    {
      for (int velocity = 0; velocity < valueFunction->cols(); velocity++)
      {
        x_t[0] = positionRange->toUnit(
            positionRange->length() * position / valueFunction->cols() + positionRange->min());
        x_t[1] = velocityRange->toUnit(
            velocityRange->length() * velocity / valueFunction->rows() + velocityRange->min());
        double v = control->computeValueFunction(&x_t);
        valueFunction->at(position, velocity) = v;
        if (v > maxValue)
          maxValue = v;
        if (v < minValue)
          minValue = v;
      }
    }
    emit signal_add(window->valueFunctionVector[index], valueFunction, minValue, maxValue);
    emit signal_draw(window->valueFunctionVector[index]);
  }
}
