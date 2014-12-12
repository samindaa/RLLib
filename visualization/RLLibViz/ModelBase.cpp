/*
 * ModelBase.cpp
 *
 *  Created on: Oct 11, 2013
 *      Author: sam
 */
#include "ModelBase.h"

using namespace RLLibViz;

ModelBase::ModelBase()
{
  valueFunction2D.resize(100, 100);
}

ModelBase::~ModelBase()
{
}

void ModelBase::updateValueFunction(Window* window, const RLLib::Control<double>* control,
    const TRStep<double>* output, const RLLib::Ranges<double>* ranges,
    const bool& isEndingOfEpisode, const int& index)
{
  // Value function
  if (isEndingOfEpisode && output->o_tp1->dimension() == 2 /*FixMe*/)
  {
    RLLib::PVector<double> stateVariable(2);
    const Range<double>* positionRange = ranges->at(0);
    const Range<double>* velocityRange = ranges->at(1);

    for (int position = 0; position < valueFunction2D.rows(); position++)
    {
      for (int velocity = 0; velocity < valueFunction2D.cols(); velocity++)
      {
        stateVariable[0] = positionRange->toUnit(
            positionRange->length() * position / valueFunction2D.cols() + positionRange->min());
        stateVariable[1] = velocityRange->toUnit(
            velocityRange->length() * velocity / valueFunction2D.rows() + velocityRange->min());
        double v = control->computeValueFunction(&stateVariable);
        valueFunction2D(position, velocity) = v;
      }
    }
    emit signal_add(window->valueFunctionVector[index], valueFunction2D);
    emit signal_draw(window->valueFunctionVector[index]);
  }

  // Nearest target position
  emit signal_add(window->valueFunctionVector[index],
      Vec(output->o_tp1->getEntry(0) * (valueFunction2D.rows() - 1),
          output->o_tp1->getEntry(1) * (valueFunction2D.cols() - 1), 0, 1),
      Vec(0.0, 0.0, 0.0, 1.0));
}
