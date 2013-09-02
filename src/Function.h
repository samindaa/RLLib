/*
 * Function.h
 *
 *  Created on: Sep 1, 2013
 *      Author: sam
 */

#ifndef FUNCTION_H_
#define FUNCTION_H_

namespace RLLib
{

template<class T>
class ParameterizedFunction
{
  public:
    virtual ~ParameterizedFunction()
    {
    }
    virtual void persist(const std::string& f) const =0;
    virtual void resurrect(const std::string& f) =0;
};

template<class T>
class LinearLearner: public ParameterizedFunction<T>
{
  public:
    virtual ~LinearLearner()
    {
    }
    virtual double initialize() =0;
    virtual void reset() =0;
};

}  // namespace RLLib

#endif /* FUNCTION_H_ */
