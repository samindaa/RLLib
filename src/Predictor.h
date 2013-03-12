/*
 * Predictor.h
 *
 *  Created on: Aug 19, 2012
 *      Author: sam
 */

#ifndef PREDICTOR_H_
#define PREDICTOR_H_

#include "Vector.h"

namespace RLLib
{

template<class T>
class Predictor
{
  public:
    virtual ~Predictor()
    {
    }
    virtual int dimension() const=0;
    virtual double predict(const SparseVector<T>& x) const =0;
    virtual double initialize() =0;
    virtual void reset() =0;

    virtual void persist(const std::string& f) const =0;
    virtual void resurrect(const std::string& f) =0;
};

template<class T>
class OnPolicyTD: public Predictor<T>
{
  public:
    virtual ~OnPolicyTD()
    {
    }
    virtual double update(const SparseVector<T>& x_t,
        const SparseVector<T>& x_tp1, double r_tp1) =0;
};

template<class T>
class GVF: public Predictor<T>
{
  public:
    virtual ~GVF()
    {
    }
    virtual double update(const SparseVector<T>& x_t,
        const SparseVector<T>& x_tp1, const double& gamma_tp1,
        const double& lambda_tp1, const double& rho_t, const double& r_tp1,
        const double& z_tp1) =0;
};

} // namespace RLLib

#endif /* PREDICTOR_H_ */
