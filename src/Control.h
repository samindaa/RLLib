/*
 * Control.h
 *
 *  Created on: Aug 19, 2012
 *      Author: sam
 */

#ifndef CONTROL_H_
#define CONTROL_H_

#include "Action.h"
#include "Vector.h"
#include "Policy.h"

namespace RLLib
{

template<class T, class O>
class Control
{
  public:
    virtual ~Control()
    {
    }
    virtual void reset() =0;
    virtual const Action& initialize(const DenseVector<O>& x_0) =0;
    virtual const Action& proposeAction(const DenseVector<O>& x) =0;
    virtual const Action& step(const DenseVector<O>& x_t, const Action& a_t,
        const DenseVector<O>& x_tp1, const double& r_tp1,
        const double& z_tp1) =0;
    virtual const double computeValueFunction(const DenseVector<O>& x) const =0;

    virtual void persist(const std::string& f) const =0;
    virtual void resurrect(const std::string& f) =0;
};

template<class T, class O>
class OnPolicyControlLearner: public Control<T, O>
{
  public:
    virtual ~OnPolicyControlLearner()
    {
    }
};

template<class T, class O>
class OffPolicyControlLearner: public Control<T, O>
{
  public:
    virtual ~OffPolicyControlLearner()
    {
    }
};

template<class T, class O>
class ActorOffPolicy
{
  public:
    virtual ~ActorOffPolicy()
    {
    }
    virtual void initialize() =0;
    virtual void reset() =0;
    virtual void update(const Representations<T>& xas_t, const Action& a_t,
        double const& rho_t, double const& gamma_t, double delta_t) =0;
    virtual void updatePolicy(const Representations<T>& xas) =0;
    virtual const Action& proposeAction(const Representations<T>& xas) =0;
    virtual double pi(const Action& a) const =0;

    virtual void persist(const std::string& f) const =0;
    virtual void resurrect(const std::string& f) =0;

};

template<class T, class O>
class ActorOnPolicy
{
  public:
    virtual ~ActorOnPolicy()
    {
    }

    virtual void initialize() =0;
    virtual void reset() =0;
    virtual void update(const Representations<T>& xas_t, const Action& a_t,
        double delta_t) =0;
    virtual const Action& proposeAction(const Representations<T>& xas) =0;
    virtual const Action& decide(const Representations<T>& xas) =0;
};

} // namespace RLLib

#endif /* CONTROL_H_ */
