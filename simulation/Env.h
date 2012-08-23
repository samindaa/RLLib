/*
 * Env.h
 *
 *  Created on: Jun 29, 2012
 *      Author: sam
 */

#ifndef ENV_H_
#define ENV_H_

#include "../src/Vector.h"
#include "../src/Action.h"

/*
 * Represent an environment, a plant or an simulation.
 */
template<class O>
class Env
{
  protected:
    int numActions;
    DenseVector<O>* __vars;
    ActionList* actions;

  public:
    Env(int numVars, int numActions) :
        numActions(numActions), __vars(new DenseVector<O>(numVars)),
            actions(new TabularActionList(numActions))
    {
    }

    virtual ~Env()
    {
      delete __vars;
      delete actions;
    }

    virtual void initialize() =0;
    virtual void update() =0;
    virtual void step(const Action& action) =0;
    virtual bool endOfEpisode() const =0;
    virtual float r() const =0;
    virtual float z() const =0;

    ActionList& getActionList() const
    {
      return *actions;
    }

    int getNumActions() const
    {
      return numActions;
    }

    const DenseVector<O>& getVars() const
    {
      return *__vars;
    }

};

template<typename T>
inline int sgn(T val)
{
  return (T(0) < val) - (val < T(0));
}

#endif /* ENV_H_ */
