/*
 * Action.h
 *
 *  Created on: Aug 19, 2012
 *      Author: sam
 */

#ifndef ACTION_H_
#define ACTION_H_

#include <vector>

#include "Vector.h"
#include "Projector.h"

class Action
{
  protected:
    int id;
    std::vector<double> values;
  public:
    Action(const int& id) :
        id(id)
    {
    }

    void add(const double& value)
    {
      values.push_back(value);
    }

    const double at(const int& i = 0 /*default to a single action*/) const
    {
      return values.at(i);
    }

    operator unsigned int() const
    {
      return id;
    }

    const bool operator==(const Action& that) const
    {
      return id == that.id;
    }

};

class ActionList
{
  protected:
    std::vector<Action*> actions;
  public:
    virtual ~ActionList()
    {
    }

    virtual const unsigned int dimension() const =0;
    virtual const Action& operator[](const int& index) const =0;
    virtual const Action& at(const int& index) const =0;
    virtual void add(const int& index, const double& value) =0;

    typedef std::vector<Action*>::iterator iterator;
    typedef std::vector<Action*>::const_iterator const_iterator;

    iterator begin()
    {
      return actions.begin();
    }

    const_iterator begin() const
    {
      return actions.begin();
    }

    iterator end()
    {
      return actions.end();
    }

    const_iterator end() const
    {
      return actions.end();
    }

};

class TabularActionList: public ActionList
{
  public:
    TabularActionList(const int& numActions)
    {
      for (int i = 0; i < numActions; i++)
        actions.push_back(new Action(i));
    }
    ~TabularActionList()
    {
      for (std::vector<Action*>::iterator iter = actions.begin();
          iter != actions.end(); ++iter)
        delete *iter;
      actions.clear();
    }

    const Action& operator[](const int& index) const
    {
      return *actions.at(index);
    }

    const Action& at(const int& index) const
    {
      return *actions.at(index);
    }

    void add(const int& index, const double& value)
    {
      actions.at(index)->add(value);
    }

    const unsigned int dimension() const
    {
      return actions.size();
    }
};

template<class T, class O>
class StateToStateAction
{
  public:
    virtual ~StateToStateAction()
    {
    }

    virtual const std::vector<SparseVector<T>*>& stateActions(
        const DenseVector<O>& x) =0;
    virtual const SparseVector<T>& stateAction(
        const std::vector<SparseVector<T>*>& xas,
        const Action& action) const =0;
};

// Tile coding base projector to state action
template<class T, class O>
class StateActionTilings: public StateToStateAction<T, O>
{
  protected:
    Projector<T, O>* projector;
    ActionList* actions;
    std::vector<SparseVector<T>*> xas;
  public:
    StateActionTilings(Projector<T, O>* projector, ActionList* actions) :
        projector(projector), actions(actions)
    {
      for (unsigned int i = 0; i < actions->dimension(); i++)
        xas.push_back(new SparseVector<T>(projector->dimension()));
    }
    ~StateActionTilings()
    {
      for (typename std::vector<SparseVector<T>*>::iterator iter = xas.begin();
          iter != xas.end(); ++iter)
        delete *iter;
      xas.clear();
    }

    const std::vector<SparseVector<T>*>& stateActions(const DenseVector<O>& x)
    {
      assert(actions->dimension() == xas.size());
      for (ActionList::const_iterator a = actions->begin(); a != actions->end();
          ++a)
        xas.at(**a)->set(projector->project(x, **a));
      return xas;
    }

    const SparseVector<T>& stateAction(const std::vector<SparseVector<T>*>& xas,
        const Action& action) const
    {
      return *xas.at(action);
    }
};

#endif /* ACTION_H_ */
