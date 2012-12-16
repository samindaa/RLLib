/*
 * Representation.h
 *
 *  Created on: Aug 31, 2012
 *      Author: sam
 */

#ifndef REPRESENTATION_H_
#define REPRESENTATION_H_

#include <vector>

#include "Vector.h"
#include "Projector.h"

namespace RLLib
{

template<class T>
class Representations
{
  protected:
    std::vector<SparseVector<T>*>* phis;
  public:
    Representations(const int& numFeatures, const int& numActions) :
        phis(new std::vector<SparseVector<T>*>())
    {
      for (int i = 0; i < numActions; i++)
        phis->push_back(new SparseVector<T>(numFeatures));
    }
    ~Representations()
    {
      for (typename std::vector<SparseVector<T>*>::iterator iter =
          phis->begin(); iter != phis->end(); ++iter)
        delete *iter;
      phis->clear();
      delete phis;
    }

    const unsigned int dimension() const
    {
      return phis->size();
    }

    void set(const SparseVector<T>& phi, const Action& action)
    {
      phis->at(action)->set(phi);
    }

    const SparseVector<T>& at(const Action& action) const
    {
      return *phis->at(action);
    }

    typedef typename std::vector<SparseVector<T>*>::iterator iterator;
    typedef typename std::vector<SparseVector<T>*>::const_iterator const_iterator;

    iterator begin()
    {
      return phis->begin();
    }

    iterator end()
    {
      return phis->end();
    }

    const_iterator begin() const
    {
      return phis->begin();
    }

    const_iterator end() const
    {
      return phis->end();
    }
};

template<class T, class O>
class StateToStateAction
{
  public:
    virtual ~StateToStateAction()
    {
    }
    virtual const SparseVector<T>& stateAction(const DenseVector<O>& x) =0;
    virtual const Representations<T>& stateActions(const DenseVector<O>& x) =0;
    virtual const Projector<T, O>& getProjector() const =0;
    virtual const ActionList& getActionList() const =0;
};

// Tile coding base projector to state action
template<class T, class O>
class StateActionTilings: public StateToStateAction<T, O>
{
  protected:
    Projector<T, O>* projector;
    ActionList* actions;
    Representations<T>* phis;
  public:
    StateActionTilings(Projector<T, O>* projector, ActionList* actions) :
        projector(projector), actions(actions), phis(
            new Representations<T>(projector->dimension(),
                actions->dimension()))
    {
    }

    ~StateActionTilings()
    {
      delete phis;
    }

    const SparseVector<T>& stateAction(const DenseVector<O>& x)
    {
      assert(actions->dimension() == phis->dimension());
      return projector->project(x);
    }

    const Representations<T>& stateActions(const DenseVector<O>& x)
    {
      assert(actions->dimension() == phis->dimension());
      for (ActionList::const_iterator a = actions->begin(); a != actions->end();
          ++a)
      {
        if (actions->dimension() == 1)
          phis->set(projector->project(x), **a); // projection from whole space
        else
          phis->set(projector->project(x, **a), **a);
      }
      return *phis;
    }

    const Projector<T, O>& getProjector() const
    {
      return *projector;
    }

    const ActionList& getActionList() const
    {
      return *actions;
    }
};

} // namespace RLLib

#endif /* REPRESENTATION_H_ */
