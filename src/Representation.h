/*
 * Copyright 2013 Saminda Abeyruwan (saminda@cs.miami.edu)
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
 * Representation.h
 *
 *  Created on: Aug 31, 2012
 *      Author: sam
 */

#ifndef REPRESENTATION_H_
#define REPRESENTATION_H_

#include <map>
#include "Action.h"
#include "Vector.h"
#include "Projector.h"

namespace RLLib
{

template<class T>
class Representations
{
  public:
    typedef typename std::map<int, SparseVector<T>*>::iterator iterator;
    typedef typename std::map<int, SparseVector<T>*>::const_iterator const_iterator;

  protected:
    std::map<int, SparseVector<T>*>* phis;
  public:
    Representations(const int& numFeatures, const ActionList* actions) :
        phis(new std::map<int, SparseVector<T>*>())
    {
      for (ActionList::const_iterator iter = actions->begin(); iter != actions->end(); ++iter)
        phis->insert(std::make_pair((*iter)->id(), new SparseVector<T>(numFeatures)));
    }
    ~Representations()
    {
      for (typename std::map<int, SparseVector<T>*>::iterator iter = phis->begin();
          iter != phis->end(); ++iter)
        delete iter->second;
      phis->clear();
      delete phis;
    }

    const int dimension() const
    {
      return phis->size();
    }

    void set(const SparseVector<T>& phi, const Action& action)
    {
      phis->at(action.id())->set(phi);
    }

    const SparseVector<T>& at(const Action& action) const
    {
      return *phis->at(action.id());
    }

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

    void clear()
    {
      for (typename std::map<int, SparseVector<T>*>::iterator iter = phis->begin();
          iter != phis->end(); ++iter)
        iter->second->clear();
    }
};

template<class T, class O>
class StateToStateAction
{
  public:
    virtual ~StateToStateAction()
    {
    }
    virtual const Representations<T>& stateActions(const DenseVector<O>& x) =0;
    virtual const ActionList& getActionList() const =0;
    virtual double vectorNorm() const =0;
    virtual int dimension() const =0;
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
            new Representations<T>(projector->dimension(), actions))
    {
    }

    ~StateActionTilings()
    {
      delete phis;
    }

    const Representations<T>& stateActions(const DenseVector<O>& x)
    {
      assert(actions->dimension() == phis->dimension());
      if (x.empty())
      {
        phis->clear();
        return *phis;
      }
      for (ActionList::const_iterator a = actions->begin(); a != actions->end(); ++a)
      {
        if (actions->dimension() == 1)
          phis->set(projector->project(x), **a); // projection from whole space
        else
          phis->set(projector->project(x, (*a)->id()), **a);
      }
      return *phis;
    }

    const ActionList& getActionList() const
    {
      return *actions;
    }

    double vectorNorm() const
    {
      return projector->vectorNorm();
    }

    int dimension() const
    {
      return projector->dimension();
    }
};

template<class T, class O>
class TabularAction: public StateToStateAction<T, O>
{
  protected:
    Projector<T, O>* projector;
    ActionList* actions;
    Representations<T>* phis;
    SparseVector<T>* _phi;
    bool includeActiveFeature;
  public:
    TabularAction(Projector<T, O>* projector, ActionList* actions, bool includeActiveFeature = true) :
        projector(projector), actions(actions), phis(
            new Representations<T>(
                includeActiveFeature ?
                    actions->dimension() * projector->dimension() + 1 :
                    actions->dimension() * projector->dimension(), actions)), _phi(
            new SparseVector<T>(
                includeActiveFeature ?
                    actions->dimension() * projector->dimension() + 1 :
                    actions->dimension() * projector->dimension())), includeActiveFeature(
            includeActiveFeature)
    {
    }

    ~TabularAction()
    {
      delete phis;
      delete _phi;
    }

    const Representations<T>& stateActions(const DenseVector<O>& x)
    {
      assert(actions->dimension() == phis->dimension());
      const SparseVector<T>& phi = projector->project(x);
      for (ActionList::const_iterator a = actions->begin(); a != actions->end(); ++a)
      {
        _phi->set(phi, projector->dimension() * (*a)->id());
        if (includeActiveFeature)
          _phi->insertLast(1.0);
        phis->set(*_phi, **a);
      }
      return *phis;
    }

    const ActionList& getActionList() const
    {
      return *actions;
    }

    double vectorNorm() const
    {
      return includeActiveFeature ? projector->vectorNorm() + 1 : projector->vectorNorm();
    }

    int dimension() const
    {
      return
          includeActiveFeature ?
              actions->dimension() * projector->dimension() + 1 :
              actions->dimension() * projector->dimension();
    }
};

} // namespace RLLib

#endif /* REPRESENTATION_H_ */
