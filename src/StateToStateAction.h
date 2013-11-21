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
 * StateToStateAction.h
 *
 *  Created on: Nov 4, 2013
 *      Author: sam
 */

#ifndef STATETOSTATEACTION_H_
#define STATETOSTATEACTION_H_

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
    typedef typename std::map<int, Vector<T>*>::iterator iterator;
    typedef typename std::map<int, Vector<T>*>::const_iterator const_iterator;

  protected:
    std::map<int, Vector<T>*>* phis;
  public:
    Representations(const int& numFeatures, const Actions<T>* actions) :
        phis(new std::map<int, Vector<T>*>())
    {
      for (typename Actions<T>::const_iterator iter = actions->begin(); iter != actions->end();
          ++iter)
        phis->insert(std::make_pair((*iter)->id(), new SVector<T>(numFeatures)));
    }
    ~Representations()
    {
      for (typename std::map<int, Vector<T>*>::iterator iter = phis->begin(); iter != phis->end();
          ++iter)
        delete iter->second;
      phis->clear();
      delete phis;
    }

    const int dimension() const
    {
      return phis->size();
    }

    void set(const Vector<T>* phi, const Action<T>* action)
    {
      phis->at(action->id())->set(phi);
    }

    Vector<T>* at(const Action<T>* action)
    {
      return phis->at(action->id());
    }

    const Vector<T>* at(const Action<T>* action) const
    {
      return phis->at(action->id());
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
      for (typename std::map<int, Vector<T>*>::iterator iter = phis->begin(); iter != phis->end();
          ++iter)
        iter->second->clear();
    }
};

template<class T>
class StateToStateAction
{
  public:
    virtual ~StateToStateAction()
    {
    }
    virtual const Vector<T>* stateAction(const Vector<T>* x, const Action<T>* a) =0;
    virtual const Representations<T>* stateActions(const Vector<T>* x) =0;
    virtual const Actions<T>* getActions() const =0;
    virtual double vectorNorm() const =0;
    virtual int dimension() const =0;
};

// Tile coding base projector to state action
template<class T>
class StateActionTilings: public StateToStateAction<T>
{
  protected:
    Projector<T>* projector;
    Actions<T>* actions;
    Representations<T>* phis;
  public:
    StateActionTilings(Projector<T>* projector, Actions<T>* actions) :
        projector(projector), actions(actions), phis(
            new Representations<T>(projector->dimension(), actions))
    {
    }

    ~StateActionTilings()
    {
      delete phis;
    }

    const Vector<T>* stateAction(const Vector<T>* x, const Action<T>* a)
    {
      if (actions->dimension() == 1)
        return projector->project(x); // projection from whole space
      else
        return projector->project(x, a->id());
    }

    const Representations<T>* stateActions(const Vector<T>* x)
    {
      assert(actions->dimension() == phis->dimension());
      if (x->empty())
      {
        phis->clear();
        return phis;
      }
      for (typename Actions<T>::const_iterator a = actions->begin(); a != actions->end(); ++a)
        phis->set(stateAction(x, *a), *a);
      return phis;
    }

    const Actions<T>* getActions() const
    {
      return actions;
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

template<class T>
class TabularAction: public StateToStateAction<T>
{
  protected:
    Projector<T>* projector;
    Actions<T>* actions;
    Representations<T>* phis;
    Vector<T>* _phi;
    bool includeActiveFeature;
  public:
    TabularAction(Projector<T>* projector, Actions<T>* actions, bool includeActiveFeature = true) :
        projector(projector), actions(actions), phis(
            new Representations<T>(
                includeActiveFeature ?
                    actions->dimension() * projector->dimension() + 1 :
                    actions->dimension() * projector->dimension(), actions)), _phi(
            new SVector<T>(
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

    const Vector<T>* stateAction(const Vector<T>* x, const Action<T>* a)
    {
      _phi->set(projector->project(x), projector->dimension() * a->id());
      if (includeActiveFeature)
        _phi->setEntry(_phi->dimension() - 1, 1.0);
      return _phi;
    }

    const Representations<T>* stateActions(const Vector<T>* x)
    {
      assert(actions->dimension() == phis->dimension());
      for (typename Actions<T>::const_iterator a = actions->begin(); a != actions->end(); ++a)
        phis->set(stateAction(x, *a), *a);
      return phis;
    }

    const Actions<T>* getActions() const
    {
      return actions;
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

#endif /* STATETOSTATEACTION_H_ */
