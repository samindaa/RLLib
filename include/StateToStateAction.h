/*
 * Copyright 2015 Saminda Abeyruwan (saminda@cs.miami.edu)
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

#include "Projector.h"

namespace RLLib
{

  template<typename T>
  class Representations
  {
    protected:
      std::vector<Vector<T>*> phis;

    public:
      Representations(const int& numFeatures, const Actions<T>* actions)
      {
        for (typename Actions<T>::const_iterator iter = actions->begin(); iter != actions->end();
            ++iter)
          phis.push_back(new SVector<T>(numFeatures)); // fixme: generalize
      }

      ~Representations()
      {
        for (typename std::vector<Vector<T>*>::iterator iter = phis.begin(); iter != phis.end();
            ++iter)
          delete *iter;
        phis.clear();
      }

      int dimension() const
      {
        return phis.size();
      }

      int vectorSize() const
      {
        ASSERT(dimension() > 0);
        return phis[0]->dimension();
      }

      void set(const Vector<T>* phi, const Action<T>* action)
      {
        ASSERT(action->id() < static_cast<int>(phis.size()));
        phis[action->id()]->set(phi);
      }

      Vector<T>* at(const Action<T>* action)
      {
        ASSERT(action->id() < static_cast<int>(phis.size()));
        return phis[action->id()];
      }

      const Vector<T>* at(const Action<T>* action) const
      {
        ASSERT(action->id() < static_cast<int>(phis.size()));
        return phis[action->id()];
      }

      void clear()
      {
        for (typename std::vector<Vector<T>*>::iterator iter = phis.begin(); iter != phis.end();
            ++iter)
          (*iter)->clear();
      }
  };

  template<typename T>
  class StateToStateAction
  {
    public:
      virtual ~StateToStateAction()
      {
      }
      virtual const Vector<T>* stateAction(const Vector<T>* x, const Action<T>* a) =0;
      virtual const Representations<T>* stateActions(const Vector<T>* x) =0;
      virtual const Actions<T>* getActions() const =0;
      virtual T vectorNorm() const =0;
      virtual int dimension() const =0;
  };

// Tile coding base projector to state action
  template<typename T>
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
        ASSERT(actions->dimension() == phis->dimension());
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

      T vectorNorm() const
      {
        return projector->vectorNorm();
      }

      int dimension() const
      {
        return projector->dimension();
      }
  };

  template<typename T>
  class TabularAction: public StateToStateAction<T>
  {
    protected:
      Projector<T>* projector;
      Actions<T>* actions;
      Representations<T>* phis;
      Vector<T>* phi;
      bool includeActiveFeature;
    public:
      TabularAction(Projector<T>* projector, Actions<T>* actions, bool includeActiveFeature = true) :
          projector(projector), actions(actions), //
          phis(
              new Representations<T>(
                  includeActiveFeature ?
                      actions->dimension() * projector->dimension() + 1 :
                      actions->dimension() * projector->dimension(), actions)), //
          phi(new SVector<T>(phis->vectorSize())), includeActiveFeature(includeActiveFeature)
      {
      }

      ~TabularAction()
      {
        delete phis;
        delete phi;
      }

      const Vector<T>* stateAction(const Vector<T>* x, const Action<T>* a)
      {
        phi->clear();
        if (x->empty())
          return phi;
        phi->set(projector->project(x), projector->dimension() * a->id());
        if (includeActiveFeature)
          phi->setEntry(phi->dimension() - 1, 1.0);
        return phi;
      }

      const Representations<T>* stateActions(const Vector<T>* x)
      {
        ASSERT(actions->dimension() == phis->dimension());
        for (typename Actions<T>::const_iterator a = actions->begin(); a != actions->end(); ++a)
          phis->set(stateAction(x, *a), *a);
        return phis;
      }

      const Actions<T>* getActions() const
      {
        return actions;
      }

      T vectorNorm() const
      {
        return includeActiveFeature ? projector->vectorNorm() + 1 : projector->vectorNorm();
      }

      int dimension() const
      {
        return phis->vectorSize();
      }
  };

} // namespace RLLib

#endif /* STATETOSTATEACTION_H_ */
