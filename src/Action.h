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
 * Action.h
 *
 *  Created on: Aug 19, 2012
 *      Author: sam
 */

#ifndef ACTION_H_
#define ACTION_H_

#include <vector>
#include <cassert>

namespace RLLib
{

class Action
{
  private:
    int actionID;
    std::vector<double> values;
  public:
    explicit Action(const int& actionID) :
        actionID(actionID)
    {
    }

    void push_back(const double& value)
    {
      values.push_back(value);
    }

    const double at(const int& i = 0 /*default to a single action*/) const
    {
      return values.at(i);
    }

    void update(const unsigned int& i, const double& value)
    {
      assert(values.size() != 0 && (i >= 0 && i < values.size()));
      values[i] = value;
    }

    const bool operator==(const Action& that) const
    {
      return actionID == that.actionID;
    }

    const bool operator!=(const Action& that) const
    {
      return actionID != that.actionID;
    }

    // Id within an associated id group
    const int id() const
    {
      return actionID;
    }

    const unsigned int dimension() const
    {
      return values.size();
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
    virtual void push_back(const int& index, const double& value) =0;
    virtual void update(const int& actionIndex, const unsigned int& vectorIndex,
        const double& value) =0;

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

class GeneralActionList: public ActionList
{
  public:
    GeneralActionList(const int& numActions)
    {
      for (int i = 0; i < numActions; i++)
        actions.push_back(new Action(i));
    }
    virtual ~GeneralActionList()
    {
      for (std::vector<Action*>::iterator iter = actions.begin(); iter != actions.end(); ++iter)
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

    void push_back(const int& index, const double& value)
    {
      actions.at(index)->push_back(value);
    }

    void update(const int& actionIndex, const unsigned int& vectorIndex, const double& value)
    {
      actions.at(actionIndex)->update(vectorIndex, value);
    }

    const unsigned int dimension() const
    {
      return actions.size();
    }
};

} // namespace RLLib

#endif /* ACTION_H_ */
