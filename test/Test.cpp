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
 * Test.cpp
 *
 * Created on: May 12, 2013
 *     Author: sam
 *
 * Runs the test suit.
 *
 */

#include "HeaderTest.h"

RLLibTestRegistry::RLLibTestRegistry()
{
}

RLLibTestRegistry::~RLLibTestRegistry()
{
  registry.clear();
}

RLLibTestRegistry* RLLibTestRegistry::instance = 0;

RLLibTestRegistry* RLLibTestRegistry::getInstance()
{
  if (!instance)
    instance = new RLLibTestRegistry;
  return instance;
}

void RLLibTestRegistry::deleteInstance()
{
  assert(instance);
  delete instance;
}

void RLLibTestRegistry::registerInstance(RLLibTestCase* testCase)
{
  RLLibTestRegistry::getInstance()->registry.insert(make_pair(testCase->getName(), testCase));
}

int main(int argc, char** argv)
{
  RLLibTestRegistry* registry = RLLibTestRegistry::getInstance();
  cout << "*** number of registered test cases = " << registry->dimension() << endl;
  set<string> activeTestSet, onlyTestSet;
  ifstream inf("test/test.cfg");
  if (inf.is_open())
  {
    string str;
    while (getline(inf, str))
    {
      if (str.length() && (str.at(0) == '@'))
      {
        string tstr = str.substr(1);
        onlyTestSet.insert(tstr);
        activeTestSet.insert(tstr);
      }
      else if (str.length() && (str.at(0) != '#'))
        activeTestSet.insert(str);
      else
      {/*Not defined yet*/
      }
    }
  }
  else
  {
    cerr << "ERROR! test/test.cfg is not found!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "*** Number of active test cases = " << activeTestSet.size() << endl;
  cout << "*** Number of only   test cases = " << onlyTestSet.size() << endl;
  for (set<string>::const_iterator iter = activeTestSet.begin(); iter != activeTestSet.end();
      ++iter)
  {
    RLLibTestRegistry::iterator iter2 = registry->find(*iter);
    if (iter2 == registry->end())
    {
      cerr << "ERROR! Framework cannot find a registered test = " << *iter << endl;
      exit(EXIT_FAILURE);
    }
  }
  set<string>* running = !onlyTestSet.empty() ? &onlyTestSet : &activeTestSet;
  for (set<string>::const_iterator iter = running->begin(); iter != running->end();
      ++iter)
  {
    RLLibTestRegistry::iterator iter2 = registry->find(*iter);
    RLLibTestCase* testCase = iter2->second;
    cout << "*** starts " << testCase->getName() << endl;
    testCase->run();
    cout << "*** ends   " << testCase->getName() << endl;
  }

  RLLibTestRegistry::deleteInstance();
  cout << "*** Tests completed! " << endl;
  return EXIT_SUCCESS;
}

