/*
 * Test.cpp
 *
 * Created on: May 12, 2013
 *     Author: sam
 *
 * Runs the test suit.
 *
 */

#include "HeaderTest.h"

RLLibTestRegistry::~RLLibTestRegistry()
{
  registry.clear();
}

RLLibTestRegistry* RLLibTestRegistry::inst = 0;

RLLibTestRegistry* RLLibTestRegistry::instance()
{
  if (!inst)
    inst = new RLLibTestRegistry;
  return inst;
}

void RLLibTestRegistry::registerInstance(RLLibTestCase* testCase)
{
  RLLibTestRegistry::instance()->registry.insert(make_pair(testCase->getName(), testCase));
}

int main(int argc, char** argv)
{
  RLLibTestRegistry* registry = RLLibTestRegistry::instance();
  cout << "*** number of registered test cases = " << registry->dimension() << endl;
  set<string> activeTestSet;
  ifstream inf("test/test.cfg");
  if (inf.is_open())
  {
    string str;
    while (getline(inf, str))
    {
      if (str.length())
      {
        unsigned int found = str.find_first_of("#");
        if ((found != string::npos) && (found > str.size()))
          activeTestSet.insert(str);
      }
    }
  }
  else
  {
    cerr << "ERROR! test/test.cfg is not found!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "*** number of active test cases = " << activeTestSet.size() << endl;
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
  for (set<string>::const_iterator iter = activeTestSet.begin(); iter != activeTestSet.end();
      ++iter)
  {
    RLLibTestRegistry::iterator iter2 = registry->find(*iter);
    RLLibTestCase* testCase = iter2->second;
    cout << "*** starts " << testCase->getName() << endl;
    testCase->run();
    cout << "***   ends " << testCase->getName() << endl;
  }

  cout << "*** Tests completed! " << endl;
  return EXIT_SUCCESS;
}

