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
  RLLibTestRegistry::instance()->registry.push_back(testCase);
}

int main(int argc, char** argv)
{
  RLLibTestRegistry* registry = RLLibTestRegistry::instance();
  cout << "*** nbRLLibTests=" << registry->dimension() << endl;
  for (RLLibTestRegistry::iterator iter = registry->begin(); iter != registry->end(); ++iter)
  {
    RLLibTestCase* testCase = *iter;
    cout << "*** starts " << testCase->getName() << endl;
    testCase->run();
    cout << "***   ends " << testCase->getName() << endl;
  }

  cout << "*** Tests completed! " << endl;
  return EXIT_SUCCESS;
}

