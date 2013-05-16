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

RLLibTestRegistory::~RLLibTestRegistory()
{
  for (std::vector<RLLibTestCase*>::iterator iter = registory.begin(); iter != registory.end();
      ++iter)
    delete *iter;
  registory.clear();

}

RLLibTestRegistory* RLLibTestRegistory::inst = 0;

RLLibTestRegistory* RLLibTestRegistory::instance()
{
  if (!inst)
    inst = new RLLibTestRegistory;
  return inst;
}

void RLLibTestRegistory::registerInstance(RLLibTestCase* testCase)
{
  RLLibTestRegistory::instance()->registory.push_back(testCase);
}

int main(int argc, char** argv)
{
  RLLibTestRegistory* registory = RLLibTestRegistory::instance();
  for (RLLibTestRegistory::iterator iter = registory->begin(); iter != registory->end(); ++iter)
  {
    RLLibTestCase* testCase = *iter;
    cout << "*** starts " << testCase->getName() << endl;
    testCase->run();
    cout << "***   ends " << testCase->getName() << endl;
  }

  cout << "*** Tests completed! " << endl;
  return EXIT_SUCCESS;
}

