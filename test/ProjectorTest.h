/*
 * ProjectorTest.h
 *
 *  Created on: May 15, 2013
 *      Author: sam
 */

#ifndef PROJECTORTEST_H_
#define PROJECTORTEST_H_

#include "HeaderTest.h"
#include "Projector.h"

RLLIB_TEST(ProjectorTest)

class ProjectorTest: public ProjectorTestBase
{
  public:
    ProjectorTest() {}

    virtual ~ProjectorTest() {}
    void run();

  private:
    void testProjector();
};

#endif /* PROJECTORTEST_H_ */
