/*
 * FiniteStateGraphTest.cpp
 *
 *  Created on: Oct 25, 2013
 *      Author: sam
 */

#include "FiniteStateGraphTest.h"

void FiniteStateGraphTest::testSimpleProblemTrajectory()
{
  LineProblem sp;
  Assert::assertObjectEquals(FiniteStateGraph::StepData(0, sp.O, sp.X, sp.A, 0.0, sp.Move), sp.step());
  Assert::assertObjectEquals(FiniteStateGraph::StepData(1, sp.A, sp.Move, sp.B, 0.0, sp.Move), sp.step());
  Assert::assertObjectEquals(FiniteStateGraph::StepData(2, sp.B, sp.Move, sp.C, 0.0, sp.Move), sp.step());
  Assert::assertObjectEquals(FiniteStateGraph::StepData(3, sp.C, sp.Move, sp.O, 1.0, sp.X), sp.step());
  Assert::assertObjectEquals(FiniteStateGraph::StepData(4, sp.O, sp.X, sp.A, 0.0, sp.Move), sp.step());
  Assert::assertObjectEquals(FiniteStateGraph::StepData(5, sp.A, sp.Move, sp.B, 0.0, sp.Move), sp.step());
}

void FiniteStateGraphTest::testRandomWalkRightTrajectory()
{
  RandomWalk sp;
  sp.enableOnlyRightPolicy();
  Assert::assertObjectEquals(FiniteStateGraph::StepData(0, sp.O, sp.X, sp.C, 0.0, sp.Right), sp.step());
  Assert::assertObjectEquals(FiniteStateGraph::StepData(1, sp.C, sp.Right, sp.D, 0.0, sp.Right), sp.step());
  Assert::assertObjectEquals(FiniteStateGraph::StepData(2, sp.D, sp.Right, sp.E, 0.0, sp.Right), sp.step());
  Assert::assertObjectEquals(FiniteStateGraph::StepData(3, sp.E, sp.Right, sp.O, 1.0, sp.X), sp.step());
  Assert::assertObjectEquals(FiniteStateGraph::StepData(4, sp.O, sp.X, sp.C, 0.0, sp.Right), sp.step());
  Assert::assertObjectEquals(FiniteStateGraph::StepData(5, sp.C, sp.Right, sp.D, 0.0, sp.Right), sp.step());
}

void FiniteStateGraphTest::testRandomWalkLeftTrajectory()
{
  RandomWalk sp;
  sp.enableOnlyLeftPolicy();
  Assert::assertObjectEquals(FiniteStateGraph::StepData(0, sp.O, sp.X, sp.C, 0.0, sp.Left), sp.step());
  Assert::assertObjectEquals(FiniteStateGraph::StepData(1, sp.C, sp.Left, sp.B, 0.0, sp.Left), sp.step());
  Assert::assertObjectEquals(FiniteStateGraph::StepData(2, sp.B, sp.Left, sp.A, 0.0, sp.Left), sp.step());
  Assert::assertObjectEquals(FiniteStateGraph::StepData(3, sp.A, sp.Left, sp.O, 0.0, sp.X), sp.step());
  Assert::assertObjectEquals(FiniteStateGraph::StepData(4, sp.O, sp.X, sp.C, 0.0, sp.Left), sp.step());
  Assert::assertObjectEquals(FiniteStateGraph::StepData(5, sp.C, sp.Left, sp.B, 0.0, sp.Left), sp.step());
}

void FiniteStateGraphTest::run()
{
  testSimpleProblemTrajectory();
  testRandomWalkRightTrajectory();
  testRandomWalkLeftTrajectory();
}

RLLIB_TEST_MAKE(FiniteStateGraphTest)
