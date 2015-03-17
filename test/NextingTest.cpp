/*
 * NextingTest.cpp
 *
 *  Created on: Nov 25, 2013
 *      Author: sam
 */

#include "NextingTest.h"

NextingProblem::NextingProblem(Random<double>* random) :
    RLProblem<double>(random, //
        10/*IRDistance*/+ 4/*Light*/+ 8/*IRLight*/+ 4/*Thermal*/+ 1/*RotationalVelocity*/+ 3/*Mag*/
        + 3/*Accel*/+ 3/*MortorSpeed*/+ 3/*MotorVoltate*/+ 3/*MotorCurrent*/+ 3/*MotorTemperature*/
        + 1/*OverheatingFlag*/+ 3/*LastAction*/, 1, 1)
{
}

NextingProblem::~NextingProblem()
{
}

void NextingProblem::initialize()
{
  for (int i = 0; i < dimension(); i++)
    output->o_tp1->clear();
}

void NextingProblem::step(const Action<double>* action)
{
  /*do some work*/
}

void NextingProblem::updateTRStep()
{
  for (int i = 0; i < dimension(); i++)
    output->o_tp1->setEntry(i, random->nextReal());
}

bool NextingProblem::endOfEpisode() const
{
  return false;
}

double NextingProblem::r() const
{
  return 0;
}

double NextingProblem::z() const
{
  return 0;
}

// ==
NextingProjector::NextingProjector()
{
  nbTiles = 10 * (8 + 4 + 4 + 4) + // IRDistance
      4 * (8 + 1) +         // Light
      8 * (6 + 1 + 1 + 1) +  // IRLight
      (8 / 2/*author said half*/) * 4 +               // Thermal
      1 * 8 +               // RotationalVelocity
      3 * 8 +               // Mag
      3 * 8 +               // Accel
      3 * (4 + 8) +         // MortorSpeed
      3 * 2 +               // MotorVoltate
      3 * 2 +               // MotorCurrent
      3 * 4 +               // MotorTemperature
      1 * 4 +               // OverheatingFlag
      3 * 4;                // LastAction

  memory = 10 * (8 * 8 + 2 * 4 + 4 * 4 * 4 + 4 * 4 * 4) + // IRDistance
      4 * (4 * 8 + 4 * 4 * 1) +          // Light
      8 * (8 * 6 + 4 * 1 + 8 * 8 * 1 + 8 * 8 * 1) +  // IRLight
      (8 / 2) * 8 * 4 +                  // Thermal
      1 * 8 * 8 +                // Rotational velocity
      3 * 8 * 8 +                  // Mag
      3 * 8 * 8 +                  // Accel
      3 * (8 * 4 + 8 * 8 * 8) +          // MortorSpeed
      3 * 8 * 2 +                  // MotorVoltate
      3 * 8 * 2 +                   // MotorCurrent
      3 * 4 * 4 +                   // MotorTemperature
      1 * 2 * 4 +                // OverheatingFlag
      3 * 6 * 4;                   // LastAction

  std::cout << "nbTiles=" << nbTiles << std::endl;
  std::cout << "diff=" << (456 - nbTiles) << std::endl;
  std::cout << "memory=" << memory << std::endl;

  vector = new SVector<double>(memory + 1/*bias unit*/);
  random = new Random<double>;
  hashing = new MurmurHashing<double>(random, memory);
  tiles = new Tiles<double>(hashing);
}

NextingProjector::~NextingProjector()
{
  delete vector;
  delete random;
  delete hashing;
  delete tiles;
}

const Vector<double>* NextingProjector::project(const Vector<double>* x, const int& h1)
{
  vector->clear();
  if (x->empty())
    return vector;
  ASSERT(false);
  return vector;
}

const Vector<double>* NextingProjector::project(const Vector<double>* x)
{
  vector->clear();
  if (x->empty())
    return vector;

  int h2 = 0;
  int i, j, k;
  // IRdistance
  j = 0;
  for (i = 0; i < 10; i++)
  {
    tiles->tiles1(vector, 8, memory, x->getEntry(j + i) * 8, h2);
    ++h2;

    tiles->tiles1(vector, 4, memory, x->getEntry(j + i) * 2, h2);
    ++h2;

    k = j + (i + 1) % 10;
    tiles->tiles2(vector, 4, memory, x->getEntry(j + i) * 4, x->getEntry(k) * 4, h2);
    ++h2;

    k = j + (i + 2) % 10;
    tiles->tiles2(vector, 4, memory, x->getEntry(j + i) * 4, x->getEntry(k) * 4, h2);
    ++h2;
  }

  // Light
  j += i;
  for (i = 0; i < 4; i++)
  {
    tiles->tiles1(vector, 8, memory, x->getEntry(j + i) * 4, h2);
    ++h2;
    k = j + (i + 1) % 4;
    tiles->tiles2(vector, 1, memory, x->getEntry(j + i) * 4, x->getEntry(k) * 4, h2);
    ++h2;
  }

  // IRlight
  j += i;
  for (i = 0; i < 8; i++)
  {
    tiles->tiles1(vector, 6, memory, x->getEntry(j + i) * 8, h2);
    ++h2;

    tiles->tiles1(vector, 1, memory, x->getEntry(j + i) * 4, h2);
    ++h2;

    k = j + (i + 1) % 8;
    tiles->tiles2(vector, 1, memory, x->getEntry(j + i) * 8, x->getEntry(k) * 8, h2);
    ++h2;

    k = j + (i + 2) % 8;
    tiles->tiles2(vector, 1, memory, x->getEntry(j + i) * 8, x->getEntry(k) * 8, h2);
    ++h2;
  }

  // Thermal
  j += i;
  for (i = 0; i < 4; i++)
  {
    tiles->tiles1(vector, 4, memory, x->getEntry(j + i) * 8, h2);
    ++h2;
  }

  // RotationalVelocity
  j += i;
  for (i = 0; i < 1; i++)
  {
    tiles->tiles1(vector, 8, memory, x->getEntry(j + i) * 8, h2);
    ++h2;
  }

  // Magnetic
  j += i;
  for (i = 0; i < 3; i++)
  {
    tiles->tiles1(vector, 8, memory, x->getEntry(j + i) * 8, h2);
    ++h2;
  }

  // Acceleration
  j += i;
  for (i = 0; i < 3; i++)
  {
    tiles->tiles1(vector, 8, memory, x->getEntry(j + i) * 8, h2);
    ++h2;
  }

  // MotorSpeed
  j += i;
  for (i = 0; i < 3; i++)
  {
    tiles->tiles1(vector, 4, memory, x->getEntry(j + i) * 8, h2);
    ++h2;
    k = j + (i + 1) % 3;
    tiles->tiles2(vector, 8, memory, x->getEntry(j + i) * 8, x->getEntry(k) * 8, h2);
    ++h2;
  }

  // MotorVoltage
  j += i;
  for (i = 0; i < 3; i++)
  {
    tiles->tiles1(vector, 2, memory, x->getEntry(j + i) * 8, h2);
    ++h2;
  }

  // MotorCurrent
  j += i;
  for (i = 0; i < 3; i++)
  {
    tiles->tiles1(vector, 2, memory, x->getEntry(j + i) * 8, h2);
    ++h2;
  }

  // MotorTemperature
  j += i;
  for (i = 0; i < 3; i++)
  {
    tiles->tiles1(vector, 4, memory, x->getEntry(j + i) * 4, h2);
    ++h2;
  }

  // LastMotorRequest
  j += i;
  for (i = 0; i < 3; i++)
  {
    tiles->tiles1(vector, 4, memory, x->getEntry(j + i) * 6, h2);
    ++h2;
  }

  // OverheatingFlag
  j += i;
  for (i = 0; i < 1; i++)
  {
    tiles->tiles1(vector, 4, memory, x->getEntry(j + i) * 2, h2);
    ++h2;
  }
  vector->setEntry(vector->dimension() - 1, 1.0f);
  return vector;
}

double NextingProjector::vectorNorm() const
{
  return nbTiles + 1;
}

int NextingProjector::dimension() const
{
  return vector->dimension();
}

// ==

void NextingTest::testNexting()
{
  Random<double>* random = new Random<double>;
  RLProblem<double>* problem = new NextingProblem(random);
  Projector<double>* projector = new NextingProjector;
  StateToStateAction<double>* toStateAction = new TabularAction<double>(projector,
      problem->getDiscreteActions(), true);
  Policy<double>* policy = new RandomPolicy<double>(random, problem->getDiscreteActions());
  OnPolicyControlLearner<double>* control = new PolicyBasedControl<double>(policy, toStateAction);
  Horde<double>* horde = new Horde<double>(projector);
  RLAgent<double>* hordeAgent = new HordeAgent<double>(control, horde);
  RLRunner<double>* sim = new RLRunner<double>(hordeAgent, problem, 1, 1, 1);

  double alpha = 0.1 / projector->vectorNorm();
  double gamma = 0.98;
  double lambda = 0.7;

  int nbDemons = 8000 / 4; // per thread
  for (int i = 0; i < nbDemons; i++)
  {
    Trace<double>* e = new ATrace<double>(projector->dimension());
    OnPolicyTD<double>* td = new TDLambda<double>(alpha, gamma, lambda, e);
    RewardFunction<double>* reward = new FakeRewardFunction<double>();
    horde->addOnPolicyDemon(e, reward, td);
  }
  sim->run();

  delete random;
  delete problem;
  delete projector;
  delete toStateAction;
  delete policy;
  delete control;
  delete horde;
  delete hordeAgent;
  delete sim;
}

void NextingTest::run()
{
  testNexting();
}

RLLIB_TEST_MAKE(NextingTest)
