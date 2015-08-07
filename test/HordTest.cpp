/*
 * HordTest.cpp
 *
 *  Created on: Jul 20, 2015
 *      Author: sam
 */

#include "HordTest.h"

#define DATA_FILE "./databases/0.txt"
#define PRED_FILE "./databases/pred2.txt"
#define GAMMA 0.5f
#define LAMBDA 0.9f

RLLIB_TEST_MAKE(HordTest)

HordTest::HordTest()
{
  random = new RLLib::Random<double>;
  problem = new HordeProblem(random);
  hashing = new RLLib::MurmurHashing<double>(random, 512 * 4);
  projector = new HordeProjector(hashing);
  toStateAction = new RLLib::StateActionTilings<double>(projector, problem->getDiscreteActions());
  e = new RLLib::ATrace<double>(projector->dimension());

  alpha = 0.1f / projector->vectorNorm();
  gamma = GAMMA;
  lambda = LAMBDA;

  sarsa = new SarsaTrue<double>(alpha, gamma, lambda, e);
  target = new HordePolicy(problem->getDiscreteActions(), sarsa);
  control = new SarsaControl<double>(target, toStateAction, sarsa);

  agent = new RLLib::LearnerAgent<double>(control);
  runner = new RLLib::RLRunner<double>(agent, problem, -1, 4000, 1);
}

HordTest::~HordTest()
{
  delete random;
  delete problem;
  delete hashing;
  delete projector;
  delete toStateAction;
  delete e;
  delete target;
  delete control;
  delete agent;
  delete runner;
}

void HordTest::run()
{
  runner->run();

  problem->setNextLine(0);
  runner->runEvaluate(10, 1);
}

// RLLib
HordeProblem::HordeProblem(RLLib::Random<double>* random) :
    RLLib::RLProblem<double>(random, //
        HordeProblem_NUM_VAR * HordeProblem_HIS_LEN, //
        1/*predictor problem*/, 1/*predictor problem*/), //
    m_endOfEpisode(false), m_r(0), nextLine(0)
{
  sonarRange = new RLLib::Range<double>(0.25f, 2.55f);

  // read data
  std::ifstream ifs(DATA_FILE);
  std::string str;
  int nbRows = 0;
  while (std::getline(ifs, str))
  {
    if (str.size() == 0)
      continue;
    std::istringstream iss(str);
    std::vector<double> tokens //
    { std::istream_iterator<double> { iss }, std::istream_iterator<double> { } };
    M_data.conservativeResize(nbRows + 1, tokens.size());

    for (size_t i = 0; i < tokens.size(); ++i)
      M_data(nbRows, i) = tokens[i];
    ++nbRows;
  }

  std::cout << "M_data: " << M_data.rows() << "x" << M_data.cols() << std::endl;
}

HordeProblem::~HordeProblem()
{
  delete sonarRange;
}

void HordeProblem::updateStateVariables()
{
  //std::cout << m_x.transpose() << std::endl;

  int x_h = 0;
  for (int h = 0; h < HordeProblem_HIS_LEN; ++h)
  {
    for (int v = 0; v < HordeProblem_NUM_VAR; ++v)
    {
      output->observation_tp1->setEntry(h * HordeProblem_NUM_VAR + v,
          (double) m_x(h * HordeProblem_NUM_VAR + v));
      output->o_tp1->setEntry(h * HordeProblem_NUM_VAR + v,
          sonarRange->toUnit(output->observation_tp1->getEntry(h * HordeProblem_NUM_VAR + v)));
      ++x_h;
    }
  }
  assert(x_h == HordeProblem_HIS_LEN * HordeProblem_NUM_VAR);

  //std::cout << output->o_tp1 << std::endl;
  //std::cout << output->observation_tp1 << std::endl;
}

void HordeProblem::initialize()
{
  if (nextLine == M_data.rows())
  {
    nextLine = 0;
    //std::cout << "nextLine rest" << std::endl;
  }

  m_x = M_data.row(nextLine);
  assert(m_x(m_x.size() - 1) == -1);
  updateStateVariables();
}

void HordeProblem::step(const RLLib::Action<double>* action)
{
  ++nextLine;
  assert(nextLine < M_data.rows());
  m_x = M_data.row(nextLine);
  m_endOfEpisode = (m_x(m_x.size() - 1) == 2);
  m_r = m_x(10);

  if (m_endOfEpisode)
  {
    //std::cout << "episode ended" << std::endl;
    ++nextLine; // Advance to next episode
  }

}

void HordeProblem::updateTRStep()
{
  updateStateVariables();
}

bool HordeProblem::endOfEpisode() const
{
  return m_endOfEpisode;
}

double HordeProblem::r() const
{
  return m_r;
}

double HordeProblem::z() const
{
  return 0;
}

// ==
HordeProjector::HordeProjector(RLLib::Hashing<double>* hashing)
{
  std::cout << "memory: " << hashing->getMemorySize() << std::endl;
  vector = new RLLib::SVector<double>(hashing->getMemorySize() + 1 /*bias*/);
  std::cout << "vector: " << vector->dimension() << std::endl;
  random = new RLLib::Random<double>;
  tiles = new RLLib::Tiles<double>(hashing);
  inputs = new RLLib::PVector<double>(HordeProblem_NUM_VAR);
  gridResolutions = new RLLib::PVector<double>(HordeProblem_NUM_VAR);
  numTilings = 8;
  gridResolution = 12.0f;
  gridResolutions->set(gridResolution);
}

HordeProjector::~HordeProjector()
{
  delete vector;
  delete random;
  delete tiles;
}

const RLLib::Vector<double>* HordeProjector::project(const RLLib::Vector<double>* x, const int& h1)
{
  return project(x, h1, 1);
}

const RLLib::Vector<double>* HordeProjector::project(const RLLib::Vector<double>* x)
{
  return project(x, -1, -1);
}

const RLLib::Vector<double>* HordeProjector::project(const RLLib::Vector<double>* x, const int& h1,
    const int& h3)
{
  vector->clear();
  if (x->empty())
    return vector;

  assert(x->dimension() == HordeProblem_HIS_LEN * HordeProblem_NUM_VAR);
  int h2 = 0;
  int x_h = 0;
  for (int h = 0; h < HordeProblem_HIS_LEN; ++h)
  {
    for (int v = 0; v < HordeProblem_NUM_VAR; ++v)
    {
      inputs->setEntry(v, x->getEntry(h * HordeProblem_NUM_VAR + v));
      ++x_h;
    }

    //std::cout << inputs << std::endl;
    inputs->ebeMultiplyToSelf(gridResolutions);
    if (h3 > 0)
      tiles->tiles(vector, numTilings, inputs, h1, h2);
    else
      tiles->tiles(vector, numTilings, inputs, h2);
    ++h2;
  }
  assert(x_h == x->dimension());
  vector->setEntry(vector->dimension() - 1, 1.0);
  return vector;
}

double HordeProjector::vectorNorm() const
{
  return numTilings * HordeProblem_HIS_LEN + 1;
}

int HordeProjector::dimension() const
{
  return vector->dimension();
}

// ==

HordePolicy::HordePolicy(Actions<double>* actions, Predictor<double>* predictor) :
    actions(actions), predictor(predictor)
{
  m_ofstream.open(PRED_FILE);
  m_prediction = 0.0f;
}

void HordePolicy::update(const Representations<double>* phi_tp1)
{
  for (Actions<double>::const_iterator iter = actions->begin(); iter != actions->end(); ++iter)
    m_prediction = predictor->predict(phi_tp1->at(*iter));
}

double HordePolicy::pi(const Action<double>* a)
{
  return 1.0f;
}

const Action<double>* HordePolicy::sampleAction()
{

  return actions->getEntry(0);
}

const Action<double>* HordePolicy::sampleBestAction()
{
  m_ofstream << (m_prediction) << std::endl;
  return actions->getEntry(0);
}
