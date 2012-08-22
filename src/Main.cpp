//============================================================================
// Name        : RLLib.cpp
// Author      : Sam Abeyruwan
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <map>
#include <fstream>

// From the RLLib
#include "Vector.h"
#include "Trace.h"
#include "Projector.h"
#include "Algorithm.h"
#include "../simulation/Simulator.h"
#include "../simulation/MCar2D.h"

using namespace std;

void testFullVector()
{
  DenseVector<float> v(10);
  cout << v << endl;
  srand48(time(0));
  for (int i = 0; i < v.dimension(); i++)
  {
    double k = drand48();
    v[i] = k;
    cout << k << " ";
  }
  cout << endl;
  cout << v << endl;
  DenseVector<float> d;
  cout << d << endl;
  d = v;
  d * 100;
  cout << d << endl;
  cout << d.maxNorm() << endl;

  DenseVector<float> i(5);
  i[0] = 1.0;
  cout << i << endl;
  cout << i.maxNorm() << endl;
  cout << i.euclideanNorm() << endl;
}

void testSparseVector()
{
  srand48(time(0));
  /*SparseVector<> s(16);
   cout << s << endl;

   for (int i = 0; i < s.dimension() ; i++)
   {
   double k = drand48();
   s.insertEntry(i, k);
   cout << "[i=" << i << " v=" << k << "] ";
   }
   cout << endl;
   cout << s << endl;*/

  SparseVector<float> a(20);
  SparseVector<float> b(20);
  for (int i = 0; i < 5; i++)
  {
    a.insertEntry(i, 1);
    b.insertEntry(i, 2);
  }

  cout << a << endl;
  cout << b << endl;
  cout << a.numActiveEntries() << " " << b.numActiveEntries() << endl;
  b.removeEntry(2);
  cout << a.numActiveEntries() << " " << b.numActiveEntries() << endl;
  cout << a << endl;
  cout << b << endl;
  cout << "dot=" << a.dot(b) << endl;
  cout << a.addToSelf(b) << endl;
  a.clear();
  b.clear();
  cout << a << endl;
  cout << b << endl;

}

void testProjector()
{
  srand48(time(0));

  int numObservations = 2;
  int memorySize = 512;
  int numTiling = 32;
  SparseVector<double> w(memorySize);
  for (int t = 0; t < 50; t++)
    w.insertEntry(rand() % memorySize, drand48());
  FullTilings<double, float> coder(memorySize, numTiling, true);
  DenseVector<float> x(numObservations);
  for (int p = 0; p < 5; p++)
  {
    for (int o = 0; o < numObservations; o++)
      x[o] = drand48() / 0.25;
    const SparseVector<double>& vect = coder.project(x);
    cout << w << endl;
    cout << vect << endl;
    cout << w.dot(vect) << endl;
    cout << "---------" << endl;
  }
}

void testProjectorMachineLearning()
{
  // simple sine curve estimation
  // training samples
  srand48(time(0));
  multimap<double, double> X;
  for (int i = 0; i < 100; i++)
  {
    double x = -M_PI_2 + 2 * M_PI * drand48(); // @@>> input noise?
    double y = sin(2 * x); // @@>> output noise?
    X.insert(make_pair(x, y));
  }

  // train
  int numObservations = 1;
  int memorySize = 512;
  int numTiling = 32;
  FullTilings<double, float> coder(memorySize, numTiling, true);
  SparseVector<double> w(memorySize);
  DenseVector<float> x(numObservations);
  int traininCounter = 0;
  while (++traininCounter < 100)
  {
    for (multimap<double, double>::const_iterator iter = X.begin();
        iter != X.end(); ++iter)
    {
      x[0] = iter->first / (2 * M_PI) / 0.25; // normalized and unit generalized
      const SparseVector<double>& phi = coder.project(x);
      double result = w.dot(phi);
      double alpha = 0.1 / coder.vectorNorm();
      w.addToSelf(alpha * (iter->second - result), phi);
    }
  }

  // output
  ofstream outFile("mest.dat");
  for (multimap<double, double>::const_iterator iter = X.begin();
      iter != X.end(); ++iter)
  {
    x[0] = iter->first / (2 * M_PI) / 0.25;
    const SparseVector<double>& phi = coder.project(x);
    if (outFile.is_open())
      outFile << iter->first << " " << iter->second << " " << w.dot(phi)
          << endl;
  }
  outFile.close();
}

void testSarsaMountainCar()
{
  Env<float>* problem = new MCar2D;
  Projector<double, float>* projector = new FullTilings<double, float>(
      1000000 + 1, 10, false);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<
      double, float>(projector, problem->getActionList());
  Trace<double>* e = new RTrace<double>(projector->dimension());
  double alpha = 0.15 / projector->vectorNorm();
  double gamma = 0.99;
  double lambda = 0.9;
  Sarsa<double>* sarsa = new Sarsa<double>(alpha, gamma, lambda, e);
  double epsilon = 0.01;
  Policy<double>* acting = new EpsilonGreedy<double>(sarsa,
      problem->getActionList(), epsilon);
  OnPolicyControlLearner<double, float>* control = new SarsaControl<double,
      float>(acting, toStateAction, sarsa);

  Simulator<double, float>* sim = new Simulator<double, float>(control,
      problem);
  sim->run(10, 5000, 100);

  delete problem;
  delete projector;
  delete toStateAction;
  delete e;
  delete sarsa;
  delete acting;
  delete control;
  delete sim;
}

void testGreedyGQMountainCar()
{
  srand(time(0));
  srand48(time(0));
  Env<float>* problem = new MCar2D;
  Projector<double, float>* projector = new FullTilings<double, float>(
      1000000 + 1, 10, false);
  StateToStateAction<double, float>* toStateAction = new StateActionTilings<
      double, float>(projector, problem->getActionList());
  Trace<double>* e = new ATrace<double>(projector->dimension());
  double alpha_v = 0.1 / projector->vectorNorm();
  double alpha_w = .0001 / projector->vectorNorm();
  double gamma_tp1 = 0.99;
  double beta_tp1 = 1.0 - gamma_tp1;
  double lambda_t = 0.4;
  GQ<double>* gq = new GQ<double>(alpha_v, alpha_w, beta_tp1, lambda_t, e);
  double epsilon = 0.5;
  Policy<double>* behavior = new EpsilonGreedy<double>(gq,
      problem->getActionList(), epsilon);
  Policy<double>* target = new Greedy<double>(gq, problem->getActionList());
  OffPolicyControlLearner<double, float>* control = new GreedyGQ<double, float>(
      target, behavior, problem->getActionList(), toStateAction, gq);

  Simulator<double, float>* sim = new Simulator<double, float>(control,
      problem);
  sim->run(20, 5000, 500);

  delete problem;
  delete projector;
  delete toStateAction;
  delete e;
  delete gq;
  delete behavior;
  delete target;
  delete control;
  delete sim;
}

int main()
{
  cout << "## start" << endl; // prints @@ start
//  testSparseVector();
//  testProjector();
//  testProjectorMachineLearning();
//  testSarsaMountainCar();
  testGreedyGQMountainCar();
  cout << "## end" << endl;
  return 0;
}
