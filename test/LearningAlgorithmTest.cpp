/*
 * LearningAlgorithmTest.cpp
 *
 *  Created on: Dec 18, 2012
 *      Author: sam
 */

#include "HeaderTest.h"
#include "Projector.h"
#include "SupervisedAlgorithm.h"

class AdalineTest
{
  public:
    AdalineTest()
    {
      srand(time(0));
    }

    void run()
    {
      // simple sine curve estimation
      // training samples
      srand(time(0));
      multimap<double, double> X;
      for (int i = 0; i < 100; i++)
      {
        double x = -M_PI_2 + 2 * M_PI * Random::nextDouble(); // @@>> input noise?
        double y = sin(2 * x); // @@>> output noise?
        X.insert(make_pair(x, y));
      }

      // train
      int numObservations = 1;
      int memorySize = 512;
      int numTiling = 32;
      FullTilings<double, float> coder(memorySize, numTiling, true);
      DVecFloatType x(numObservations);
      Adaline<double> lms(coder.dimension(), 0.1 / coder.vectorNorm());
      int traininCounter = 0;
      while (++traininCounter < 100)
      {
        for (multimap<double, double>::const_iterator iter = X.begin();
            iter != X.end(); ++iter)
        {
          x[0] = iter->first / (2 * M_PI) / 0.25; // normalized and unit generalized
          const SVecDoubleType& phi = coder.project(x);
          lms.learn(phi, iter->second);
        }
      }

      // output
      ofstream outFile("visualization/mest.dat");
      for (multimap<double, double>::const_iterator iter = X.begin();
          iter != X.end(); ++iter)
      {
        x[0] = iter->first / (2 * M_PI) / 0.25;
        const SVecDoubleType& phi = coder.project(x);
        if (outFile.is_open())
          outFile << iter->first << " " << iter->second << " "
              << lms.predict(phi) << endl;
      }
      outFile.close();
    }
};

int main(int argc, char** argv)
{
  AdalineTest adalineTest;
  adalineTest.run();
  return 0;
}

