/*
 * LearningAlgorithmTest.cpp
 *
 *  Created on: Dec 18, 2012
 *      Author: sam
 */

#include "HeaderTest.h"
#include "Projector.h"
#include "SupervisedAlgorithm.h"

class SupervisedAlgorithmTest
{
  public:
    SupervisedAlgorithmTest()
    {
      srand(time(0));
    }

    void run()
    {
      // simple sine curve estimation
      // training samples
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
      int numTiling = 16;
      TileCoderHashing<double, float> coder(memorySize, numTiling, true);
      DVecFloatType x(numObservations);
      Adaline<double> adaline(coder.dimension(), 0.1 / coder.vectorNorm());
      IDBD<double> idbd(coder.dimension(), 0.001); // This value looks good
      Autostep<double> autostep(coder.dimension());
      int traininCounter = 0;
      ofstream outFileError("visualization/learningAlgorithmTestError.dat");
      while (++traininCounter < 100)
      {
        for (multimap<double, double>::const_iterator iter = X.begin();
            iter != X.end(); ++iter)
        {
          x[0] = 2.0 * iter->first / M_PI; // normalized and unit generalized
          const SVecDoubleType& phi = coder.project(x);
          adaline.learn(phi, iter->second);
          idbd.learn(phi, iter->second);
          autostep.learn(phi, iter->second);
        }

        // Calculate the error
        double mse[3] =
        { 0 };
        for (multimap<double, double>::const_iterator iterMse = X.begin();
            iterMse != X.end(); ++iterMse)
        {
          x[0] = 2.0 * iterMse->first / M_PI;
          const SVecDoubleType& phi = coder.project(x);
          mse[0] += pow(iterMse->second - adaline.predict(phi), 2) / X.size();
          mse[1] += pow(iterMse->second - idbd.predict(phi), 2) / X.size();
          mse[2] += pow(iterMse->second - autostep.predict(phi), 2) / X.size();
        }
        if (outFileError.is_open())
          outFileError << mse[0] << " " << mse[1] << " " << mse[2] << endl;
      }
      outFileError.close();

      // output
      ofstream outFilePrediction(
          "visualization/learningAlgorithmTestPrediction.dat");
      for (multimap<double, double>::const_iterator iter = X.begin();
          iter != X.end(); ++iter)
      {
        x[0] = 2.0 * iter->first / M_PI;
        const SVecDoubleType& phi = coder.project(x);
        if (outFilePrediction.is_open())
          outFilePrediction << iter->first << " " << iter->second << " "
              << adaline.predict(phi) << " " << idbd.predict(phi) << " "
              << autostep.predict(phi) << endl;
      }
      outFilePrediction.close();
    }
};

int main(int argc, char** argv)
{
  cout << "*** LearningAlgorithmTest starts " << endl;
  SupervisedAlgorithmTest supervisedAlgorithmTest;
  supervisedAlgorithmTest.run();
  cout << "*** LearningAlgorithmTest ends " << endl;
  return 0;
}

