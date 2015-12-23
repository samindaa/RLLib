/*
 * TreeFittedTest.cpp
 *
 *  Created on: Nov 21, 2014
 *      Author: sam
 */

#include "TreeFittedTest.h"
#include "Mathema.h"
#include "util/TreeFitted/ExtraTreeEnsemble.h"

RLLIB_TEST_MAKE(TreeFittedTest)

TreeFittedTest::TreeFittedTest()
{
}

TreeFittedTest::~TreeFittedTest()
{
}

void TreeFittedTest::testRosenbrock()
{
  const int input_size = 2;
  const int output_size = 1;
  RLLib::Random<float>* random = new RLLib::Random<float>();
  RLLib::Range<float>* range = new RLLib::Range<float>(-2.0f, 2.0f);
  PoliFitted::Dataset* trainingDataset = new PoliFitted::Dataset(input_size, output_size);
  PoliFitted::Dataset* testingDataset = new PoliFitted::Dataset(input_size, output_size);
  PoliFitted::Regressor* reg = new PoliFitted::ExtraTreeEnsemble(input_size, output_size, 30, 2);

  for (int i = 0; i < 500 + 200; i++)
  {
    const float x = range->choose(random);
    const float y = range->choose(random);
    const float f_xy = std::pow((1 - x), 2) + 100.0f * std::pow((y - std::pow(x, 2)), 2);
    PoliFitted::Tuple* input = new PoliFitted::Tuple(input_size);
    PoliFitted::Tuple* output = new PoliFitted::Tuple(output_size);
    input->at(0) = x;
    input->at(1) = y;
    output->at(0) = f_xy;
    if (i < 500)
      trainingDataset->AddSample(input, output);
    else
      testingDataset->AddSample(input, output);
  }

  std::cout << "Learning: " << std::endl;
  reg->Train(trainingDataset);

  std::cout << "Evaluation: " << std::endl;
  for (PoliFitted::Dataset::iterator iter = testingDataset->begin(); iter != testingDataset->end();
      ++iter)
  {
    PoliFitted::Tuple* input = (*iter)->GetInputTuple();
    PoliFitted::Tuple* ouput = (*iter)->GetOutputTuple();
    PoliFitted::Tuple result(input_size);
    reg->Evaluate(input, result);
    std::cout << "f_actual: " << ouput->at(0) << " f_estimated: " << result.at(0) << std::endl;
  }

  std::cout << "training_err (L2): " << reg->ComputeTrainError(trainingDataset)
      << " testing_err (L2): " << reg->ComputeTrainError(testingDataset) << std::endl;

  for (PoliFitted::Dataset::iterator iter = trainingDataset->begin();
      iter != trainingDataset->end(); ++iter)
    delete (*iter);
  for (PoliFitted::Dataset::iterator iter = testingDataset->begin(); iter != testingDataset->end();
      ++iter)
    delete (*iter);

  delete random;
  delete range;
  delete trainingDataset;
  delete testingDataset;
  delete reg;

}

void TreeFittedTest::testRastrigin()
{
  const int input_size = 100;
  const int output_size = 1;
  const float A = 10.0f;
  RLLib::Random<float>* random = new RLLib::Random<float>();
  RLLib::Range<float>* range = new RLLib::Range<float>(-5.12f, 5.12f);
  PoliFitted::Dataset* trainingDataset = new PoliFitted::Dataset(input_size, output_size);
  PoliFitted::Dataset* testingDataset = new PoliFitted::Dataset(input_size, output_size);
  PoliFitted::Regressor* reg = new PoliFitted::ExtraTreeEnsemble(input_size,
      output_size/*using default parameters*/);

  const int nbTrainingSamples = 1000;
  const int nbTestingSamples = 200;
  for (int i = 0; i < nbTrainingSamples + nbTestingSamples; i++)
  {
    PoliFitted::Tuple* input = new PoliFitted::Tuple(input_size);
    PoliFitted::Tuple* output = new PoliFitted::Tuple(output_size);
    float f_x = A * input_size;
    for (int j = 0; j < input_size; j++)
    {
      input->at(j) = range->choose(random);
      f_x += (std::pow(input->at(j), 2) - A * cos(2.0f * M_PI * input->at(j)));
    }
    output->at(0) = f_x;
    if (i < nbTrainingSamples)
      trainingDataset->AddSample(input, output);
    else
      testingDataset->AddSample(input, output);
  }

  std::cout << "Learning: " << std::endl;
  reg->Train(trainingDataset);

  std::cout << "Evaluation: " << std::endl;
  for (PoliFitted::Dataset::iterator iter = testingDataset->begin(); iter != testingDataset->end();
      ++iter)
  {
    PoliFitted::Tuple* input = (*iter)->GetInputTuple();
    PoliFitted::Tuple* ouput = (*iter)->GetOutputTuple();
    PoliFitted::Tuple result(input_size);
    reg->Evaluate(input, result);
    std::cout << "f_actual: " << ouput->at(0) << " f_estimated: " << result.at(0) << std::endl;
  }

  std::cout << "training_err (L2): " << reg->ComputeTrainError(trainingDataset)
      << " testing_err (L2): " << reg->ComputeTrainError(testingDataset) << std::endl;

  for (PoliFitted::Dataset::iterator iter = trainingDataset->begin();
      iter != trainingDataset->end(); ++iter)
    delete (*iter);
  for (PoliFitted::Dataset::iterator iter = testingDataset->begin(); iter != testingDataset->end();
      ++iter)
    delete (*iter);

  delete random;
  delete range;
  delete trainingDataset;
  delete testingDataset;
  delete reg;

}

void TreeFittedTest::testCigar()
{
  const int input_size = 100;
  const int output_size = 1;
  RLLib::Random<float>* random = new RLLib::Random<float>();
  RLLib::Range<float>* range = new RLLib::Range<float>(-5, 5);
  PoliFitted::Dataset* trainingDataset = new PoliFitted::Dataset(input_size, output_size);
  PoliFitted::Dataset* testingDataset = new PoliFitted::Dataset(input_size, output_size);
  PoliFitted::Regressor* reg = new PoliFitted::ExtraTreeEnsemble(input_size, output_size, 50, 5, 2,
      0.0f, LeafType::LINEAR);
  RLLib::Timer* timer = new RLLib::Timer;

  const int nbTrainingSamples = 1000;
  const int nbTestingSamples = 200;
  for (int i = 0; i < nbTrainingSamples + nbTestingSamples; i++)
  {
    PoliFitted::Tuple* input = new PoliFitted::Tuple(input_size);
    PoliFitted::Tuple* output = new PoliFitted::Tuple(output_size);
    input->at(0) = range->choose(random);
    float f_x = std::pow(input->at(0), 2);
    for (int j = 1; j < input_size; j++)
    {
      input->at(j) = range->choose(random);
      f_x += std::pow(10.0f * input->at(j), 2);
    }
    output->at(0) = f_x;
    if (i < nbTrainingSamples)
      trainingDataset->AddSample(input, output);
    else
      testingDataset->AddSample(input, output);
  }

  std::cout << "Learning: " << std::endl;
  timer->start();
  reg->Train(trainingDataset);
  timer->stop();
  std::cout << "train (reg) (ms): " << timer->getElapsedTimeInMilliSec() << std::endl;

  std::cout << "Evaluation reg: " << std::endl;

  for (PoliFitted::Dataset::iterator iter = testingDataset->begin(); iter != testingDataset->end();
      ++iter)
  {
    PoliFitted::Tuple* input = (*iter)->GetInputTuple();
    PoliFitted::Tuple* ouput = (*iter)->GetOutputTuple();
    PoliFitted::Tuple result(input_size);
    timer->start();
    reg->Evaluate(input, result);
    timer->stop();
    std::cout << "f_actual: " << ouput->at(0) << " f_estimated: " << result.at(0) << " ms: "
        << timer->getElapsedTimeInMilliSec() << std::endl;
  }

  std::cout << "reg_training_err (L2): " << reg->ComputeTrainError(trainingDataset)
      << " reg_testing_err (L2): " << reg->ComputeTrainError(testingDataset) << std::endl;

  std::string fName = "ExtraTreeEnsemble.bin";
  std::ofstream of(fName);
  reg->WriteOnStream(of);

  PoliFitted::Regressor* reg2 = new PoliFitted::ExtraTreeEnsemble(input_size, output_size, 50, 5, 2,
      0.0f, LeafType::CONSTANT);
  std::ifstream in(fName);
  reg2->ReadFromStream(in);

  std::cout << "Evaluation reg2: " << std::endl;
  for (PoliFitted::Dataset::iterator iter = testingDataset->begin(); iter != testingDataset->end();
      ++iter)
  {
    PoliFitted::Tuple* input = (*iter)->GetInputTuple();
    PoliFitted::Tuple* ouput = (*iter)->GetOutputTuple();
    PoliFitted::Tuple result(input_size);
    timer->start();
    reg2->Evaluate(input, result);
    timer->stop();
    std::cout << "f_actual: " << ouput->at(0) << " f_estimated: " << result.at(0) << " ms: "
        << timer->getElapsedTimeInMilliSec() << std::endl;
  }

  std::cout << "reg2_training_err (L2): " << reg2->ComputeTrainError(trainingDataset)
      << " reg2_testing_err (L2): " << reg2->ComputeTrainError(testingDataset) << std::endl;

  for (PoliFitted::Dataset::iterator iter = trainingDataset->begin();
      iter != trainingDataset->end(); ++iter)
    delete (*iter);
  for (PoliFitted::Dataset::iterator iter = testingDataset->begin(); iter != testingDataset->end();
      ++iter)
    delete (*iter);

  delete random;
  delete range;
  delete trainingDataset;
  delete testingDataset;
  delete reg;
  delete reg2;
  delete timer;
}

void TreeFittedTest::testRegularizedLinearRegression()
{
  // This test uses Eigen3 functionality in all our fitting codes.
  std::string line;
  std::ifstream infile("databases/housing.data");
  double value;
  Eigen::MatrixXd X = Eigen::MatrixXd::Zero(506, 14); // These values are only housing.data
  Eigen::VectorXd y(506);
  X.col(0).array() += 1.0;

  int i = 0;
  while (std::getline(infile, line))  // this does the checking!
  {
    std::istringstream iss(line);
    if (line.size() > 0)
    {
      int j = 0;
      while (iss >> value)
      {
        if (j < 13)
          X(i, j + 1) = value;
        else
          y(i) = value;
        ++j;
      }
      ASSERT(j == 14);
    }
    ++i;
  }
  ASSERT(i == 506);

  double lambda = 1.0f;
  // print a row
  int myrow = 0;
  for (int j = 0; j < 14; j++)
    std::cout << X(myrow, j) << " ";
  std::cout << std::endl;
  std::cout << y(0);
  std::cout << std::endl;

  Eigen::MatrixXd A_new = (X.transpose() * X + lambda * MatrixXd::Identity(X.cols(), X.cols()));
  Eigen::VectorXd b_new = X.transpose() * y;
  Eigen::VectorXd theta_estimated = A_new.colPivHouseholderQr().solve(b_new); // Fit
  RLLib::PVector<double> theta(theta_estimated.rows());
  for (int i = 0; i < theta.dimension(); i++)
    theta.setEntry(i, (double) theta_estimated(i));
  std::cout << theta << std::endl;

  double mrsq = sqrt((y - X * theta_estimated).array().square().sum() / double(506.0));
  std::cout << "mrsq: " << mrsq << std::endl;

  Eigen::VectorXd y_estimated = X.topRows<10>() * theta_estimated;

  for (int i = 0; i < 10; i++)
    std::cout << y(i) << " " << y_estimated(i) << "\n";
  std::cout << std::endl;
}

void TreeFittedTest::run()
{
  testRosenbrock();
  testRastrigin();
  testCigar();
  testRegularizedLinearRegression();
}

