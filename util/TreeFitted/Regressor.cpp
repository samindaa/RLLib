/************************************************************************
 Regressor.h.cpp - Copyright marcello

 Here you can write a license for your code, some comments or any other
 information you want to have in your generated code. To to this simply
 configure the "headings" directory in uml to point to a directory
 where you have your heading files.

 or you can just replace the contents of this file with your own.
 If you want to do this, this file is located at

 /usr/share/apps/umbrello/headings/heading.cpp

 -->Code Generators searches for heading files based on the file extension
 i.e. it will look for a file name ending in ".h" to include in C++ header
 files, and for a file name ending in ".java" to include in all generated
 java code.
 If you name the file "heading.<extension>", Code Generator will always
 choose this file even if there are other files with the same extension in the
 directory. If you name the file something else, it must be the only one with that
 extension in the directory to guarantee that Code Generator will choose it.

 you can use variables in your heading files which are replaced at generation
 time. possible variables are : author, date, time, filename and filepath.
 just write %variable_name%

 This file was generated on Sat Nov 10 2007 at 15:05:38
 The original location of this file is /home/marcello/Projects/fitted/Developing/Regressor.cpp
 **************************************************************************/

#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <assert.h>

#include "Regressor.h"

//<< Sam
using namespace PoliFitted;

// Constructors/Destructors
//

Regressor::Regressor(string type, unsigned int input_size, unsigned int output_size) :
    mType(type), mInputSize(input_size), mOutputSize(output_size), mIsNormalized(false)
{
}

Regressor::~Regressor()
{
}

//
// Methods
//

float Regressor::ComputeTrainError(Dataset* data, NormType type)
{
  assert(data != 0);
  unsigned int output_size = data->GetOutputSize();
  float error = 0;
  Dataset::iterator it;
  for (it = data->begin(); it != data->end(); it++)
  {
    Tuple result(mOutputSize);
    Evaluate((*it)->GetInputTuple(), result);
    float e_i = result.Distance((*it)->GetOutputTuple(), output_size, L1);
    switch (type)
    {
    case L1:
      error += fabs(e_i);
      break;
    case L2:
      error += e_i * e_i;
      break;
    case Linfty:
      error = (e_i > error) ? e_i : error;
      break;
    default:
      cerr << "ERROR: Undefined norm type!" << endl;
      exit(0);
    }
  }

  if (L2 == type)
  {
    error = sqrt(error);
  }

  return error;
}

map<string, float> Regressor::ComputePerfMetrics(Dataset* data)
{
  assert(data != 0);
  map<string, float> metrics;
  float raae = 0.0, rsquare = 0.0, rmae = 0.0, mse = 0.0, m = 0.0, m2 = 0.0, variance = 0.0, y_ps =
      -1e6, y_po = -1e6;
  Dataset::iterator it;
  for (it = data->begin(); it != data->end(); ++it)
  {
    Tuple result(mOutputSize);
    Evaluate((*it)->GetInputTuple(), result);
    float output = (*(*it)->GetOutputTuple())[0];
    if (output > y_po)
    {
      y_po = output;
      y_ps = result[0];
    }
    float e_i = output - result[0];
    mse += e_i * e_i;
    rmae = (fabs(e_i) > rmae) ? fabs(e_i) : rmae;
    raae += fabs(e_i);
    m += output;
    m2 += output * output;
  }
  m /= (float) data->size();
  m2 /= (float) data->size();
  mse /= (float) data->size();
  variance = m2 - m * m;

  rsquare = 1 - mse / variance;
  raae = raae / (sqrtf(variance) * (float) data->size());
  rmae = rmae / sqrtf(variance);
  metrics.insert(pair<string, float>(string("R2"), rsquare));
  metrics.insert(pair<string, float>(string("RAAE"), raae));
  metrics.insert(pair<string, float>(string("RMAE"), rmae));
  metrics.insert(pair<string, float>(string("RMSE"), sqrtf(mse)));
  metrics.insert(pair<string, float>(string("PEP"), 100 * (y_ps - y_po) / y_po));

  return metrics;
}

Dataset* Regressor::EvaluateOnDataset(Dataset* data)
{
  Dataset* ds = new Dataset(mInputSize, mOutputSize);
  Dataset::iterator it;
  for (it = data->begin(); it != data->end(); it++)
  {
    Tuple* ti = (*it)->GetInputTuple();
    Tuple* to = new Tuple(mOutputSize);
    Evaluate(ti, *to);
    Sample* s = new Sample(ti, to);
    ds->push_back(s);
  }
  return ds;
}

float Regressor::ComputeResiduals(Dataset* data, set<unsigned int> inputs,
    set<unsigned int> outputs, set<unsigned int> outputs_residual)
{
  float mse = 0.0, m = 0.0, m2 = 0.0;
  unsigned int output_id = *outputs.begin() - data->GetInputSize();
  Dataset::iterator it;
  for (it = data->begin(); it != data->end(); ++it)
  {
    Tuple* input = (*it)->GetInputTuple(inputs);
    float output = (*(*it)->GetOutputTuple())[output_id];
    Tuple result(mOutputSize);
    Evaluate(input, result);
    float e_i = output - result[0];
    mse += e_i * e_i;
    m += output;
    m2 += output * output;
    (*it)->GetOutput(*(outputs_residual.begin())) = e_i;
//     cout << *(outputs_residual.begin()) << " input = " << (*input)[0] << " result = " << result[0] << " output = " << output << " e_i = " << e_i << endl;
    delete input;
  }
  m /= (float) data->size();
  m2 /= (float) data->size();
  mse /= (float) data->size();
//   cout << "R2 = " <<  1 - mse/(m2-m*m) << " mse = " << mse << " m2 = " << m2 << " m = " << m<< endl;
  return 1 - mse / (m2 - m * m); // R^2
}

map<string, float> Regressor::CrossValidate(Dataset* data, unsigned int num_folds, bool log,
    string filename)
{
  map<string, float> result_metrics;
  result_metrics["R2"] = 0.0;
  result_metrics["RAAE"] = 0.0;
  result_metrics["RMAE"] = 0.0;
  result_metrics["RMSE"] = 0.0;
  result_metrics["PEP"] = 0.0;
  if (log)
  {
    ofstream log_file;
    log_file.open(filename.c_str(), ios::out);
    log_file.close();
  }

  for (unsigned int i = 0; i < num_folds; i++)
  {
    Dataset ds_train(data->GetInputSize(), data->GetOutputSize());
    Dataset ds_test(data->GetInputSize(), data->GetOutputSize());
    data->GetTrainAndTestDataset(num_folds, i, &ds_train, &ds_test);
//     cout << "Training..."; flush(cout);
    Initialize();
    Train(&ds_train);
//     cout << " done" << endl;

    map<string, float> metrics = ComputePerfMetrics(&ds_test);
    result_metrics["R2"] += metrics["R2"];
    result_metrics["RAAE"] += metrics["RAAE"];
    result_metrics["RMAE"] += metrics["RMAE"];
    result_metrics["RMSE"] += metrics["RMSE"];
    result_metrics["PEP"] += metrics["PEP"];

    if (log)
    {
      Dataset* data = EvaluateOnDataset(&ds_test);
      data->Save(filename, APPEND);
    }
    cout << "Testing:" << endl;
    cout << "R2 = " << metrics["R2"] << " RAAE = " << metrics["RAAE"] << " RMAE = "
        << metrics["RMAE"] << " RMSE = " << metrics["RMSE"] << " PEP = " << metrics["PEP"] << endl;
  }
  result_metrics["R2"] /= num_folds;
  result_metrics["RAAE"] /= num_folds;
  result_metrics["RMAE"] /= num_folds;
  result_metrics["RMSE"] /= num_folds;
  result_metrics["PEP"] /= num_folds;

  return result_metrics;
}

double* Regressor::NormalizeInput(double* data)
{
  double* new_data = new double[mInputParameters.size()];
  for (unsigned int i = 0; i < mInputParameters.size(); i++)
  {
    double range = mInputParameters[i].second - mInputParameters[i].first;
    if (range > 1e-16)
      new_data[i] = ((data[i] - mInputParameters[i].first) * 2.0) / range - 1.0;
  }
  return new_data;
}

double* Regressor::DenormalizeOutput(double* data)
{
  double* new_data = new double[mOutputParameters.size()];
  for (unsigned int i = 0; i < mOutputParameters.size(); i++)
  {
    double range = mOutputParameters[i].second - mOutputParameters[i].first;
    if (range > 1e-16)
      new_data[i] = ((data[i] + 1.0) / 2.0) * range + mOutputParameters[i].first;
  }
  return new_data;
}

Tuple* Regressor::NormalizeInput(Tuple* data)
{
  Tuple* new_data = new Tuple(mInputParameters.size());
  for (unsigned int i = 0; i < mInputParameters.size(); i++)
  {
    double range = mInputParameters[i].second - mInputParameters[i].first;
    if (range > 1e-16)
      (*new_data)[i] = (((*data)[i] - mInputParameters[i].first) * 2.0) / range - 1.0;
  }
  return new_data;
}

Tuple* Regressor::DenormalizeOutput(Tuple* data)
{
  Tuple* new_data = new Tuple(mOutputParameters.size());
  for (unsigned int i = 0; i < mOutputParameters.size(); i++)
  {
    double range = mOutputParameters[i].second - mOutputParameters[i].first;
    if (range > 1e-16)
      (*new_data)[i] = (((*data)[i] + 1.0) / 2.0) * range + mOutputParameters[i].first;
  }
  return new_data;
}

void Regressor::PostWriteOnStream(ofstream& out)
{
  vector<pair<float, float> >::iterator it;
  out << mInputParameters.size() << endl;
  for (it = mInputParameters.begin(); it != mInputParameters.end(); ++it)
  {
    out << it->first << " " << it->second << endl;
  }
  out << mOutputParameters.size() << endl;
  for (it = mOutputParameters.begin(); it != mOutputParameters.end(); ++it)
  {
    out << it->first << " " << it->second << endl;
  }
}

void Regressor::PostReadFromStream(ifstream& in)
{
  mInputParameters.clear();
  mOutputParameters.clear();
  unsigned int num;
  in >> num;
  for (unsigned int i = 0; i < num; i++)
  {
    pair<float, float> p;
    in >> p.first;
    in >> p.second;
    mInputParameters.push_back(p);
  }
  in >> num;
  for (unsigned int i = 0; i < num; i++)
  {
    pair<float, float> p;
    in >> p.first;
    in >> p.second;
    mOutputParameters.push_back(p);
    mIsNormalized = true;
  }
}

ofstream& PoliFitted::operator<<(ofstream& out, PoliFitted::Regressor& r)
{
  r.WriteOnStream(out);
  r.PostWriteOnStream(out);
  return out;
}

ifstream& PoliFitted::operator>>(ifstream& in, PoliFitted::Regressor& r)
{
  r.ReadFromStream(in);
  r.PostReadFromStream(in);
  return in;
}

