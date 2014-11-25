/************************************************************************
 Dataset.h.cpp - Copyright marcello

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
 The original location of this file is /home/marcello/Projects/fitted/Developing/Dataset.cpp
 **************************************************************************/

#include <stdlib.h>
#include <iostream>
#include <assert.h>

//#include <gsl/gsl_statistics.h>
//#include <gsl/gsl_linalg.h>
//#include <gsl/gsl_blas.h>

//<< Sam
#include "Dataset.h"

using namespace PoliFitted;

// Constructors/Destructors
//

Dataset::Dataset(unsigned int input_size, unsigned int output_size)
{
  mInputSize = input_size;
  mOutputSize = output_size;
  mMean = 0.0;
  mVariance = 0.0;
}

Dataset::~Dataset()
{
}
//
// Methods
//

void Dataset::Clear(bool clear_input)
{
  for (vector<Sample*>::iterator it = begin(); it != end(); ++it)
  {
    if (clear_input)
    {
      (*it)->ClearInput();
    }
    delete *it;
  }
}

// Accessor methods
//

/*double* Dataset::GetMatrix()
 {
 unsigned int num_cols = mInputSize + mOutputSize;
 double* matrix = (double*) malloc(sizeof(double) * size() * num_cols);
 for (unsigned int i = 0; i < size(); i++)
 {
 for (unsigned int j = 0; j < mInputSize; j++)
 {
 matrix[i * num_cols + j] = at(i)->GetInput(j);
 }
 for (unsigned int j = 0; j < mOutputSize; j++)
 {
 matrix[i * num_cols + mInputSize + j] = at(i)->GetOutput(j);
 }
 }
 return matrix;
 }*/

/*gsl_matrix* Dataset::GetGSLMatrix()
 {
 unsigned int num_cols = mInputSize + mOutputSize;
 gsl_matrix* matrix = gsl_matrix_alloc(size(), num_cols);
 for (unsigned int i = 0; i < size(); i++)
 {
 for (unsigned int j = 0; j < mInputSize; j++)
 {
 gsl_matrix_set(matrix, i, j, at(i)->GetInput(j));
 }
 for (unsigned int j = 0; j < mOutputSize; j++)
 {
 gsl_matrix_set(matrix, i, j, at(i)->GetOutput(j));
 }
 }
 return matrix;
 }*/

/*gsl_matrix* Dataset::GetCovarianceMatrix()
 {
 gsl_vector_view a, b;
 size_t i, j;
 gsl_matrix* m = GetGSLMatrix();
 gsl_matrix* r = gsl_matrix_alloc(m->size2, m->size2);
 for (i = 0; i < m->size2; i++)
 {
 for (j = 0; j < m->size2; j++)
 {
 double v;
 a = gsl_matrix_column(m, i);
 b = gsl_matrix_column(m, j);
 v = gsl_stats_covariance(a.vector.data, a.vector.stride, b.vector.data, b.vector.stride,
 a.vector.size);
 gsl_matrix_set(r, i, j, v);
 }
 }
 return r;
 }*/

Dataset* Dataset::Clone()
{
  Dataset* ds = new Dataset(mInputSize, mOutputSize);
  unsigned int size = this->size();
  for (unsigned int i = 0; i < size; i++)
  {
    ds->push_back(this->at(i)->Clone(mInputSize, mOutputSize));
  }
  return ds;
}

void Dataset::ResizeOutput(unsigned int new_size)
{
  Dataset::iterator it = this->begin();
  for (; it != this->end(); ++it)
  {
    (*it)->ResizeOutput(new_size);
  }
  mOutputSize = new_size;
}

Dataset* Dataset::GetReducedDataset(unsigned int size, bool random)
{
  Dataset* ds = new Dataset(mInputSize, mOutputSize);
  unsigned int old_size = this->size();
  if (size > old_size)
  {
    return this;
  }
  else
  {
    if (random)
    {
      unsigned int i = 0;
      while (i < size)
      {
        ds->push_back(this->at(rand() % size));
        i++;
      }
    }
    else
    {
      unsigned int i = 0;
      while (i < size)
      {
        ds->push_back(this->at(i));
        i++;
      }
    }
  }
  return ds;
}

Dataset* Dataset::GetReducedDataset(float proportion, bool random)
{
  Dataset* ds = new Dataset(mInputSize, mOutputSize);
  unsigned int old_size = this->size();
  if (proportion > 1.0)
  {
    return this;
  }
  else if (proportion < 0.0)
  {
    return 0;
  }
  else
  {
    unsigned int size = old_size * proportion;
    if (random)
    {
      unsigned int i = 0;
      while (i < size)
      {
        ds->push_back(this->at(rand() % size));
        i++;
      }
    }
    else
    {
      unsigned int i = 0;
      while (i < size)
      {
        ds->push_back(this->at(i));
        i++;
      }
    }
  }
  return ds;
}

void Dataset::GetTrainAndTestDataset(unsigned int num_partitions, unsigned int partition,
    Dataset* train_ds, Dataset* test_ds)
{
  unsigned int size = this->size();
  assert(size > num_partitions);
  unsigned int partition_size = size / num_partitions;
  for (unsigned int i = 0; i < size; i++)
  {
    if (i < partition * partition_size || i > (partition + 1) * partition_size)
    {
      train_ds->push_back((*this)[i]);
    }
    else
    {
      test_ds->push_back((*this)[i]);
    }
  }
}

map<float, Dataset*> Dataset::SplitByAttribute(unsigned int attribute)
{
  map<float, Dataset*> data;
  map<float, Dataset*>::iterator ds_it;
  Dataset::iterator it;
  for (it = this->begin(); it != this->end(); it++)
  {
    float value = (*it)->GetInput(attribute);
    ds_it = data.find(value);
    if (ds_it != data.end())
    {
      data[value]->push_back(*it);
    }
    else
    {
      data[value] = new Dataset(mInputSize, mOutputSize);
      data[value]->push_back(*it);
    }
  }
  return data;
}

vector<Dataset*> Dataset::SplitDataset(unsigned int parts)
{
  unsigned int old_size = this->size();
  unsigned int size = old_size / parts;
  vector<Dataset*> datasets;
  for (unsigned int i = 0; i < parts; i++)
  {
    Dataset* ds = new Dataset(mInputSize, mOutputSize);
    unsigned int k = i * size;
    for (unsigned int j = 0; j < size; j++)
    {
      ds->push_back((*this)[k + j]);
    }
    datasets.push_back(ds);
  }
  return datasets;
}

Dataset* Dataset::ExtractNewDataset(set<unsigned int> inputs, set<unsigned int> outputs)
{
  unsigned int input_size = inputs.size();
  unsigned int output_size = outputs.size();
  unsigned int i = 0;
  ;
  Dataset* ds = new Dataset(input_size, output_size);
  Dataset::iterator it;
  for (it = this->begin(); it != this->end(); ++it)
  {
    i = 0;
    Tuple* ti = new Tuple(input_size);
    set<unsigned int>::iterator in_it;
    for (in_it = inputs.begin(); in_it != inputs.end(); ++in_it)
    {
      if (*in_it < mInputSize)
      {
        (*ti)[i] = (*it)->GetInput(*in_it);
      }
      else
      {
        (*ti)[i] = (*it)->GetOutput((*in_it) - mInputSize);
      }
      i++;
    }
    i = 0;
    Tuple* to = new Tuple(output_size);
    set<unsigned int>::iterator out_it;
    for (out_it = outputs.begin(); out_it != outputs.end(); ++out_it)
    {
      if (*out_it < mInputSize)
      {
        (*to)[i] = (*it)->GetInput(*out_it);
      }
      else
      {
        (*to)[i] = (*it)->GetOutput((*out_it) - mInputSize);
      }
      i++;
    }
    ds->AddSample(ti, to);
  }
  return ds;
}

Dataset* Dataset::NormalizeMinMax(vector<pair<float, float> >& input_parameters,
    vector<pair<float, float> >& output_parameters)
{
  //Initialization of min and max parameters
  for (unsigned int i = 0; i < mInputSize; i++)
  {
    input_parameters.push_back(
        pair<float, float>((*begin())->GetInput(i), (*begin())->GetInput(i)));
  }
  for (unsigned int i = 0; i < mOutputSize; i++)
  {
    output_parameters.push_back(
        pair<float, float>((*begin())->GetOutput(i), (*begin())->GetOutput(i)));
  }
  //Computing the min and max values in the db for each attribute
  Dataset::iterator it;
  for (it = begin(); it != end(); ++it)
  {
    for (unsigned int i = 0; i < mInputSize; i++)
    {
      float input_value = (*it)->GetInput(i);
      if (input_parameters[i].first > input_value)
      {
        input_parameters[i].first = input_value;
      }
      if (input_parameters[i].second < input_value)
      {
        input_parameters[i].second = input_value;
      }
    }
    for (unsigned int i = 0; i < mOutputSize; i++)
    {
      float output_value = (*it)->GetOutput(i);
      if (output_value != 123.456)
      {
        if (output_parameters[i].first > output_value)
        {
          output_parameters[i].first = output_value;
        }
        if (output_parameters[i].second < output_value)
        {
          output_parameters[i].second = output_value;
        }
      }
    }
  }
  //Creating the new dataset with normalized values
  Dataset* ds = new Dataset(mInputSize, mOutputSize);
  for (it = begin(); it != end(); ++it)
  {
    Tuple* ti = new Tuple(mInputSize);
    for (unsigned int i = 0; i < mInputSize; i++)
    {
      (*ti)[i] = 2 * ((*it)->GetInput(i) - input_parameters[i].first)
          / (input_parameters[i].second - input_parameters[i].first) - 1;
    }
    Tuple* to = new Tuple(mOutputSize);
    for (unsigned int i = 0; i < mOutputSize; i++)
    {
      (*to)[i] = 2 * ((*it)->GetOutput(i) - output_parameters[i].first)
          / (output_parameters[i].second - output_parameters[i].first) - 1;
    }
    ds->AddSample(ti, to);
  }

  return ds;
}

Dataset* Dataset::NormalizeOutputMinMax(vector<pair<float, float> >& output_parameters)
{
  //Initialization of min and max parameters
  for (unsigned int i = 0; i < mOutputSize; i++)
  {
    output_parameters.push_back(
        pair<float, float>((*begin())->GetOutput(i), (*begin())->GetOutput(i)));
  }
  //Computing the min and max values in the db for each attribute
  Dataset::iterator it;
  for (it = begin(); it != end(); ++it)
  {
    for (unsigned int i = 0; i < mOutputSize; i++)
    {
      float output_value = (*it)->GetOutput(i);
      if (output_value != 123.456)
      {
        if (output_parameters[i].first > output_value)
        {
          output_parameters[i].first = output_value;
        }
        if (output_parameters[i].second < output_value)
        {
          output_parameters[i].second = output_value;
        }
      }
    }
  }
  //Creating the new dataset with normalized values
  Dataset* ds = new Dataset(mInputSize, mOutputSize);
  for (it = begin(); it != end(); ++it)
  {
    Tuple* to = new Tuple(mOutputSize);
    for (unsigned int i = 0; i < mOutputSize; i++)
    {
      (*to)[i] = 2 * ((*it)->GetOutput(i) - output_parameters[i].first)
          / (output_parameters[i].second - output_parameters[i].first) - 1;
    }
    ds->AddSample((*it)->GetInputTuple(), to);
  }

  return ds;
}

float Dataset::Mean()
{
  if (mMean == 0.0 && mVariance == 0.0)
  {
    ComputeMeanVariance();
  }
  return mMean;
}

float Dataset::Variance()
{
  if (mMean == 0.0 && mVariance == 0.0)
  {
    ComputeMeanVariance();
  }
  return mVariance;
}

void Dataset::ComputeMeanVariance()
{
  if (this->size() == 0)
  {
    return;
  }
  float m2 = 0.0;
  Dataset::iterator it;
  for (it = this->begin(); it != this->end(); ++it)
  {
    float value = (*it)->GetOutput();
    mMean += value;
    m2 += value * value;
  }
  mMean /= this->size();
  m2 /= this->size();
  mVariance = m2 - mMean * mMean;
}

Sample* Dataset::GetNearestNeighbor(Tuple& input, MetricType metric)
{
  double min_distance = 0.0;
  double distance = 0.0;
  Sample* min_dist_sample = 0;
  Dataset::iterator it;
  if (metric == EUCLIDEAN || metric == MANHATTAN)
  {
    if (metric == EUCLIDEAN)
    {
      min_distance = input.Distance((*(this->begin()))->GetInputTuple(), mInputSize, L2);
    }
    else
    {
      min_distance = input.Distance((*(this->begin()))->GetInputTuple(), mInputSize, L1);
    }

    min_dist_sample = *(this->begin());
    for (it = this->begin(); it != this->end(); ++it)
    {
      if (metric == EUCLIDEAN)
      {
        distance = input.Distance((*it)->GetInputTuple(), mInputSize, L2);
      }
      else
      {
        distance = input.Distance((*it)->GetInputTuple(), mInputSize, L1);
      }

      if (distance < min_distance)
      {
        min_distance = distance;
        min_dist_sample = *it;
      }
    }
  }
  else if (metric == MAHALANOBIS)
  {
    /*gsl_matrix* m = GetGSLMatrix();
     gsl_matrix* m_trans = gsl_matrix_alloc(m->size1, m->size2);
     gsl_matrix* cov = GetCovarianceMatrix();
     gsl_linalg_cholesky_decomp(cov);
     gsl_linalg_cholesky_invert(cov);
     gsl_vector* mean = gsl_vector_alloc(m->size2);
     gsl_vector* x[m->size2];
     for (unsigned int i = 0; i < m->size2; i++)
     {
     x[i] = gsl_vector_alloc(m->size1);
     gsl_matrix_get_col(x[i], m, i);
     gsl_vector_set(mean, i, gsl_stats_mean(x[i]->data, 1, m->size1));
     gsl_vector_free(x[i]);
     }
     gsl_vector* v = input.GetGSLVector(m->size2);
     for (unsigned int i = 0; i < m->size1; i++)
     {
     gsl_vector* t = gsl_vector_alloc(m->size2);
     gsl_blas_dgemv(CblasNoTrans, 1.0, cov, v, 0.0, t);
     gsl_blas_ddot(v, t, &distance);
     gsl_vector_free(t);
     if (distance < min_distance || i == 0)
     {
     min_distance = distance;
     min_dist_sample = *it;
     }
     }
     gsl_matrix_free(m_trans);
     gsl_vector_free(v);*/
  }

  return min_dist_sample;
}

namespace PoliFitted
{

ofstream& operator<<(ofstream& out, Dataset& ds)
{
  unsigned int size = ds.size();
  unsigned int input_size = ds.GetInputSize();
  unsigned int output_size = ds.GetOutputSize();
  out << input_size << " " << output_size << endl;
  for (unsigned int i = 0; i < size; i++)
  {
    Sample* s = ds[i];
    for (unsigned int j = 0; j < input_size; j++)
    {
      out << s->GetInput(j) << " ";
    }
    for (unsigned int j = 0; j < output_size; j++)
    {
      out << s->GetOutput(j) << " ";
    }
    out << endl;
  }

  return out;
}

ifstream& operator>>(ifstream& in, Dataset& ds)
{
  unsigned int input_size, output_size;
  in >> input_size >> output_size;
  ds.SetInputSize(input_size);
  ds.SetOutputSize(output_size);
  while (1)
  {
    Sample* s = new Sample(new Tuple(input_size), new Tuple(output_size));
    for (unsigned int j = 0; j < input_size; j++)
    {
      in >> s->GetInput(j);
    }
    for (unsigned int j = 0; j < output_size; j++)
    {
      in >> s->GetOutput(j);
    }
    if (in.eof() || in.fail())
    {
      in.clear();
      s->ClearInput();
      delete s;
      break;
    }
    ds.push_back(s);
  }

  return in;
}

}
