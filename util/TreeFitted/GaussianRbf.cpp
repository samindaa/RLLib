#include "GaussianRbf.h"

#include <cmath>
#include <cassert>

using namespace std;

//<< Sam
using namespace PoliFitted;

GaussianRbf::GaussianRbf() :
    mDimension(0), mpMean(0), mScale(0)
{
}

GaussianRbf::GaussianRbf(unsigned int dimension, float mean[], float scale) :
    mDimension(dimension), mScale(scale)
{
  mpMean = new float[dimension];
  for (unsigned i = 0; i < dimension; ++i)
  {
    mpMean[i] = mean[i];
  }
}

GaussianRbf::~GaussianRbf()
{
  delete[] mpMean;
}

BasisFunction* GaussianRbf::GetNewFunction()
{
  return new GaussianRbf(mDimension, mpMean, mScale);
}

float GaussianRbf::Evaluate(Tuple* input)
{

  double normv = 0.0;
  for (unsigned i = 0; i < mDimension; ++i)
  {
    normv += ((*input)[i] - mpMean[i]) * ((*input)[i] - mpMean[i]);
  }
  double retv = -normv / (mScale);
  retv = exp(retv);
  return retv;
}

double GaussianRbf::max(double* lb, double* ub)
{
  return 1.0;
}

void GaussianRbf::WriteOnStream(ofstream& out)
{
  out << "GaussianRBF " << mDimension << endl;
  for (unsigned int i = 0; i < mDimension; i++)
  {
    out << mpMean[i] << " ";
  }
  out << mScale;
}

void GaussianRbf::ReadFromStream(ifstream& in)
{
  in >> mDimension;

  mpMean = new float[mDimension];
  float value;
  for (unsigned int i = 0; i < mDimension; i++)
  {
    in >> value;
    mpMean[i] = value;
  }
  in >> value;
  mScale = value;
}
