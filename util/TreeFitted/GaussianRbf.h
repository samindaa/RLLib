#ifndef GAUSSIANRBF_H
#define GAUSSIANRBF_H

#include <vector>

#include "BasisFunction.h"

//<< Sam
namespace PoliFitted
{

class GaussianRbf: public BasisFunction
{
  public:
    /**
     *
     */
    GaussianRbf();
    /**
     *
     */
    GaussianRbf(unsigned int dimension, float mean[], float scale);

    /**
     *
     */
    virtual ~GaussianRbf();

    /**
     *
     */
    virtual float Evaluate(Tuple* input);

    virtual float* getMean() const;

    virtual unsigned int dimension();

    /**
     *
     */
    virtual BasisFunction* GetNewFunction();

    /**
     *
     */
    virtual void WriteOnStream(std::ofstream &out);

    /**
     *
     */
    virtual void ReadFromStream(std::ifstream& in);

    /**
     * @brief max
     * @return
     */
    virtual double max(double* lb = 0, double* ub = 0);

  private:
    unsigned int mDimension;
    float* mpMean, mScale;
};

inline float* GaussianRbf::getMean() const
{
  return mpMean;
}

inline unsigned int GaussianRbf::dimension()
{
  return mDimension;
}

}
#endif // GAUSSIANRBF_H
