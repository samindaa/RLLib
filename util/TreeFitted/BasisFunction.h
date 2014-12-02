#ifndef BASISFUNCTION_H
#define BASISFUNCTION_H

#include "Tuple.h"

//<< Sam
namespace PoliFitted
{

/**
 * A basis function \f$ \phi : R^n \to R\f$
 * represents a generic scalar function with \f$n\f$-dimensional
 * domain.
 * @brief The BasisFunction class defines a transformation of
 * the input value.
 */
class BasisFunction
{
  public:
    BasisFunction();
    virtual ~BasisFunction();

    /**
     * @brief Evaluate the basis function in the given point
     * @param input the point where the function is evaluated
     * @return the scalar value of the function
     */
    virtual float Evaluate(Tuple* input) = 0;

    /**
     * @brief GetNewFunction returns a deep copy of the
     * current instance
     * @return a deep copy of the object
     */
    virtual BasisFunction* GetNewFunction() = 0;

    /**
     * @brief Write a complete description of the instance to
     * a stream.
     * @param out the output stream
     */
    virtual void WriteOnStream(ofstream& out) = 0;

    /**
     * @brief Read the description of the basis function from
     * a file and reset the internal state according to that.
     * This function is complementary to WriteOnStream
     * @param in the input stream
     */
    virtual void ReadFromStream(ifstream& in) = 0;

    /**
     * @brief Write the internal state to the stream.
     * @see WriteOnStream
     * @param out the output stream
     * @param bf an instance of basis function
     * @return the output stream
     */
    friend ofstream& operator<<(ofstream& out, PoliFitted::BasisFunction& bf);

    /**
     * @brief Read the internal stream from a stream
     * @see ReadFromStream
     * @param in the input stream
     * @param bf an instance of basis function
     * @return the input stream
     */
    friend ifstream& operator>>(ifstream& in, PoliFitted::BasisFunction& bf);
};

}

#endif // BASISFUNCTION_H
