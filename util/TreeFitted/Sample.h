/************************************************************************
 TrainingSample.h.h - Copyright marcello

 Here you can write a license for your code, some comments or any other
 information you want to have in your generated code. To to this simply
 configure the "headings" directory in uml to point to a directory
 where you have your heading files.

 or you can just replace the contents of this file with your own.
 If you want to do this, this file is located at

 /usr/share/apps/umbrello/headings/heading.h

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
 The original location of this file is /home/marcello/Projects/fitted/Developing/Sample.h
 **************************************************************************/

#ifndef SAMPLE_H
#define SAMPLE_H

#include <set>

#include "Tuple.h"

//<< Sam
namespace PoliFitted
{

/**
 * Samples are data container composed by input and output tuples:
 * \f[ \mathcal{S} = \left\{T_{in},T_{out}\right\},\f]
 * where \f$T_{in}\f$ is an \f$n_i\f$-dimensionale tuple and \f$T_{out}\f$
 * is an \f$n_o\f$-dimensional tuple.
 *
 * For example, a sample may represent the evaluation of a function (output component)
 * in a given point (the input component).
 *
 * Notice that information about the input and output dimension are not stored internally. *
 *
 * @see Tuple
 */
class Sample
{
  public:

    // Constructors/Destructors
    //

    /**
     * Empty Constructor
     */
    Sample();

    /**
     * Constructor with parameters
     * @param input input data
     * @param output output data
     */
    Sample(Tuple* input, Tuple* output);

    /**
     * @brief Empty Destructor
     * Clear the output tuple
     */
    virtual ~Sample();

    /**
     * Clear the input tuple. Note that the
     * destructor does not clear the input tuple.
     * @brief Delete the input component
     */
    void ClearInput();

    /**
     * @brief Getter of the input tuple
     * @return the input component
     */
    Tuple* GetInputTuple();

    /**
     * @brief Getter of the output tuple
     * @return the output component
     */
    Tuple* GetOutputTuple();

    /**
     * Return a tuple containing components of the input tuple
     * of indexes specified in the input set. The elements of the input set \f$I\f$ must be valid
     * indexes: \f$\forall i \in I,\; i \in [0, n_i]\f$.
     *
     * Programs should never call this function with an argument \f$inputs\f$ that contains
     * elements out of range, since this causes undefined behavior.
     *
     * @brief Access elements
     * @param inputs a set of indexes
     * @return a tuple built up from the specified indexes (form input tuple)
     */
    Tuple* GetInputTuple(set<unsigned int> inputs);

    /**
     * Return a tuple containing components of the output tuple
     * of indexes specified in the given set. The elements of the output set \f$I\f$ must be valid
     * indexes: \f$\forall i \in I,\; i \in [0, n_o]\f$.
     *
     * Programs should never call this function with an argument \f$outputs\f$ that contains
     * elements out of range, since this causes undefined behavior.
     *
     * @brief Access elements
     * @param outputs a set of indexes
     * @return a tuple built up from the specified indexes (form output tuple)
     */
    Tuple* GetOutputTuple(set<unsigned int> outputs);

    /**
     * Return the value of the input tuple in the given position. Position
     * must be a valid position: \f$pos \in [0, n_i]\f$.
     *
     * Programs should never call this function with an argument \f$pos\f$ that is
     * out of range, since this causes undefined behavior.
     *
     * @brief Access elements
     * @param pos an index
     * @return the value of the input tuple in the given position
     */
    float& GetInput(unsigned int pos);

    /**
     * Return the value of the input tuple in the given position. Position
     * must be a valid position: \f$pos \in [0, n_o]\f$.
     *
     * Programs should never call this function with an argument \f$pos\f$ that is
     * out of range, since this causes undefined behavior.
     *
     * @brief Access elements
     * @param pos an index (default is 0)
     * @return the value of the output tuple in the given position
     */
    float& GetOutput(unsigned int pos = 0);

    /**
     * @brief Resize the output tuple
     * @param new_size the new size of the output tuple
     */
    void ResizeOutput(unsigned int new_size);

    /**
     * Clone the current instance filling the new tuple with the
     * first \f$k\f$ and \f$l\f$ components of the input and output tuples, respectively.
     *
     * @brief Clone object
     * @param input_size the number of input components to be copied (\f$k\f$)
     * @param output_size the number of output components to be copied (\f$l\f$)
     * @return a new tuple
     *
     * @see Tuple::Clone
     */
    Sample* Clone(unsigned int input_size, unsigned int output_size);

  protected:

  private:

    Tuple* mpInput;
    Tuple* mpOutput;

};

inline Tuple* Sample::GetInputTuple()
{
  return mpInput;
}

inline Tuple* Sample::GetOutputTuple()
{
  return mpOutput;
}

inline Tuple* Sample::GetInputTuple(set<unsigned int> inputs)
{
  Tuple* new_tuple = new Tuple(inputs.size());
  unsigned int i = 0;
  set<unsigned int>::iterator it;
  for (it = inputs.begin(); it != inputs.end(); ++it)
  {
    (*new_tuple)[i] = (*mpInput)[*it];
    i++;
  }
  return new_tuple;
}

inline Tuple* Sample::GetOutputTuple(set<unsigned int> outputs)
{
  Tuple* new_tuple = new Tuple(outputs.size());
  unsigned int i = 0;
  set<unsigned int>::iterator it;
  for (it = outputs.begin(); it != outputs.end(); ++it)
  {
    (*new_tuple)[i] = (*mpOutput)[*it];
    i++;
  }
  return new_tuple;
}

inline float& Sample::GetInput(unsigned int pos)
{
  return (*mpInput)[pos];
}

inline float& Sample::GetOutput(unsigned int pos)
{
  return (*mpOutput)[pos];
}

inline Sample* Sample::Clone(unsigned int input_size, unsigned int output_size)
{
  Sample* s = new Sample(mpInput->Clone(input_size), mpOutput->Clone(output_size));
  return s;
}

}

#endif // TRAININGSAMPLE_H
