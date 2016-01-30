/************************************************************************
 Tuple.h.h - Copyright marcello

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
 The original location of this file is /home/marcello/Projects/fitted/Developing/Tuple.h
 **************************************************************************/

#ifndef TUPLE_H
#define TUPLE_H

#include <fstream>
#include <vector>
//#include <gsl/gsl_vector.h>

//<< Sam
typedef enum
{
  L1,
  L2,
  Linfty
} NormType;

using namespace std;

namespace PoliFitted
{

/**
 * Tuples are sequence containers representing arrays with additional functionalities.
 *
 * Just like arrays, tuples use contiguous storage locations for their elements, which
 * means that their elements can also be accessed using offsets on regular pointers to its elements,
 * and just as efficiently as in arrays.
 *
 * Like arrays, internally, tuples do not store information about the dimension that must be handled
 * manually from outside the class.
 */
class Tuple
{
  public:

    /**
     * Converts N vectors to a single tuple obtained by
     * concatenation of the vectors (double) in the given order
     * Vectors are received as sequence of const_iterator:
     * v1.begin, v1.end, v2.begin, v2.end, v3.begin ...
     *
     * @param count    final number of elements of the tuple.
     Sum of size of each vector
     * @param ...      sequence of const_iterator (begin,end,begin,end,...)
     * @return the reference of the new tuple
     */
    //static Tuple* CONVERT(unsigned int count, ...);

    // Constructors/Destructors
    //

    /**
     * Empty Constructor
     */
    Tuple();

    /**
     * Construct a tuple of specified size
     * @param size the size of the tuple
     */
    Tuple(unsigned int size);

    /**
     * Empty Destructor
     */
    ~Tuple();

    /**
     * Returns a reference to the element at position \f$pos\f$ in the tuple.
     *
     * Programs should never call this function with an argument \f$pos\f$ that is
     * out of range, since this causes undefined behavior.
     * @brief Access element
     * @param pos Position of an element in the container. Notice that the first element has a position of 0 (not 1).
     * @return The element at the specified position in the tuple.
     */
    float& operator[](const unsigned int pos);

    /**
     * Returns a reference to the element at position \f$pos\f$ in the tuple.
     *
     * The function does not automatically check whether \f$pos\f$ is within the
     * bounds of valid elements in the vector.
     * Programs should never call this function with an argument \f$pos\f$ that is
     * out of range, since this causes undefined behavior.
     *
     * @brief Access element
     * @param pos Position of an element in the container. Notice that the first element has a position of 0 (not 1).
     * @return The element at the specified position in the container.
     */
    float& at(const unsigned int pos);

    /**
     * Compute a distance between the current and a given tuple. In other words,
     * it computes the p-norm between tuples.
     * \a p must be one of: "L1", "L2", "Linfty". "L1" is the absolute distance,
     * "L2" is the Euclidean distance, and "Linfty" is the maximum norm.
     * The norm is computed taking into account the first \a size elements of
     * the current and given tuples.
     *
     * Programs should never call this function with an argument \f$size\f$ that is
     * out of range for one of the two tuples, since this causes undefined behavior.
     *
     * @brief Distance computation
     * @param t An other tuple.
     * @param size the number of elements to be considered.
     * @param type the norm type
     * @return the \a p-norm
     */
    float Distance(Tuple* t, unsigned int size, NormType type = L2);

    /**
     * Assigns new contents to the tuple, replacing its current contents, without modifying its size.
     *
     * Since the dimension of the tuple is not changed accordingly to the given array,
     * the behavior is undefined when \a size is greater than the current tuple dimension.
     * In this case it is necessary to call the complementary function Tuple::Resize.
     *
     * The tuple is filled starting from the first elements.
     *
     * @brief Assign elements
     * @param values Array storing the new elements.
     * @param size The size of the given array
     */
    void SetValues(float* values, unsigned int size);

    /**
     * Returns a direct pointer to the memory array used internally by the tuple to store its owned elements.
     *
     * Because elements in the tuple are guaranteed to be stored in contiguous storage locations in the same
     * order as represented by the tuple, the pointer retrieved can be offset to access any element in the array.
     * @brief Access data
     * @return A pointer to the first element in the array used internally by the tuple.
     */
    float* GetValues();

    /**
     * Returns a clone of the tuple by copying the first \a input_size elements.
     *
     * @brief Clone object
     * @param input_size the number of elements to be copied
     * @return A pointer to the new tuple.
     */
    Tuple* Clone(unsigned int input_size);

    /**
     * Resizes the container so that it contains \a new_size elements.
     *
     * If \a new_size is smaller than the current container size, the content is reduced to its first n elements,
     * removing those beyond (and destroying them).
     * If \a new_size is greater than the current container size, the content is expanded by inserting at the end
     * as many elements as needed to reach a size of \a new_size.
     * The value of the newly allocated portion is indeterminate.
     *
     * Notice that this function changes the actual content of the container by inserting or erasing elements
     * from it.
     *
     * @brief Change size
     * @param new_size New container size, expressed in number of elements.
     */
    void Resize(unsigned int new_size);

    template<class T>
    T* GetValues(unsigned int input_size);

    //gsl_vector* GetGSLVector(unsigned int input_size);

    std::vector<double> ToVector(unsigned int input_size);

  protected:

  private:

    float* mValues;

};

inline float& Tuple::at(const unsigned int pos)
{
  return mValues[pos];
}

inline float& Tuple::operator[](const unsigned int pos)
{
  return mValues[pos];
}

inline void Tuple::SetValues(float* values, unsigned int size)
{
  for (unsigned int i = 0; i < size; i++)
  {
    mValues[i] = values[i];
  }
}

inline float* Tuple::GetValues()
{
  return mValues;
}

template<class T>
inline T* Tuple::GetValues(unsigned int input_size)
{
  T* values = new T[input_size];
  for (unsigned int i = 0; i < input_size; i++)
  {
    values[i] = (T) mValues[i];
  }
  return values;
}

/*inline gsl_vector* Tuple::GetGSLVector(unsigned int input_size)
 {
 gsl_vector* values = gsl_vector_alloc(input_size);
 for (unsigned int i = 0; i < input_size; i++)
 {
 gsl_vector_set(values, i, mValues[i]);
 }
 return values;
 }*/

inline std::vector<double> Tuple::ToVector(unsigned int input_size)
{
  std::vector<double> values;
  for (unsigned int i = 0; i < input_size; i++)
  {
    values.push_back(mValues[i]);
  }
  return values;
}

}

#endif // TUPLE_H
