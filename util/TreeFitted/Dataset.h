/************************************************************************
 Dataset.h.h - Copyright marcello

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
 The original location of this file is /home/marcello/Projects/fitted/Developing/Dataset.h
 **************************************************************************/

#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <set>
#include <map>

//#include <gsl/gsl_matrix.h>

#include "Sample.h"

//<< Sam
namespace PoliFitted
{

typedef enum
{
  OVERWRITE,
  APPEND
} ModalityType;
typedef enum
{
  EUCLIDEAN,
  MANHATTAN,
  MAHALANOBIS
} MetricType;

/**
 * Represents an in-memory cache of numeric data.
 * A Dataset is nothing more than what its name implies: a set of data.
 * Since each data is a sample, the dataset can be split into input and
 * output data. In contrast with the Sample class, the
 * dataset stores the information related to the dimension of the input and output
 * tuples.
 *
 * Advanced features include iterator, random subsampling, normalization,
 * statistical data, import and export.
 *
 * @brief In-memory cache of numeric data.
 */
class Dataset: public vector<Sample*>
{
  public:

    // Constructors/Destructors
    //

    /**
     * Empty Constructor
     */
    Dataset(unsigned int input_size = 1, unsigned int output_size = 1);

    /**
     * Empty Destructor
     */
    virtual ~Dataset();

    /**
     * Destroys the content of the dataset but not the dataset itself.
     *
     * The input data of each sample are destroyed only if \a clear_input
     * is set to true.
     *
     * @brief Clear tuple
     * @param clear_input If true the input data are destrayed, false otherwise.
     */
    void Clear(bool clear_input = false);

    /**
     * @brief Set input data dimension
     * @param size The dimension of the input data.
     */
    void SetInputSize(unsigned int size);

    /**
     * @brief Set output data dimension
     * @param size The dimension of the output data.
     */
    void SetOutputSize(unsigned int size);

    /**
     * @brief Return the input dimension.
     * @return The dimension of the input component of each sample.
     */
    unsigned int GetInputSize();

    /**
     * @brief Return the output dimension.
     * @return The dimension of the output component of each sample.
     */
    unsigned int GetOutputSize();

    /**
     * Adds a new element at the end of the dataset, after its current last element.
     * This effectively increases the container size by one, which causes an automatic
     * allocation of a new samples initialized with the given input and output tuples.
     *
     * @brief Add element at the end.
     * @param input The input tuple to be added.
     * @param output The output tuple to be added.
     */
    void AddSample(Tuple* input, Tuple* output);

    /**
     * Returns a pointer to a newly created memory array storing
     * the elements of the dataset.
     * Data are stored linearly in the new array \f$A\f$. For any
     * sample in the dataset, input elements are stored before
     * output elements. As a consequence the dimension of the
     * array \f$A\f$ is \f$n_{el} * (input_dim + output_dim)\f$.
     * \f[ A = {input_i(1),\dots,input_i(m), output_i(1),\dots,output_i(n)}_{i=0}^{n_{el}-1}\f]
     *
     * @brief Get a copy of the elements.
     * @return A reference to the first element of an array storing
     * the dataset elements
     */
    //double* GetMatrix();
    /**
     * Returns a pointer to a newly created GSL matrix storing
     * the elements of the dataset. Each row represents a sample
     * of the dataset. For any
     * sample in the dataset, input elements are stored before
     * output elements. As a consequence the dimension of the
     * array \f$A\f$ is \f$n_{el} \times (input_dim + output_dim)\f$.
     * The element M(i,j) corresponds to the i-th sample of the dataset,
     * if j is less then input_dimension corresponds to the j-th element
     * of the input tuple. If j is greater or equal than the input_dimension,
     * M(i,j) corresponds to the (j-input_dimension) element of the output
     * tuple.
     *
     * @brief Get a copy of the elements.
     * @return A GSL matrix storing the dataset elements
     */
    //gsl_matrix* GetGSLMatrix();
    /**
     *
     */
    //gsl_matrix* GetCovarianceMatrix();
    /**
     * Clone the current instance filling the new dataset with
     * a clone of each sample. This function performs a deep copy.
     *
     * @brief Clone object
     * @return A newly created dataset.
     *
     * @see Sample::Clone
     */
    Dataset* Clone();

    /**
     * @brief Resize the output tuple of each sample
     * @param new_size the new size of the output tuple
     */
    void ResizeOutput(unsigned int new_size);

    /**
     *
     */
    Dataset* GetReducedDataset(unsigned int size, bool random = false);

    /**
     *
     */
    Dataset* GetReducedDataset(float proportion, bool random = false);

    /**
     *
     */
    void GetTrainAndTestDataset(unsigned int num_partitions, unsigned int partition,
        Dataset* train_ds, Dataset* test_ds);

    /**
     *
     */
    vector<Dataset*> SplitDataset(unsigned int parts);

    /**
     *
     */
    map<float, Dataset*> SplitByAttribute(unsigned int attribute);

    /**
     *
     */
    Dataset* ExtractNewDataset(set<unsigned int> inputs, set<unsigned int> outputs);

    /**
     *
     */
    Sample* GetNearestNeighbor(Tuple& input, MetricType metric = EUCLIDEAN);

    /**
     *
     */
    void Save(string filename, ModalityType modality = OVERWRITE);

    /**
     *
     */
    void Load(string filename);

    /**
     *
     */
    Dataset* NormalizeMinMax(vector<pair<float, float> >& input_parameters,
        vector<pair<float, float> >& output_parameters);

    /**
     *
     */
    Dataset* NormalizeOutputMinMax(vector<pair<float, float> >& output_parameters);

    /**
     * @brief Mean of the first output element.
     * @return The mean value of the first output element.
     */
    float Mean();

    /**
     * @brief Variance of the first output element.
     * @return The variance value of the first output element.
     */
    float Variance();

    /**
     * Insert a formatted representation of the dataset into the given output stream.
     *
     * @brief Insert formatted output
     * @param out Output stream object where characters are inserted.
     * @param ds Dataset to be printed out
     * @return The reference to the stream it receives
     */
    friend ofstream& operator<<(ofstream& out, PoliFitted::Dataset& ds);

    /**
     * Extract formatted input from the stream and fill the dataset
     * accordingly to that information.
     *
     * @brief Extract formatted input
     * @param in Input stream object from which characters are extracted.
     * @param ds where the extracted information are stored.
     * @return The reference to the stream it receives
     */
    friend ifstream& operator>>(ifstream& in, PoliFitted::Dataset& ds);

  protected:

  private:
    unsigned int mInputSize;
    unsigned int mOutputSize;
    float mMean;
    float mVariance;

    /**
     * @brief Compute mean and variance of the first output element.
     */
    void ComputeMeanVariance();

};

inline void Dataset::SetInputSize(unsigned int size)
{
  mInputSize = size;
}

inline void Dataset::SetOutputSize(unsigned int size)
{
  mOutputSize = size;
}

inline unsigned int Dataset::GetInputSize()
{
  return mInputSize;
}

inline unsigned int Dataset::GetOutputSize()
{
  return mOutputSize;
}

inline void Dataset::AddSample(Tuple* input, Tuple* output)
{
  push_back(new Sample(input, output));
}

inline void Dataset::Save(string filename, ModalityType modality)
{
  ofstream log_file;
  if (modality == OVERWRITE)
  {
    log_file.open(filename.c_str(), ios::out);
  }
  else
  {
    log_file.open(filename.c_str(), ios::out | ios::app);
  }
  if (log_file.is_open())
  {
    log_file << *(this);
  }
  log_file.close();
}

inline void Dataset::Load(string filename)
{
  ifstream log_file;
  log_file.open(filename.c_str(), ios::in);
  log_file >> *(this);
  log_file.close();
}

}

#endif // DATASET_H
