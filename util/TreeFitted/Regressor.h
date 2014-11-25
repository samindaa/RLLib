/************************************************************************
 Regressor.h.h - Copyright marcello

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
 The original location of this file is /home/marcello/Projects/fitted/Developing/Regressor.h
 **************************************************************************/

#ifndef REGRESSOR_H
#define REGRESSOR_H

#include <sstream>

#include "Dataset.h"

//<< Sam
namespace PoliFitted
{

class Regressor
{
  public:

    // Constructors/Destructors
    //

    /**
     * Empty Constructor
     */
    Regressor(string type, unsigned int input_size = 1, unsigned int output_size = 1);

    /**
     * Empty Destructor
     */
    virtual ~Regressor();

    /**
     * Initialize the regressor by clearing the internal structures
     */
    virtual void Initialize()
    {
    }
    ;

    /**
     * Builds an approximation model for the training set
     * @param   data The training set
     * @param   overwrite When several training steps are run on the same inputs,
     it can be more efficient to reuse some structures. This
     can be done by setting this parameter to false
     * @param  normalize True means that the dataset will be normalized
     */
    virtual void Train(Dataset* data, bool overwrite = true, bool normalize = true) = 0;

    /**
     * @return Tuple
     * @param  input The input data on which the model is evaluated
     */
    virtual void Evaluate(Tuple* input, Tuple& output) = 0;

    /**
     * @return Value
     * @param  input The input data on which the model is evaluated
     */
    virtual void Evaluate(Tuple* input, float& output) = 0;

    /**
     *
     */
    virtual Regressor* GetNewRegressor() = 0;

    /**
     *
     */
    virtual void PostWriteOnStream(ofstream& out);

    /**
     *
     */
    virtual void PostReadFromStream(ifstream& in);

    /**
     *
     */
    virtual void WriteOnStream(ofstream& out) = 0;

    /**
     *
     */
    virtual void ReadFromStream(ifstream& in) = 0;

    /**
     *
     */
    float ComputeTrainError(Dataset* data, NormType type = L2);

    /**
     *
     */
    map<string, float> ComputePerfMetrics(Dataset* data);

    /**
     *
     */
    Dataset* EvaluateOnDataset(Dataset* data);

    /**
     *
     */
    float ComputeResiduals(Dataset* data, set<unsigned int> inputs, set<unsigned int> outputs,
        set<unsigned int> outputs_residual);

    /**
     *
     */
    map<string, float> CrossValidate(Dataset* data, unsigned int num_folds = 1, bool log = false,
        string filename = "cross_validation.txt");

    /**
     *
     */
    string GetType();

    /**
     *
     */
    string GetParameters();

    /**
     *
     */
    void SetInputSize(unsigned int size);

    /**
     *
     */
    void SetType(string type);

    /**
     *
     */
    friend ofstream& operator<<(ofstream& out, PoliFitted::Regressor& r);

    /**
     *
     */
    friend ifstream& operator>>(ifstream& in, PoliFitted::Regressor& r);

  protected:
    string mType;
    stringstream mParameters;

    unsigned int mInputSize;
    unsigned int mOutputSize;

    bool mIsNormalized;
    vector<pair<float, float> > mInputParameters;
    vector<pair<float, float> > mOutputParameters;

    double* NormalizeInput(double* data);

    double* DenormalizeOutput(double* data);

    Tuple* NormalizeInput(Tuple* data);

    Tuple* DenormalizeOutput(Tuple* data);

  private:

};

inline string Regressor::GetType()
{
  return mType;
}

inline string Regressor::GetParameters()
{
  return mParameters.str();
}

inline void Regressor::SetType(string type)
{
  mType = type;
}

inline void Regressor::SetInputSize(unsigned int size)
{
  mInputSize = size;
}

}

#endif // REGRESSOR_H
