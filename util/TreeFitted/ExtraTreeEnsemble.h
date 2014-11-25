/**************************************************************************
 *   File:                  extratreeensemble.h                            *
 *   Description:   Class of the Extremely randomized trees                *
 *   Copyright (C) 2007 by  Walter Corno & Daniele Dell'Aglio              *
 ***************************************************************************
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
#ifndef EXTRATREEENSEMBLE_H_
#define EXTRATREEENSEMBLE_H_

#include "Regressor.h"
#include "ExtraTree.h"
#include <thread>
#include <mutex>

//<< Sam
namespace PoliFitted
{

/**************************************************************************
 * This class implements the Extremely randomized trees                    *
 * The Extra-Trees are a method for classification and regression; it was  *
 * developed by Geurts, Ernst and Wehenkel. It builds an ensemble of trees *
 * randomly choosing every attribute and every cut-point.                  *
 ***************************************************************************/
class ExtraTreeEnsemble: public Regressor
{
  public:

    /**
     * The basic constructor
     * @param ex set of observations
     * @param m number of trees in the ensemble
     * @param k number of selectable attributes to be randomly picked
     * @param nmin minimum number of tuples in a leaf
     */
    ExtraTreeEnsemble(unsigned int input_size = 1, unsigned int output_size = 1, int m = 50, int k =
        5, int nmin = 2, float score_th = 0.0, LeafType leaf = CONSTANT);

    /**
     * Empty destructor
     */
    virtual ~ExtraTreeEnsemble();

    /**
     * Initialize the ExtraTreeEnsemble by clearing the internal structures
     */
    virtual void Initialize();

    /**
     * Set nmin
     * @param nmin the minimum number of inputs for splitting
     */
    void SetNMin(int nm);

    /**
     * Builds an approximation model for the training set with parallel threads
     * @param  data The training set
     * @param   overwrite When several training steps are run on the same inputs,
     it can be more efficient to reuse some structures.
     This can be done by setting this parameter to false
     */
    virtual void Train(Dataset* data, bool overwrite = true, bool normalize = true);

    /**
     * @return Tuple
     * @param  input The input data on which the model is evaluated
     */
    virtual void Evaluate(Tuple* input, Tuple& output);

    /**
     * @return Value
     * @param  input The input data on which the model is evaluated
     */
    virtual void Evaluate(Tuple* input, float& output);

    /**
     *
     */
    virtual Regressor* GetNewRegressor();

    /**
     *
     */
    virtual void WriteOnStream(ofstream& out);

    /**
     *
     */
    virtual void ReadFromStream(ifstream& in);

    /**
     * Initialize data structures for feature ranking
     */
    void InitFeatureRanks();

    /**
     *
     */
    multimap<float, unsigned int> EvaluateFeatures(float initial_variance,
        float min_threshold = 0.0);

  private:

    unsigned int mNumTrees; //number of trees in the ensemble
    unsigned int mNumSplits; //number of selectable attributes to be randomly picked
    unsigned int mNMin; //minimum number of tuples for splitting
    float mScoreThreshold;
    vector<ExtraTree*> mEnsemble; //the extra-trees ensemble
    LeafType mLeafType;
    float mSum;		//sum of evaluation (parallel thread)
    mutex mToken;	//token lock (parallel thread)

    /**
     * Single thread of train function
     * @param  index The extra-trees ensamble index
     * @param  data The training set
     */
    void workerTrain(unsigned int index, Dataset* data);

    /**
     * Single thread of evaluate function
     * @param  index The extra-trees ensamble index
     * @param  input The tuple
     */
    void workerEvaluateSingleOutput(unsigned int index, Tuple* input);

    /**
     * Single thread of evaluate function
     * @param  index The extra-trees ensamble index
     * @param  input The tuple
     */
    void workerEvaluateTuple(unsigned int index, Tuple* input);
};

}
#endif /*EXTRATREEENSEMBLE_H_*/
