/**************************************************************************
 *   File:                        kdtree.h                                 *
 *   Description:   Class for kd-trees                                     *
 *   Copyright (C) 2007 by  Walter Corno & Daniele Dell'Aglio              *
 ***************************************************************************
 *                                                                         *
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
#ifndef KDTREE_H
#define KDTREE_H

#include "Tree.h"
#include "Dataset.h"
#include <vector>
using namespace std;

//<< Sam
namespace PoliFitted
{

/**************************************************************************
 *   This class implements kd-tree algorithm.                              *
 *   KD-Trees (K-Dimensional Trees) are a particular type of regression    *
 *   trees, infact this class extends the regTree one.                     *
 *   In this method the regression tree is built from the training set by  *
 *   choosing the cut-point at the local median of the cut-direction so    *
 *   that the tree partitions the local training set into two subsets of   *
 *   the same cardinality. The cut-directions alternate from one node to   *
 *   the other: if the direction of cut is i j for the parent node, it is  *
 *   equal to i j+1 for the two children nodes if j+1 < n with n the number*
 *   of possible cut-directions and i1 otherwise. A node is a leaf (i.e.,  *
 *   is not partitioned) if the training sample corresponding to this node *
 *   contains less than nmin tuples. In this method the tree structure is  *
 *   independent of the output values of the training sample.              *
 ***************************************************************************/
class KDTree: public Tree
{
  public:

    /**
     * Basic constructor
     * @param nm nmin, the minimum number of tuples for splitting
     */
    KDTree(unsigned int input_size = 1, unsigned int output_size = 1, int nm = 2);

    /**
     * Empty destructor
     */
    virtual ~KDTree();

    /**
     * Set nmin
     * @param nmin the minimum number of inputs for splitting
     */
    void SetNMin(int nm);

    /**
     * Set nmin
     * @param nmin the minimum number of inputs for splitting
     */
    int GetNMin();

    /**
     * Builds an approximation model for the training set
     * @param  data The training set
     * @param   overwrite When several training steps are run on the same inputs,
     it can be more efficient to reuse some structures.
     This can be done by setting this parameter to false
     */
    virtual void Train(Dataset* data, bool overwrite = true, bool normalize = true);

    /**
     * Builds an approximation model for the training set
     * @param  data The training set
     */
    virtual void TrainSample(Dataset* data);

    /**
     * @return Tuple
     * @param  input The input data on which the model is evaluated
     */
    virtual void Evaluate(Tuple* input, Tuple& output);

    /**
     * @return Tuple
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
     *
     */
    virtual Dataset* GetSamplesRecursive(rtANode* r, Tuple& in, int ax);

    /**
     *
     */
    virtual Dataset* GetSamples(Tuple* input);

    /**
     * Get the root of the tree
     * @return a pointer to the root
     */
    rtANode* GetRoot();

  private:

    /**
     * This method traverses the tree to search the output belonging to a particular input
     * @param r the node to evaluate
     * @param in the input
     * @param ax the axis on which the node is splitted
     * @return the output value
     */
    float TraverseTree(rtANode* r, Tuple& in, int ax);

    /**
     * This method checks if all the inputs of a cut direction are constant
     * @param ex the vector containing the inputs
     * @param cutDir the cut direction
     * @return true if all the inputs are constant, false otherwise
     */
    bool FixedInput(Dataset* ds, int cutDir);

    /**
     * This method build the KD-Tree
     * @param ex the vector containing the training set
     * @param cutDir the current cut direction
     * @param store_sample allow to store samples into leaves
     * @return a pointer to the root
     */
    rtANode* BuildKDTree(Dataset* ds, int cutDir, bool store_sample = false);

    unsigned int mNMin;  // minimum number of tuples for splitting

};

inline void KDTree::SetNMin(int nm)
{
  mNMin = nm;
}

inline int KDTree::GetNMin()
{
  return mNMin;
}

}
#endif
