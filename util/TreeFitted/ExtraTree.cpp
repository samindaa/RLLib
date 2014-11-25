/**************************************************************************
 *   File:                     extratree.cpp                               *
 *   Description:   Class for extra-trees                                  *
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
#include <cmath>
#include <iostream>
//#include <gsl/gsl_cdf.h>

#include "ExtraTree.h"
#include "rtnode.h"
//#inclute "state.h"

//define values to mark if an attribute can be selected or not
#define SELECTABLE 0
#define SELECTED 1
#define NOT_SELECTABLE -1

// #define FEATURE_PROPAGATION

#define SPLIT_UNIFORM
#define SPLIT_VARIANCE
// #define VAR_RED_CORR

//<< Sam
using namespace PoliFitted;

#ifdef SPLIT_ANALYSIS
map<int, string> rtLeaf::mPlotCutsF;
map<int, string> rtLeaf::mPlotCutsG;
map<int, string> rtLeaf::mPlotCutsS;
#endif

ExtraTree::ExtraTree(unsigned int input_size, unsigned int output_size, int k, int nmin,
    float score_th, LeafType leaf) :
    Tree("ExtraTree", input_size, output_size)
{
  root = NULL;
  mNumSplits = k;
  mNMin = nmin;
  mFeatureRelevance = 0;
  mScoreThreshold = score_th;
  mLeafType = leaf;
  mParameters << "K" << k << "Nmin" << nmin << "Sth" << mScoreThreshold;
}

ExtraTree::~ExtraTree()
{
  if (root != NULL)
  {
    delete root;
  }
  if (mFeatureRelevance != 0)
  {
    delete[] mFeatureRelevance;
  }
}

void ExtraTree::InitFeatureRanks()
{
  if (mFeatureRelevance == 0)
  {
    mFeatureRelevance = new float[mInputSize];
  }
  for (unsigned int i = 0; i < mInputSize; i++)
  {
    mFeatureRelevance[i] = 0.0;
  }
}

void ExtraTree::Evaluate(Tuple* input, Tuple& output)
{
  output[0] = TraverseTree(root, *input);
}

void ExtraTree::Evaluate(Tuple* input, float& output)
{
  output = TraverseTree(root, *input);
}

void ExtraTree::Train(Dataset* data, bool overwrite, bool normalize)
{
  if (!overwrite)
  {
    cerr << "Not implemented!" << endl;
  }
#ifdef SPLIT_ANALYSIS
  for (unsigned int i = 0; i < data->GetInputSize(); i++)
  {
    float min, max, tmp;
    //initialize min and max with the attribute value of the first observation
    min = data->at(0)->GetInput(i);
    max = min;
    //looking for min and max value of the dataset
    for (unsigned int c = 1; c < data->size(); c++)
    {
      tmp = data->at(c)->GetInput(i);
      if (tmp < min)
      {
        min = tmp;
      }
      else if (tmp > max)
      {
        max = tmp;
      }
    }
    char initf[100];
    sprintf(initf, "f%d(x) = 0  \n", i);
    rtLeaf::mPlotCutsF.insert(pair<int, string> (i, string(initf)));
    char inits[100];
    sprintf(inits, "s%d(x) = 0  \n", i);
    rtLeaf::mPlotCutsS.insert(pair<int, string> (i, string(inits)));
    char initg[100];
    sprintf(initg, "g%d(x) = 0  \nplot [%f:%f] f%d(x)/g%d(x)\nplot [%f:%f] s%d(x)/g%d(x)", i, min, max, i, i, min, max, i, i);
    rtLeaf::mPlotCutsG.insert(pair<int, string> (i, string(initg)));
  }
#endif
  root = BuildExtraTree(data);
#ifdef SPLIT_ANALYSIS
  ofstream out_plot("plot.txt");
  out_plot << "set multiplot layout " << 2 * (data->GetInputSize() / 4 + 1) << "," << data->GetInputSize() % 4 << " columnsfirst" << endl;
  for (unsigned int i = 0; i < data->GetInputSize(); i++)
  {
    out_plot << rtLeaf::mPlotCutsF[i] << endl;
    out_plot << rtLeaf::mPlotCutsS[i] << endl;
    out_plot << rtLeaf::mPlotCutsG[i] << endl;
  }
  out_plot << "unset multiplot" << endl;
  out_plot.close();
#endif
}

Regressor* ExtraTree::GetNewRegressor()
{
  return new ExtraTree(mInputSize, mOutputSize, mNumSplits, mNMin);
}

void ExtraTree::WriteOnStream(ofstream& out)
{
  out << mNumSplits << " " << mNMin << " " << mInputSize << " " << mOutputSize << " " << mLeafType
      << endl;
  out << *root;
}

void ExtraTree::ReadFromStream(ifstream& in)
{
  string type;
  int leaf_type;
  in >> mNumSplits >> mNMin >> mInputSize >> mOutputSize >> leaf_type;
  mLeafType = (LeafType) leaf_type;
  in >> type;
  if ("L" == type)
  {
    root = new rtLeaf();
  }
  else if ("LLI" == type)
  {
    root = new rtLeafLinearInterp();
  }
  else
  {
    root = new rtINode();
  }
  in >> *root;
}

/**************************************************************************
 *   This method recursively builds the tree.                              *
 ***************************************************************************/
rtANode* ExtraTree::BuildExtraTree(Dataset* ds)
{
  /*************** part 1 - END CONDITIONS ********************/
  int size = ds->size(); //size of dataset
  // END CONDITION 1: return a leaf if |ex| is less than nmin
  if (size < mNMin)
  {
// 		cout << "size = " << size << endl;
    if (size == 0)
    {
      return 0;    //EMPTYLEAF
    }
    else
    {
      if (mLeafType == CONSTANT)
      {
        return new rtLeaf(ds);
      }
      else
      {
        return new rtLeafLinearInterp(ds);
      }
    }
  }
  // END CONDITION 2: return a leaf if all output variables are equals
  bool eq = true;
  float checkOut = ds->at(0)->GetOutput();
  for (int i = 1; i < size && eq; i++)
  {
    if (fabs(checkOut - ds->at(i)->GetOutput()) > 1e-7/*some small value*/)
    {
      eq = false;
    }
  }
  if (eq)
  {
    if (mLeafType == CONSTANT)
    {
      return new rtLeaf(ds);
    }
    else
    {
      return new rtLeafLinearInterp(ds);
    }
  }

  int attnum = mInputSize; //number of attributes
  bool constant[attnum]; //indicates if and attribute is costant (true) or not (false)
  float inputs[attnum];
  int end = attnum; //number of true values in constant
  Sample* s0 = ds->at(0);

  //initialize the constant vector
  for (int c = 0; c < attnum; c++)
  {
    constant[c] = true;
    inputs[c] = s0->GetInput(c);
  }

  //check if the attributes are constant and build constant vector
  eq = true;
  for (int i = 1; i < size && end > 0; i++)
  {
    Sample* si = ds->at(i);
    for (int c = 0; c < attnum; c++)
    {
      if (constant[c] && inputs[c] != si->GetInput(c))
      {
        constant[c] = false;
        end--;
        eq = false;
      }
    }
  }

  // END CONDITION 3: return a leaf if all input variables are equals
  if (eq)
  {
    if (mLeafType == CONSTANT)
    {
      return new rtLeaf(ds);
    }
    else
    {
      return new rtLeafLinearInterp(ds);
    }
  }

  /************** part 2 - TREE GENERATIONS *******************/
  //now we have a vector (costant) that indicates if
  //an attribute is constant in every example;
  //selected will indicate if an attribute is selectable to be
  //splitted
  int selected[attnum];
  int selectable = 0;
  for (int c = 0; c < attnum; c++)
  {
    //it will avoid the selection of costant attributes
    if (constant[c] == true)
    {
      selected[c] = NOT_SELECTABLE;
    }
    else
    {
      selected[c] = SELECTABLE;
      selectable++;
    }
  }

  //if the number of selectable attributes is <= k, all of
  //them will be candidate to split, else they will randomly
  //selected
  unsigned int candidates_size = selectable <= mNumSplits ? selectable : mNumSplits;
  unsigned int candidates[candidates_size];
  unsigned int num_candidates = candidates_size;
  while (num_candidates > 0)
  {
    unsigned int r = (rand() % num_candidates);
    unsigned int sel_attr = 0;
    while (selected[sel_attr] != SELECTABLE || r > 0)
    {
      if (selected[sel_attr] == SELECTABLE)
      {
        r--;
      }
      sel_attr++;
    }
    candidates[num_candidates - 1] = sel_attr;
    selected[sel_attr] = SELECTED;
    num_candidates--;
  }
  //generate the first split
  int bestattribute = candidates[0]; //best attribute (indicated by number) found
  float bestsplit = PickRandomSplit(ds, candidates[0]); //best split value
  Dataset bestSl(mInputSize, mOutputSize), bestSr(mInputSize, mOutputSize); //best left and right partitions
  float bestscore; //score of the best split
  Partitionate(ds, &bestSl, &bestSr, candidates[0], bestsplit);
  bestscore = Score(ds, &bestSl, &bestSr);
  //generates remaining splits and overwrites the actual best if better one is found
  for (unsigned int c = 1; c < candidates_size; c++)
  {
    float split = PickRandomSplit(ds, candidates[c]);
    Dataset sl(mInputSize, mOutputSize), sr(mInputSize, mOutputSize);
    Partitionate(ds, &sl, &sr, candidates[c], split);
    float s = Score(ds, &sl, &sr);
    //check if a better split was found
    if (s > bestscore)
    {
      bestscore = s;
      bestsplit = split;
      bestSl = sl;
      bestSr = sr;
      bestattribute = candidates[c];
    }
  }
//    cout << "Best: " << bestattribute << " " << bestscore << " " << mScoreThreshold << endl;
  if (bestscore < mScoreThreshold)
  {
    if (mLeafType == CONSTANT)
    {
      return new rtLeaf(ds);
    }
    else
    {
      return new rtLeafLinearInterp(ds);
    }
  }
  else
  {
    if (mFeatureRelevance != NULL)
    {
      float variance_reduction = VarianceReduction(ds, &bestSl, &bestSr) * ds->size()
          * ds->Variance();
#ifdef FEATURE_PROPAGATION
      mSplittedAttributes.insert(bestattribute);
      mSplittedAttributesCount.insert(bestattribute);
      set<int>::iterator it;
      for (it = mSplittedAttributes.begin(); it != mSplittedAttributes.end(); ++it)
      {
        mFeatureRelevance[*it] += variance_reduction / (float)mSplittedAttributes.size();
      }
#else
      mFeatureRelevance[bestattribute] += variance_reduction;
#endif
    }

    //build the left and the right children
    rtANode* left = BuildExtraTree(&bestSl);
    rtANode* right = BuildExtraTree(&bestSr);

#ifdef FEATURE_PROPAGATION
    if (mFeatureRelevance != NULL)
    {
      mSplittedAttributesCount.erase(mSplittedAttributesCount.find(bestattribute));
      if (mSplittedAttributesCount.find(bestattribute) == mSplittedAttributesCount.end())
      {
        mSplittedAttributes.erase(bestattribute);
      }
    }
#endif
    //return the current node
    return new rtINode(bestattribute, bestsplit, left, right);
  }
}

float ExtraTree::PickRandomSplit(Dataset* ds, int attsplit)
{
#ifdef SPLIT_UNIFORM
  float min, max, tmp;
  //initialize min and max with the attribute value of the first observation
  min = ds->at(0)->GetInput(attsplit);
  max = min;
  //looking for min and max value of the dataset
  for (unsigned int c = 1; c < ds->size(); c++)
  {
    tmp = ds->at(c)->GetInput(attsplit);
    if (tmp < min)
    {
      min = tmp;
    }
    else if (tmp > max)
    {
      max = tmp;
    }
  }
  //return a value in (min, max]
  float n = (float) ((rand() % 99) + 1) / 100.0;
  return min + (max - min) * n;
#else
  unsigned int r = rand() % ds->size();
  float value = ds->at(r)->GetInput(attsplit);
  float previous = value, next = value;
  for (unsigned int c = 0; c < ds->size(); c++)
  {
    float tmp = ds->at(c)->GetInput(attsplit);
    if (tmp < value && tmp > previous)
    {
      previous = tmp;
    }
    else if (tmp > value && tmp < next)
    {
      next = tmp;
    }
    else if (previous == value && tmp < value)
    {
      previous = tmp;
    }
    else if (next == value && tmp > value)
    {
      next = tmp;
    }
  }
//   cout << "R = " << r << " out of " << ds->size() << endl;
  float n = (float)((rand() % 99) + 1) / 100.0;
  return previous + (next - previous) * n;
#endif
}

void ExtraTree::Partitionate(Dataset* ds, Dataset* left, Dataset* right, int attribute, float split)
{
  Sample* bound = 0;
  unsigned int size = ds->size();
  for (unsigned int i = 0; i < size; i++)
  {
    Sample* s = ds->at(i);
    float tmp = s->GetInput(attribute);
    //if attribute value is less than split value, the observation will be added to left partition, else it will
    //be added to the right one
    if (tmp < split)
    {
      (*left).push_back(s);
    }
    else if (tmp > split)
    {
      (*right).push_back(s);
    }
    else
    {
      bound = s;
    }
  }

  if (bound != 0)
  {
    if (left->size() < right->size())
    {
      (*left).push_back(bound);
    }
    else
    {
      (*right).push_back(bound);
    }
  }
}

float ExtraTree::VarianceReduction(Dataset* ds, Dataset* dsl, Dataset* dsr)
{
  // VARIANCE REDUCTION
  float corr_fact_dsl = 1.0, corr_fact_dsr = 1.0, corr_fact_ds = 1.0;
#ifdef VAR_RED_CORR
  if (dsl->size() > 1)
  {
    corr_fact_dsl = (float)(dsl->size() / (dsl->size() - 1));
    corr_fact_dsl *= corr_fact_dsl;
  }
  if (dsr->size() > 1)
  {
    corr_fact_dsr = (float)(dsr->size() / (dsr->size() - 1));
    corr_fact_dsr *= corr_fact_dsr;
  }
  corr_fact_ds = (float)(ds->size() / (ds->size() - 1));
  corr_fact_ds *= corr_fact_ds;
#endif
  if (ds->size() == 0 || ds->Variance() == 0.0)
  {
    return 0.0;
  }
  else
  {
    return 1
        - ((float) corr_fact_dsl * dsl->size() * dsl->Variance()
            + (float) corr_fact_dsr * dsr->size() * dsr->Variance())
            / ((float) corr_fact_ds * ds->size() * ds->Variance());
  }
}

/*float ExtraTree::ProbabilityDifferentMeans(Dataset* ds, Dataset* dsl, Dataset* dsr)
 {
 if (ds->size() == 0)
 return 0.0;
 float score = 0.0;
 // T-STUDENT
 float mean_diff = fabs(dsl->Mean() - dsr->Mean());
 //   if (dsl->size() == 0 || dsr->size() == 0)
 float size_dsl = (float) dsl->size() - 1.0;
 float size_dsr = (float) dsr->size() - 1.0;
 if (size_dsl < 1.0 && size_dsr < 1.0)
 {
 //     cout << "Score = 0.0 (one set empty)" << endl;
 return 1.0;
 }
 else if (size_dsl < 1.0)
 {
 score = 2 * (gsl_cdf_tdist_P(mean_diff / sqrtf(dsr->Variance() / size_dsr), size_dsr) - 0.5);
 if (score >= 1.0)
 {
 score += mean_diff / sqrtf(dsr->Variance() / size_dsr);
 }
 //     cout << "Score = " << score << " (empty set)" << endl;
 return score;
 }
 else if (size_dsr < 1.0)
 {
 score = 2 * (gsl_cdf_tdist_P(mean_diff / sqrtf(dsl->Variance() / size_dsl), size_dsl) - 0.5);
 if (score >= 1.0)
 {
 score += mean_diff / sqrtf(dsl->Variance() / size_dsl);
 }
 //     cout << "Score = " << score << " (empty set)" << endl;
 return score;
 }
 float dsl_mean_variance = dsl->Variance() / size_dsl;
 float dsr_mean_variance = dsr->Variance() / size_dsr;
 float mean_diff_variance = dsl_mean_variance + dsr_mean_variance;
 if (mean_diff_variance < 1e-6)
 {
 if (mean_diff > 1e-6)
 {
 //       cout << "Score = 1.0 (two constant sets)" << endl;
 return 1.0;
 }
 else
 {
 //       cout << "Score = 0.0 (splitting a constant)" << endl;
 return 0.0;
 }
 }

 float dof = mean_diff_variance * mean_diff_variance
 / (dsl_mean_variance * dsl_mean_variance / size_dsl
 + dsr_mean_variance * dsr_mean_variance / size_dsr);
 score = gsl_cdf_tdist_P(mean_diff / sqrtf(mean_diff_variance), dof);
 score = 2.0 * (score - 0.5);
 if (score >= 1.0)
 {
 score += mean_diff / sqrtf(mean_diff_variance);
 }
 //   cout << " mean diff = " << mean_diff << " sqrt mean diff variance = " << sqrtf(mean_diff_variance) << " dof = " << dof << endl;
 //   cout << "Score = " << score << endl;
 return score;
 }*/

float ExtraTree::Score(Dataset* ds, Dataset* dsl, Dataset* dsr)
{
#ifdef SPLIT_VARIANCE
  return VarianceReduction(ds, dsl, dsr);
#else
  return ProbabilityDifferentMeans(ds, dsl, dsr);
#endif
}

// This method iteratively traverses the tree looking for the output value, given a input value
float ExtraTree::TraverseTree(rtANode* r, Tuple& in)
{
  if (r == 0)
  {	//EMPTYLEAF
    return 0.0;
  }
  float* values = in.GetValues();
  rtANode* son = r->getChild(values);
  while (son != NULL)
  {
    r = son;
    son = r->getChild(values);
  }
  return r->getValue(&in);
}

float* ExtraTree::EvaluateFeatures()
{
  return mFeatureRelevance;
}

/*
 void ExtraTree::AnalyzeSplitCriteria(Dataset* ds)
 {
 ofstream out;
 out.open("asc.txt");
 for (unsigned int i = 0; i < ds->GetInputSize(); i++)
 {
 Dataset::iterator s;
 for (s = ds->begin(); s != ds->end(); ++s)
 {
 float split = (*s)->GetInput(i);
 Dataset sl(mInputSize, mOutputSize), sr(mInputSize, mOutputSize);
 Partitionate(ds, &sl, &sr, i, split);
 out << i << " " << split << " " << VarianceReduction(ds, &sl, &sr) << " "
 << ProbabilityDifferentMeans(ds, &sl, &sr) << endl;
 }
 out << endl << endl;
 }
 out.close();
 }
 */
