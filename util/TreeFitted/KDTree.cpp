/**************************************************************************
 *   File:                        kdtree.cpp                               *
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
#include "KDTree.h"
#include <algorithm>
#include <iostream>
#include <cmath>

#include "rtLeafSample.h"

//  precision used for doubles
#define THRESHOLD 0.00000001

using namespace std;

//<< Sam
using namespace PoliFitted;

KDTree::KDTree(unsigned int input_size, unsigned int output_size, int nm) :
    Tree("KDTree", input_size, output_size)
{
  mNMin = nm;
  mParameters << "Nmin" << nm;
}

KDTree::~KDTree()
{
  delete root;
}

void KDTree::Evaluate(Tuple* input, Tuple& output)
{
  if (root == NULL)
  {
    return;
  }
  output[0] = TraverseTree(root, *input, 0);
}

void KDTree::Evaluate(Tuple* input, float& output)
{
  if (root == NULL)
  {
    return;
  }
  output = TraverseTree(root, *input, 0);
}

void KDTree::Train(Dataset* data, bool overwrite, bool normalize)
{
  if (!overwrite)
  {
    cerr << "Not implemented!" << endl;
  }
  root = BuildKDTree(data, 0);
}

void KDTree::TrainSample(Dataset* data)
{
  root = BuildKDTree(data, 0, true);
}

Regressor* KDTree::GetNewRegressor()
{
  return new KDTree(mInputSize, mOutputSize, mNMin);
}

rtANode* KDTree::GetRoot()
{
  return root;
}

void KDTree::WriteOnStream(ofstream& out)
{
  out << mNMin << " " << mInputSize << " " << mOutputSize << endl;
  out << *root;
}

void KDTree::ReadFromStream(ifstream& in)
{
  string type;
  in >> mNMin >> mInputSize >> mOutputSize;
  in >> type;
  if ("Leaf" == type)
  {
    root = new rtLeaf();
  }
  else if ("LeafSample" == type)
  {
    root = new rtLeafSample();
  }
  else
  {
    root = new rtINode();
  }
  in >> *root;
}

/**************************************************************************
 *   This method recursively traverses the tree to find the output (leaf   *
 *   value) belongig to a particular input.                                *
 *   If the current node is an empty leaf returns a default value of       *
 *   output, if it is a leaf, but not empty, returns the leaf's value.     *
 *   Otherwise if it is an internal node, first compare the input with the *
 *   split value of the node, and then recall itself to explore the left or*
 *   the right child, using the next cut direction.                        *
 ***************************************************************************/
float KDTree::TraverseTree(rtANode* r, Tuple& in, int ax)
{
  if (r == 0)
  {	//EMPTYLEAF
    return 0.0;
  }
  if (r->isLeaf())
  {
    //FixMe:
    float ret = ((rtLeaf*) r)->getValue(&in);
    //float ret = dynamic_cast<rtLeaf*>(r)->getValue(&in);
    return ret;
  }
  rtINode* tmp = (rtINode*) r;
  //rtINode* tmp = dynamic_cast<rtINode*>(r);
  float split = tmp->getSplit();
  if (in[ax] < split)
  {
    return TraverseTree(tmp->getLeft(), in, (ax + 1) % mInputSize);
  }
  else
  {
    return TraverseTree(tmp->getRight(), in, (ax + 1) % mInputSize);
  }
}

/**************************************************************************
 *   This method checks if all the values of a cut directions are constant.*
 *   The method scans the input vector until it's finished or a value is   *
 *   not equal to the others.                                              *
 ***************************************************************************/
bool KDTree::FixedInput(Dataset* ds, int cutDir)
{
  if (ds->size() == 0)
  {
    return true;
  }
  float val = (ds->at(0))->GetInput(cutDir);
  for (unsigned int i = 1; i < ds->size(); i++)
  {
    if (fabs(val - ((ds->at(i))->GetInput(cutDir))) > THRESHOLD)
    {
      return false;
    }
  }
  return true;
}

/**************************************************************************
 *   This method recursively builds the tree.                              *
 *   At the beginning it checks if size of input is less than nmin because *
 *   in this case a leaf is created, if size == 0 an empty leaf is created.*
 *   Another case that produces a leaf is when all the input data for all  *
 *   the cut-directions are constant.                                      *
 *   If the previous cases don't match the method search a cut point in the*
 *   current cut direction, sorting the input along the cut direction and  *
 *   choosing the median as the cut point. Then split the input data in two*
 *   subsets and recall the method for each of them.                       *
 ***************************************************************************/
rtANode* KDTree::BuildKDTree(Dataset* ds, int cutDir, bool store_sample)
{
  unsigned int size = ds->size();
  /******************part 1: end conditions**********************/
  if (size < mNMin)
  { // if true -> leaf
    if (size == 0)
    { // if true -> empty leaf
      return 0;	//EMPTYLEAF
    }
    else
    {
      if (store_sample)
      {
        return new rtLeafSample(ds->Clone());
      }
      else
      {
        return new rtLeaf(ds);
      }
    }
  }
  // control if inputs are all constants
  int cutTmp = cutDir;
  bool equal = false;
  while (FixedInput(ds, cutTmp) && !equal)
  {
    cutTmp = (cutTmp + 1) % ds->GetInputSize();
    if (cutTmp == cutDir)
    {
      equal = true;
    }
  }
  // if constants create a leaf
  if (equal)
  {
    if (store_sample)
    {
      return new rtLeafSample(ds->Clone());
    }
    else
    {
      return new rtLeaf(ds);
    }
  }
  /*****************part 2: generate the tree***************/
  //  begin operations to split the training set
  Dataset lowEx(ds->GetInputSize(), ds->GetOutputSize());
  Dataset highEx(ds->GetInputSize(), ds->GetOutputSize());
  float cutPoint;
  vector<float> tmp;
  for (unsigned int i = 0; i < size; i++)
  {
    tmp.push_back(ds->at(i)->GetInput(cutDir));
  }
  sort(tmp.begin(), tmp.end());
  vector<float> tmp1;
  float prec = tmp.at(0);
  tmp1.push_back(tmp.at(0)); //tmp1 is a vector of inputs without duplicates
  for (unsigned int i = 1; i < tmp.size(); i++)
  {
    if (tmp.at(i) != prec)
    {
      tmp1.push_back(tmp.at(i));
      prec = tmp.at(i);
    }
  }
  //  pick the cutpoint as the median of the vector without duplicates
  cutPoint = tmp1.at(tmp1.size() / 2);
  // split inputs in two subsets
  for (unsigned int i = 0; i < size; i++)
  {
    float tmp = ds->at(i)->GetInput(cutDir);
    if (tmp < cutPoint)
    {
      lowEx.push_back(ds->at(i));
    }
    else
    {
      highEx.push_back(ds->at(i));
    }
  }
  // recall the method for left and right child
  int n = ds->GetInputSize();
  rtANode* left = BuildKDTree(&lowEx, (cutDir + 1) % n, store_sample);
  rtANode* right = BuildKDTree(&highEx, (cutDir + 1) % n, store_sample);
  // return the current node
  return new rtINode(cutDir, cutPoint, left, right);

}

/**/

Dataset* KDTree::GetSamples(Tuple* input)
{
  if (root == NULL)
  {
    return NULL;
  }
  return GetSamplesRecursive(root, *input, 0);
}

/**************************************************************************
 *   This method recursively traverses the tree to find the output (leaf   *
 *   samples) belongig to a particular input.                              *
 *   If the current node is an empty leaf returns a default value of       *
 *   output, if it is a leaf, but not empty, returns the leaf's value.     *
 *   Otherwise if it is an internal node, first compare the input with the *
 *   split value of the node, and then recall itself to explore the left or*
 *   the right child, using the next cut direction.                        *
 ***************************************************************************/
Dataset* KDTree::GetSamplesRecursive(rtANode* r, Tuple& in, int ax)
{
//problema nel chiamare la funzione dall'esterno per il valore root
//ho impostato come evaluate (vedi sopra)

  if (r == 0)
  { //EMPTYLEAF
    return NULL;
  }
  if (r->isLeaf())
  {
    Dataset* ret = ((rtLeafSample*) r)->GetSample();
    //Dataset* ret = dynamic_cast<rtLeafSample*>(r)->GetSample();
    return ret;
  }

  rtINode* tmp = (rtINode*) r;
  //rtINode* tmp = dynamic_cast<rtINode*>(r);
  float split = tmp->getSplit();
  if (in[ax] < split)
  {
    return GetSamplesRecursive(tmp->getLeft(), in, (ax + 1) % mInputSize); //changed
  }
  else
  {
    return GetSamplesRecursive(tmp->getRight(), in, (ax + 1) % mInputSize); //changed
  }
}

