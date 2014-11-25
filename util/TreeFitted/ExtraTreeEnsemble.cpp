/**************************************************************************
 *   File:                  extratreeensemble.cpp                          *
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
#define MULTI_THREAD_FLAG false
#define THREADS 8 //threads >= 1

#include <vector>
#include <iostream>
#include <unistd.h>

#include "ExtraTreeEnsemble.h"

//<< Sam
using namespace PoliFitted;

ExtraTreeEnsemble::ExtraTreeEnsemble(unsigned int input_size, unsigned int output_size, int m,
    int k, int nmin, float score_th, LeafType leaf) :
    Regressor("ExtraTrees", input_size, output_size)
{
  mNumTrees = m;
  mNumSplits = k;
  mNMin = nmin;
  mScoreThreshold = score_th;
  mLeafType = leaf;
  mSum = 0;
  for (unsigned int i = 0; i < mNumTrees; i++)
  {
    mEnsemble.push_back(
        new ExtraTree(mInputSize, mOutputSize, mNumSplits, mNMin, mScoreThreshold, mLeafType));
  }
  mParameters << "M" << m << "K" << k << "Nmin" << nmin << "Sth" << mScoreThreshold;
}

ExtraTreeEnsemble::~ExtraTreeEnsemble()
{
  for (unsigned int i = 0; i < mEnsemble.size(); i++)
  {
    delete mEnsemble.at(i);
  }
}

void ExtraTreeEnsemble::Initialize()
{
  for (unsigned int i = 0; i < mEnsemble.size(); i++)
  {
    delete mEnsemble.at(i);
  }
  mEnsemble.clear();
  for (unsigned int i = 0; i < mNumTrees; i++)
  {
    mEnsemble.push_back(
        new ExtraTree(mInputSize, mOutputSize, mNumSplits, mNMin, mScoreThreshold, mLeafType));
  }
}

void ExtraTreeEnsemble::Evaluate(Tuple* input, Tuple& output)
{
  if (mEnsemble.size() != mNumTrees)
  {
    cout << mEnsemble.size() << " not equal to " << mNumTrees << endl;
    return;
  }
  unsigned int i;
  mSum = 0.0;
#if MULTI_THREAD_FLAG
  vector<thread*> workerThread;
  //return the average of the trees output
  for (i = 0; i < mEnsemble.size(); ++i)
  {
    workerThread.push_back(new thread(bind(&ExtraTreeEnsemble::workerEvaluateTuple, this, i, input)));
  }
  for (i = 0; i < workerThread.size(); i++)
  {
    workerThread[i]->join();
  }
  for (i = 0; i < workerThread.size(); i++)
  {
    delete workerThread[i];
  }
#else
  for (i = 0; i < mEnsemble.size(); ++i)
  {
    workerEvaluateSingleOutput(i, input);
  }
#endif
  output[0] = mSum / (float) mNumTrees;
}

void ExtraTreeEnsemble::Evaluate(Tuple* input, float& output)
{
  if (mEnsemble.size() != mNumTrees)
  {
    cout << mEnsemble.size() << " not equal to " << mNumTrees << endl;
    return;
  }
  unsigned int i;
  mSum = 0.0;
#if MULTI_THREAD_FLAG

  unsigned int count = THREADS - 1;
  vector<thread*> workerThread;
  //return the average of the trees output
  for (i = 0; i <= count; i++)
  { //launch all available threads
    workerThread.push_back(new thread(bind(&ExtraTreeEnsemble::workerEvaluateSingleOutput, this, i, input)));
  }
  for (i = 0; i <= count; i++)
  { //wait and launch other threads
    workerThread[i]->join();
    if (count < mEnsemble.size() - 1)
    {
      workerThread.push_back(new thread(bind(&ExtraTreeEnsemble::workerEvaluateSingleOutput, this, count + 1, input)));
      count++;
    }
  }
  /* NO threads limit
   for (i=0; i < mEnsemble.size(); ++i)
   {
   workerThread.push_back(new thread(bind(&ExtraTreeEnsemble::workerEvaluateSingleOutput,this,i,input)));
   }
   for (i = 0; i < workerThread.size(); i++)
   {
   workerThread[i]->join();
   }
   */
  for (i = 0; i < workerThread.size(); i++)
  {
    delete workerThread[i];
  }
#else
  unsigned int size = mEnsemble.size();
  for (i = 0; i < size; ++i)
  {
    workerEvaluateSingleOutput(i, input);
  }
#endif
  output = mSum / (float) mNumTrees;
}

void ExtraTreeEnsemble::workerEvaluateSingleOutput(unsigned int index, Tuple* input)
{
  float out;
  mEnsemble[index]->Evaluate(input, out);
#if MULTI_THREAD_FLAG
  boost::mutex::scoped_lock lock(mToken);
#endif
  mSum += out;
}

void ExtraTreeEnsemble::workerEvaluateTuple(unsigned int index, Tuple* input)
{
  Tuple out;
  mEnsemble[index]->Evaluate(input, out);
#if MULTI_THREAD_FLAG
  boost::mutex::scoped_lock lock(mToken);
#endif
  mSum += out[0];
}

void ExtraTreeEnsemble::Train(Dataset* data, bool overwrite, bool normalize)
{
  unsigned int i;
#if MULTI_THREAD_FLAG

  unsigned int count = THREADS - 1;
  vector<thread*> workerThread;
  //return the average of the trees output
  for (i = 0; i <= count; i++)
  { //launch all available threads
    workerThread.push_back(new thread(bind(&ExtraTreeEnsemble::workerTrain, this, i, data)));
  }
  for (i = 0; i <= count; i++)
  { //wait and launch other threads
    workerThread[i]->join();
    if (count < mEnsemble.size() - 1)
    {
      workerThread.push_back(new thread(bind(&ExtraTreeEnsemble::workerTrain, this, count + 1, data)));
      count++;
    }
  }

  /*
   vector<thread*> workerThread;

   for (i = 0; i < mNumTrees; i++)
   {
   workerThread.push_back(new thread(bind(&ExtraTreeEnsemble::workerTrain,this,i,data)));
   }
   for (i = 0; i < workerThread.size(); i++)
   {
   workerThread[i]->join();
   }
   */
  for (i = 0; i < workerThread.size(); i++)
  {
    delete workerThread[i];
  }
#else
  if (!overwrite)
  {
    cerr << "Not implemented!" << endl;
  }
  for (i = 0; i < mEnsemble.size(); ++i)
  {
    workerTrain(i, data);
  }
#endif
}

void ExtraTreeEnsemble::workerTrain(unsigned int index, Dataset* data)
{
  mEnsemble[index]->Train(data);
}

Regressor* ExtraTreeEnsemble::GetNewRegressor()
{
  return new ExtraTreeEnsemble(mInputSize, mOutputSize, mNumTrees, mNumSplits, mNMin,
      mScoreThreshold, mLeafType);
}

void ExtraTreeEnsemble::WriteOnStream(ofstream& out)
{
  out << mNumTrees << " " << mNumSplits << " " << mNMin << " " << mInputSize << " " << mOutputSize
      << " " << mScoreThreshold << " " << mLeafType << endl;
  for (unsigned int i = 0; i < mNumTrees; i++)
  {
    out << *mEnsemble[i] << endl;
  }
}

void ExtraTreeEnsemble::ReadFromStream(ifstream& in)
{
  string type;
  int leaf_type;
  in >> mNumTrees >> mNumSplits >> mNMin >> mInputSize >> mOutputSize >> mScoreThreshold
      >> leaf_type;
//  in >> mNumTrees >> mNumSplits >> mNMin >> mInputSize >> mOutputSize >> leaf_type;
  mLeafType = (LeafType) leaf_type;
  //   in >> type;
  //   ExtraTree* tree = new ExtraTree(mInputSize,mOutputSize,mNumSplits,mNMin);
  for (unsigned int i = 0; i < mEnsemble.size(); i++)
  {
    delete mEnsemble.at(i);
  }
  mEnsemble.clear();
  for (unsigned int i = 0; i < mNumTrees; i++)
  {
    mEnsemble.push_back(new ExtraTree(mInputSize, mOutputSize, mNumSplits, mNMin));
    in >> *mEnsemble[i];
    //     mEnsemble.push_back(tree);
  }
}

void ExtraTreeEnsemble::InitFeatureRanks()
{
  vector<ExtraTree*>::iterator it;
  for (it = mEnsemble.begin(); it != mEnsemble.end(); it++)
  {
    (*it)->InitFeatureRanks();
  }
}

multimap<float, unsigned int> ExtraTreeEnsemble::EvaluateFeatures(float initial_variance,
    float min_threshold)
{
  float* features = 0;  // = new int(mInputSize);
  float* avg_feat = new float[mInputSize];
  for (unsigned int i = 0; i < mInputSize; i++)
  {
    avg_feat[i] = 0.0;
  }

  cout << "Absolute %\tRelative %\tFeature\tVarianceReduction" << endl;

  vector<ExtraTree*>::iterator it;
  for (it = mEnsemble.begin(); it != mEnsemble.end(); it++)
  {
    features = (*it)->EvaluateFeatures();
    for (unsigned int i = 0; i < mInputSize; i++)
    {
      avg_feat[i] += features[i];
//       cout << i << "\t" << features[i] << endl;
    }
  }
//   cout << "Average result" << endl;
  multimap<float, unsigned int> ranking;
  float cum_sum = 0.0;
  for (unsigned int i = 0; i < mInputSize; i++)
  {
    avg_feat[i] /= (float) mEnsemble.size();
    cum_sum += avg_feat[i];
    ranking.insert(pair<float, unsigned int>(avg_feat[i], i));
  }
  delete[] avg_feat;
  if (cum_sum == 0.0)
  {
    cerr << "ERROR: Impossible to explain current output feature" << endl;
    cerr << "Try to solve the problem by increasing the number of trees" << endl;
    exit(0);
  }
  cum_sum /= 100;
  multimap<float, unsigned int>::iterator rank_it = ranking.end();
  do
  {
    --rank_it;
    cout << 100 * rank_it->first / initial_variance << " %\t" << rank_it->first / cum_sum << " %\t"
        << rank_it->second << "\t" << rank_it->first << endl;
  } while (rank_it != ranking.begin() && rank_it->first / cum_sum > min_threshold);

  return ranking;
}
