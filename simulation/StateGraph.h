/*
 * Copyright 2015 Saminda Abeyruwan (saminda@cs.miami.edu)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * StateGraph.h
 *
 *  Created on: Oct 25, 2013
 *      Author: sam
 */

#ifndef STATEGRAPH_H_
#define STATEGRAPH_H_

#include <iostream>
#include <vector>
#include <map>
#include <set>

// Matrix
#include "util/Eigen/Dense"
using Eigen::MatrixXd;

#include "Action.h"
#include "Vector.h"

namespace RLLib
{

  class GraphState
  {
    protected:
      std::map<int, GraphState*> transitions;
      Vector<double>* vectorRepresentation;

    public:
      std::string name;
      double reward;

      GraphState(const std::string& name, const double& reward) :
          vectorRepresentation(0), name(name), reward(reward)
      {
      }

      void connect(const Action<double>* action, GraphState* state)
      {
        transitions.insert(std::make_pair(action->id(), state));
      }

      GraphState* nextState(const Action<double>* action)
      {
        std::map<int, GraphState*>::iterator iter = transitions.find(action->id());
        return iter != transitions.end() ? iter->second : 0;
      }

      bool hasNextState() const
      {
        return !transitions.empty();
      }

      void setVectorRepresentation(Vector<double>* vectorRepresentation)
      {
        this->vectorRepresentation = vectorRepresentation;
      }

      Vector<double>* v() const
      {
        return vectorRepresentation;
      }
  };

  class FiniteStateGraph
  {
    public:
      class StepData
      {
        public:
          int stepTime;
          const GraphState* s_t;
          const Action<double>* a_t;
          const GraphState* s_tp1;
          double r_tp1;
          const Action<double>* a_tp1;

          StepData(const int& stepTime, const GraphState* s_t, const Action<double>* a_t,
              GraphState* s_tp1, const double& r_tp1, const Action<double>* a_tp1) :
              stepTime(stepTime), s_t(s_t), a_t(a_t), s_tp1(s_tp1), r_tp1(r_tp1), a_tp1(a_tp1)
          {
          }

          Vector<double>* v_t() const
          {
            return s_t->v();
          }

          Vector<double>* v_tp1() const
          {
            return s_tp1->v();
          }

          bool operator==(const StepData& that) const
          {
            return stepTime == that.stepTime && s_t->name == that.s_t->name
                && a_t->id() == that.a_t->id() && s_tp1->name == that.s_tp1->name
                && r_tp1 == that.r_tp1;
          }

          bool operator!=(const StepData& that) const
          {
            return !(*this == (that));
          }
      };

      GraphState* O;
      Action<double>* X;

    protected:
      int stepTime;
      GraphState* s_0;
      const Action<double>* a_t;
      GraphState* s_t;
      Policy<double>* acting;
      std::vector<GraphState*>* graphStates;

    public:
      FiniteStateGraph() :
          O(new GraphState("NULL", std::numeric_limits<int>::min())), //
          X(new Action<double>(std::numeric_limits<int>::max())), stepTime(-1), s_0(0), a_t(0), //
          s_t(0), acting(0), graphStates(0)
      {
        O->setVectorRepresentation(new PVector<double>(0));
      }

      virtual ~FiniteStateGraph()
      {
        delete O->v();
        delete O;
        delete X;
      }

      void setInitialState(GraphState* s_0)
      {
        assert(this->s_0 == 0);
        assert(s_0 != 0);
        this->s_0 = s_0;
      }

      GraphState* initialState() const
      {
        return s_0;
      }

      void setPolicy(Policy<double>* acting)
      {
        this->acting = acting;
      }

      void setStates(std::vector<GraphState*>* graphStates)
      {
        this->graphStates = graphStates;
      }

      std::vector<GraphState*>* states() const
      {
        return graphStates;
      }

      GraphState* state(const Vector<double>* s)
      {
        return graphStates->at(s->getEntry(0));
      }

      Policy<double>* policy() const
      {
        return acting;
      }

      void initialize()
      {
        stepTime = -1;
        s_t = 0;
        a_t = 0;
      }

      StepData/*small object*/step()
      {
        stepTime += 1;
        GraphState* s_tm1 = s_t;
        if (s_tm1 == 0)
          s_tm1 = O;
        const Action<double>* a_tm1 = X;
        if (s_t == 0 || s_t == O)
          s_t = s_0;
        else
        {
          a_tm1 = a_t;
          s_t = s_tm1->nextState(a_tm1);
          if (s_t == 0)
            s_t = O;
        }
        //a_t = Policies.decide(acting, s_t.v());
        a_t = acting->sampleAction(); // fixMe
        double r_t = s_t->reward;
        if (!s_t->hasNextState())
        {
          a_t = X;
          s_t = O;
        }
        return StepData(stepTime, s_tm1, a_tm1, s_t, r_t, a_t);
      }

      virtual double gamma() const=0;
      virtual const Vector<double>* expectedDiscountedSolution() const=0;
      virtual Actions<double>* getActions() const =0;

      static double distanceToSolution(const Vector<double>* solution, const Vector<double>* theta)
      {
        assert(solution->dimension() == theta->dimension());
        double maxValue = 0;
        for (int i = 0; i < solution->dimension(); i++)
          maxValue = std::max(maxValue, std::fabs(solution->getEntry(i) - theta->getEntry(i)));
        return maxValue;
      }
  };

  class LineProblem: public FiniteStateGraph
  {
    public:
      const double Gamma;
      Actions<double>* actions;
      Policy<double>* acting;
      std::vector<GraphState*>* states;
      DenseVector<double>* solution;
      const Action<double>* Move;

      GraphState* A;
      GraphState* B;
      GraphState* C;
      GraphState* D;

      LineProblem() :
          FiniteStateGraph(), Gamma(0.9), actions(new ActionArray<double>(1)), //
          acting(new SingleActionPolicy<double>(actions)), states(new std::vector<GraphState*>()), //
          solution(new PVector<double>(3)), Move(actions->getEntry(0))
      {
        A = new GraphState("A", 0.0);
        B = new GraphState("B", 0.0);
        C = new GraphState("C", 0.0);
        D = new GraphState("D", 1.0);

        states->push_back(A);
        states->push_back(B);
        states->push_back(C);
        states->push_back(D);

        int stateIndex = 0;
        for (std::vector<GraphState*>::iterator i = states->begin(); i != states->end(); ++i)
        {
          DenseVector<double>* v = new PVector<double>(1);
          v->at(0) = stateIndex++;
          (*i)->setVectorRepresentation(v);
        }

        setPolicy(acting);
        setStates(states);
        setInitialState(A);

        solution->at(0) = ::pow(Gamma, 2);
        solution->at(1) = ::pow(Gamma, 1);
        solution->at(2) = ::pow(Gamma, 0);

        A->connect(Move, B);
        B->connect(Move, C);
        C->connect(Move, D);

      }

      virtual ~LineProblem()
      {
        delete actions;
        delete acting;
        for (std::vector<GraphState*>::iterator i = states->begin(); i != states->end(); ++i)
        {
          GraphState* state = *i;
          delete state->v();
          delete state;
        }
        delete states;
        delete solution;
      }

      double gamma() const
      {
        return Gamma;
      }

      const Vector<double>* expectedDiscountedSolution() const
      {
        return solution;
      }

      Actions<double>* getActions() const
      {
        return actions;
      }

  };

  class RandomWalk: public FiniteStateGraph
  {
    private:
      Random<double>* random;

    public:
      const double Gamma;
      Actions<double>* actions;
      std::vector<GraphState*>* states;
      DenseVector<double>* solution;
      const Action<double>* Left;
      const Action<double>* Right;

      GraphState* TL;
      GraphState* A;
      GraphState* B;
      GraphState* C;
      GraphState* D;
      GraphState* E;
      GraphState* TR;

      Policy<double>* acting;

      RandomWalk(Random<double>* random) :
          FiniteStateGraph(), random(random), Gamma(0.9), actions(new ActionArray<double>(2)), //
          states(new std::vector<GraphState*>()), solution(new PVector<double>(5)), //
          Left(actions->getEntry(0)), Right(actions->getEntry(1))
      {
        TL = new GraphState("TL", 0.0);
        A = new GraphState("A", 0.0);
        B = new GraphState("B", 0.0);
        C = new GraphState("C", 0.0);
        D = new GraphState("D", 0.0);
        E = new GraphState("E", 0.0);
        TR = new GraphState("TR", 1.0);

        states->push_back(TL);
        states->push_back(A);
        states->push_back(B);
        states->push_back(C);
        states->push_back(D);
        states->push_back(E);
        states->push_back(TR);

        PVector<double> distribution(actions->dimension());
        for (int i = 0; i < distribution.dimension(); i++)
          distribution[i] = 1.0 / actions->dimension();

        acting = new ConstantPolicy<double>(random, actions, &distribution);

        int stateIndex = 0;
        for (std::vector<GraphState*>::iterator i = states->begin(); i != states->end(); ++i)
        {
          DenseVector<double>* v = new PVector<double>(1);
          v->at(0) = stateIndex++;
          (*i)->setVectorRepresentation(v);
        }

        setPolicy(acting);
        setStates(states);
        setInitialState(C);

        solution->at(0) = 0.056;
        solution->at(1) = 0.140;
        solution->at(2) = 0.258;
        solution->at(3) = 0.431;
        solution->at(4) = 0.644;

        A->connect(Left, TL);
        A->connect(Right, B);

        B->connect(Left, A);
        B->connect(Right, C);

        C->connect(Left, B);
        C->connect(Right, D);

        D->connect(Left, C);
        D->connect(Right, E);

        E->connect(Left, D);
        E->connect(Right, TR);

      }

      virtual ~RandomWalk()
      {
        delete actions;
        //delete acting; // FixMe
        for (std::vector<GraphState*>::iterator i = states->begin(); i != states->end(); ++i)
        {
          GraphState* state = *i;
          delete state->v();
          delete state;
        }
        states->clear();
        delete states;
        delete solution;
      }

      double gamma() const
      {
        return Gamma;
      }

      const Vector<double>* expectedDiscountedSolution() const
      {
        return solution;
      }

      Actions<double>* getActions() const
      {
        return actions;
      }

      void enableOnlyLeftPolicy()
      {
        actions->erase(Right->id());
        if (acting)
          delete acting;
        acting = new SingleActionPolicy<double>(actions);
      }

      void enableOnlyRightPolicy()
      {
        actions->erase(Left->id());
        if (acting)
          delete acting;
        acting = new SingleActionPolicy<double>(actions);
      }

      static Policy<double>* newPolicy(Random<double>* random, Actions<double>* actionList,
          const double& leftProbability)
      {
        PVector<double> distribution(actionList->dimension());
        distribution[0] = leftProbability;
        distribution[1] = 1.0f - leftProbability;
        return new ConstantPolicy<double>(random, actionList, &distribution);
      }
  };

  class RandomWalk2: public FiniteStateGraph
  {
    private:
      Random<double>* random;

    public:
      const double Gamma;
      const double pRight;
      Actions<double>* actions;
      std::vector<GraphState*>* states;
      DenseVector<double>* solution;
      const Action<double>* Left;
      const Action<double>* Right;
      Policy<double>* acting;

      RandomWalk2(Random<double>* random) :
          FiniteStateGraph(), random(random), Gamma(0.99), pRight(0.9), //
          actions(new ActionArray<double>(2)), states(new std::vector<GraphState*>()), //
          solution(new PVector<double>(10)), Left(actions->getEntry(0)), Right(actions->getEntry(1))
      {
        for (int i = solution->dimension(); i >= 0; --i)
        {
          std::stringstream ss;
          ss << i;
          states->push_back(new GraphState(ss.str(), (i == 0) ? 1.0f : 0.0f));
        }

        PVector<double> distribution(actions->dimension());
        distribution[0] = 1.0f - pRight;
        distribution[1] = pRight;

        acting = new ConstantPolicy<double>(random, actions, &distribution);

        int stateIndex = 0;
        for (std::vector<GraphState*>::iterator i = states->begin(); i != states->end(); ++i)
        {
          DenseVector<double>* v = new PVector<double>(1);
          v->at(0) = stateIndex++;
          (*i)->setVectorRepresentation(v);
        }

        setPolicy(acting);
        setStates(states);
        setInitialState(*states->begin());

        (*states->begin())->connect(Left, *states->begin());
        (*states->begin())->connect(Right, *(states->begin() + 1));
        for (size_t i = 1; i < states->size() - 1; ++i)
        {
          GraphState* prevGraphState = states->at(i - 1);
          GraphState* currGraphState = states->at(i);
          GraphState* nextGraphState = states->at(i + 1);
          currGraphState->connect(Left, prevGraphState);
          currGraphState->connect(Right, nextGraphState);
        }

        PVector<double> tmpSolution(solution->dimension() + 1);

        // VI: V = TV
        for (int i = 0; i < 10000; ++i)
        {
          for (int s = 0; s < static_cast<int>(states->size()) - 1; ++s)
          {
            const int s_right = s + 1;
            const int s_left = std::max(0, s - 1);
            tmpSolution.at(s) = (1.0f - pRight)
                * (states->at(s)->nextState(Left)->reward + Gamma * tmpSolution.at(s_left))
                + (pRight)
                    * (states->at(s)->nextState(Right)->reward + Gamma * tmpSolution.at(s_right));
          }
        }

        for (int i = 0; i < solution->dimension(); ++i)
        {
          solution->at(i) = tmpSolution.at(i);
          std::cout << "i: " << solution->getEntry(i) << std::endl;
        }

      }

      virtual ~RandomWalk2()
      {
        delete actions;
        //delete acting; // FixMe
        for (std::vector<GraphState*>::iterator i = states->begin(); i != states->end(); ++i)
        {
          GraphState* state = *i;
          delete state->v();
          delete state;
        }
        states->clear();
        delete states;
        delete solution;
      }

      double gamma() const
      {
        return Gamma;
      }

      const Vector<double>* expectedDiscountedSolution() const
      {
        return solution;
      }

      Actions<double>* getActions() const
      {
        return actions;
      }
  };

  class FSGAgentState: public StateToStateAction<double>
  {
    protected:
      FiniteStateGraph* graph;
      Vector<double>* featureState;
      std::map<GraphState*, int>* stateIndexes;
      Representations<double>* phis;

    public:
      FSGAgentState(FiniteStateGraph* graph) :
          graph(graph), featureState(0), stateIndexes(new std::map<GraphState*, int>())
      {
        std::vector<GraphState*>* states = graph->states();
        for (std::vector<GraphState*>::iterator iter = states->begin(); iter != states->end();
            ++iter)
        {
          GraphState* gs = *iter;
          if (gs->hasNextState())
            stateIndexes->insert(std::make_pair(gs, stateIndexes->size()));
        }
        featureState = new PVector<double>(stateIndexes->size());
        phis = new Representations<double>(
            featureState->dimension() * graph->getActions()->dimension(), graph->getActions());
      }

      ~FSGAgentState()
      {
        delete featureState;
        delete stateIndexes;
        delete phis;
      }

      FiniteStateGraph::StepData step()
      {
        FiniteStateGraph::StepData stepData = graph->step();
        const Vector<double>* x = stepData.v_tp1();
        featureState->clear();
        if (!x->empty())
        {
          GraphState* sg = graph->state(x);
          if (sg->hasNextState())
            featureState->setEntry(stateIndexes->at(sg), 1.0); // fixme
        }
        return stepData;
      }

      Vector<double>* currentFeatureState() const
      {
        return featureState;
      }

      const Vector<double>* stateAction(const Vector<double>* x, const Action<double>* a)
      {
        GraphState* sg = graph->state(x);
        if (sg->hasNextState())
          phis->at(a)->setEntry(a->id() * stateIndexes->size() + stateIndexes->at(sg), 1);
        return phis->at(a);
      }

      const Representations<double>* stateActions(const Vector<double>* x)
      {
        phis->clear();
        if (x->empty())
          return phis;
        for (Actions<double>::const_iterator a = graph->getActions()->begin();
            a != graph->getActions()->end(); ++a)
          stateAction(x, *a);
        return phis;
      }

      const Actions<double>* getActions() const
      {
        return graph->getActions();
      }

      double vectorNorm() const
      {
        return 1.0;
      }

      int dimension() const
      {
        return stateIndexes->size();
      }

      const std::map<GraphState*, int>* getStateIndexes() const
      {
        return stateIndexes;
      }

    private:
      std::vector<GraphState*>* states() const
      {
        return graph->states();
      }

      int nbStates() const
      {
        return states()->size();
      }

      int nbNonAbsorbingState() const
      {
        return stateIndexes->size();
      }

      MatrixXd createPhi() const
      {
        MatrixXd result = MatrixXd::Zero(nbStates(), nbNonAbsorbingState());
        for (int i = 0; i < result.rows(); i++)
        {
          GraphState* sg = states()->at(i);
          if (sg->hasNextState())
            result(i, stateIndexes->at(sg)) = 1.0f;
        }
        return result;
      }

      void absorbingStatesSet(std::set<int>& endStates)
      {
        for (int i = 0; i < nbStates(); i++)
        {
          if (!states()->at(i)->hasNextState())
            endStates.insert(i);
        }
      }

      MatrixXd createTransitionProbablityMatrix(Policy<double>* policy)
      {
        MatrixXd p = MatrixXd::Zero(nbStates(), nbStates());
        for (int si = 0; si < nbStates(); si++)
        {
          GraphState* s_t = states()->at(si);
          policy->update(stateActions(s_t->v()));
          for (Actions<double>::const_iterator a = graph->getActions()->begin();
              a != graph->getActions()->end(); ++a)
          {
            double pa = policy->pi(*a);
            GraphState* s_tp1 = s_t->nextState(*a);
            if (s_tp1)
            {
              for (int sj = 0; sj < nbStates(); sj++)
              {
                if (states()->at(sj) == s_tp1)
                {
                  p(si, sj) = pa;
                  break;
                }
              }
            }
          }
        }
        std::set<int> endStates;
        absorbingStatesSet(endStates);
        for (std::set<int>::iterator absorbingState = endStates.begin();
            absorbingState != endStates.end(); ++absorbingState)
          p(*absorbingState, *absorbingState) = 1.0;

        return p;
      }

      MatrixXd removeColumnAndRow(const MatrixXd& m, const std::set<int>& absorbingState)
      {
        MatrixXd result = MatrixXd::Zero(nbNonAbsorbingState(), nbNonAbsorbingState());
        int ci = 0;
        for (int i = 0; i < m.rows(); i++)
        {
          if (absorbingState.find(i) != absorbingState.end())
            continue;
          int cj = 0;
          for (int j = 0; j < m.cols(); j++)
          {
            if (absorbingState.find(j) != absorbingState.end())
              continue;
            result(ci, cj) = m(i, j);
            ++cj;
          }
          ++ci;
        }
        return result;
      }

      MatrixXd createInitialStateDistribution()
      {
        MatrixXd result = MatrixXd::Zero(1, nbNonAbsorbingState());
        int ci = 0;
        for (int i = 0; i < nbStates(); i++)
        {
          GraphState* s = states()->at(i);
          if (!s->hasNextState())
            continue;
          if (s != graph->initialState())
            result(0, ci) = 0.0f;
          else
            result(0, ci) = 1.0f;
          ++ci;
        }
        return result;
      }

      MatrixXd createStateDistribution(const MatrixXd& p)
      {
        std::set<int> absorbingState;
        absorbingStatesSet(absorbingState);
        MatrixXd p_copy = removeColumnAndRow(p, absorbingState);
        MatrixXd id = MatrixXd::Identity(p_copy.rows(), p_copy.cols());
        MatrixXd inv = (id - p_copy).inverse();
        MatrixXd mu = createInitialStateDistribution();
        MatrixXd visits = mu * inv;
        double sum = 0.0f;
        for (int i = 0; i < visits.cols(); i++)
          sum += visits(0, i);
        visits = visits / sum;
        return visits.transpose();
      }

      MatrixXd createStateDistributionMatrix(const MatrixXd& d)
      {
        MatrixXd d_pi = MatrixXd::Zero(nbStates(), nbStates());
        int ci = 0;
        for (int i = 0; i < nbStates(); i++)
        {
          GraphState* s = states()->at(i);
          if (!s->hasNextState())
            continue;
          d_pi(i, i) = d(ci, 0);
          ++ci;
        }
        return d_pi;
      }

      MatrixXd computeIdMinusGammaLambdaP(const MatrixXd& p, const double& gamma,
          const double& lambda)
      {
        return (MatrixXd::Identity(p.cols(), p.cols()) - (p * (gamma * lambda))).inverse();
      }

      MatrixXd computePLambda(const MatrixXd& p, const double& gamma, const double& lambda)
      {
        MatrixXd inv = computeIdMinusGammaLambdaP(p, gamma, lambda);
        return (inv * p * (1.0f - lambda));
      }

      MatrixXd computeAverageReward(const MatrixXd& p)
      {
        MatrixXd result = MatrixXd::Zero(1, p.cols());
        for (int i = 0; i < nbStates(); i++)
        {
          if (!states()->at(i)->hasNextState())
            continue;
          double sum = 0.0f;
          for (int j = 0; j < nbStates(); j++)
            sum += p(i, j) * states()->at(j)->reward;
          result(0, i) = sum;
        }
        return result.transpose();
      }

      MatrixXd computeA(const MatrixXd& phi, const MatrixXd& d_pi, const double& gamma,
          const MatrixXd& pLambda)
      {
        return phi.transpose()
            * (d_pi * ((pLambda * gamma - MatrixXd::Identity(phi.rows(), phi.rows())) * phi));
      }

      MatrixXd computeB(const MatrixXd& phi, const MatrixXd& d_pi, const MatrixXd& p,
          const MatrixXd& r_bar, const double& gamma, const double& lambda)
      {
        MatrixXd inv = computeIdMinusGammaLambdaP(p, gamma, lambda);
        return phi.transpose() * (d_pi * (inv * r_bar));
      }

    public:
      const PVector<double> computeSolution(Policy<double>* policy, const double& gamma,
          const double& lambda)
      {
        MatrixXd phi = createPhi();
        MatrixXd p = createTransitionProbablityMatrix(policy);
        MatrixXd d = createStateDistribution(p);
        MatrixXd d_pi = createStateDistributionMatrix(d);
        MatrixXd p_lambda = computePLambda(p, gamma, lambda);
        MatrixXd r_bar = computeAverageReward(p);
        MatrixXd A = computeA(phi, d_pi, gamma, p_lambda);
        MatrixXd b = computeB(phi, d_pi, p, r_bar, gamma, lambda);
        MatrixXd minusAInverse = A.inverse() * -1.0f;
        MatrixXd result = minusAInverse * b;
        PVector<double> solution(result.rows());
        for (int i = 0; i < result.rows(); i++)
          solution[i] = result(i, 0);
        return solution;
      }
  };

}  // namespace RLLib

#endif /* STATEGRAPH_H_ */
