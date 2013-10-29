/*
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

#include "Action.h"
#include "Vector.h"

namespace RLLib
{
typedef DenseVector<double> RealVector;

class GraphState
{
  protected:
    std::map<int, GraphState*> transitions;
    RealVector* vectorRepresentation;

  public:
    std::string name;
    double reward;

    GraphState(const std::string& name, const double& reward) :
        vectorRepresentation(0), name(name), reward(reward)
    {
    }

    void connect(const Action* action, GraphState* state)
    {
      transitions.insert(std::make_pair(action->id(), state));
    }

    GraphState* nextState(const Action& action)
    {
      return transitions[action.id()];
    }

    bool hasNextState() const
    {
      return !transitions.empty();
    }

    void setVectorRepresentation(RealVector* vectorRepresentation)
    {
      this->vectorRepresentation = vectorRepresentation;
    }

    RealVector* v() const
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
        const Action* a_t;
        const GraphState* s_tp1;
        double r_tp1;
        const Action* a_tp1;

        StepData(const int& stepTime, const GraphState* s_t, const Action* a_t, GraphState* s_tp1,
            const double& r_tp1, const Action* a_tp1) :
            stepTime(stepTime), s_t(s_t), a_t(a_t), s_tp1(s_tp1), r_tp1(r_tp1), a_tp1(a_tp1)
        {
        }

        const RealVector& v_t() const
        {
          return *s_t->v();
        }

        const RealVector& v_tp1() const
        {
          return *s_tp1->v();
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
    Action* X;

  protected:
    int stepTime;
    GraphState* s_0;
    const Action* a_t;
    GraphState* s_t;
    Policy<double>* acting;
    std::vector<GraphState*>* graphStates;

  public:
    FiniteStateGraph() :
        O(new GraphState("NULL", std::numeric_limits<int>::min())), X(
            new Action(std::numeric_limits<int>::max())), stepTime(-1), s_0(0), a_t(0), s_t(0), acting(
            0), graphStates(0)
    {
      O->setVectorRepresentation(new RealVector(0));
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

    void setPolicy(Policy<double>* acting)
    {
      this->acting = acting;
    }

    void setStates(std::vector<GraphState*>* graphStates)
    {
      this->graphStates = graphStates;
    }

    std::vector<GraphState*>& states() const
    {
      return *graphStates;
    }

    GraphState* state(const RealVector& s)
    {
      return graphStates->at(s.at(0));
    }

    StepData/*small object*/step()
    {
      stepTime += 1;
      GraphState* s_tm1 = s_t;
      if (s_tm1 == 0)
        s_tm1 = O;
      const Action* a_tm1 = X;
      if (s_t == 0 || s_t == O)
        s_t = s_0;
      else
      {
        a_tm1 = a_t;
        s_t = s_tm1->nextState(*a_tm1);
        if (s_t == 0)
          s_t = O;
      }
      //a_t = Policies.decide(acting, s_t.v()); // fixMe
      a_t = &acting->sampleAction(); // fixMe
      double r_t = s_t->reward;
      if (!s_t->hasNextState())
      {
        a_t = X;
        s_t = O;
      }
      return StepData(stepTime, s_tm1, a_tm1, s_t, r_t, a_t);
    }

    virtual double gamma() const=0;
    virtual const DenseVector<double>& expectedDiscountedSolution() const=0;
    virtual ActionList* actions() const =0;
};

class LineProblem: public FiniteStateGraph
{
  public:
    const double Gamma;
    ActionList* actionList;
    Policy<double>* acting;
    std::vector<GraphState*>* states;
    DenseVector<double>* solution;
    const Action* Move;

    GraphState* A;
    GraphState* B;
    GraphState* C;
    GraphState* D;

    LineProblem() :
        FiniteStateGraph(), Gamma(0.9), actionList(new GeneralActionList(1)), acting(
            new SingleActionPolicy<double>(actionList)), states(new std::vector<GraphState*>()), solution(
            new RealVector(3)), Move(&actionList->at(0))
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
        RealVector* v = new RealVector;
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
      delete actionList;
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

    const DenseVector<double>& expectedDiscountedSolution() const
    {
      return *solution;
    }

    ActionList* actions() const
    {
      return actionList;
    }

};

class RandomWalk: public FiniteStateGraph
{
  public:
    const double Gamma;
    ActionList* actionList;
    std::vector<GraphState*>* states;
    DenseVector<double>* solution;
    const Action* Left;
    const Action* Right;

    GraphState* TL;
    GraphState* A;
    GraphState* B;
    GraphState* C;
    GraphState* D;
    GraphState* E;
    GraphState* TR;

    RealVector* distribution;
    Policy<double>* acting;

    RandomWalk() :
        FiniteStateGraph(), Gamma(0.9), actionList(new GeneralActionList(2)), states(
            new std::vector<GraphState*>()), solution(new RealVector(5)), Left(&actionList->at(0)), Right(
            &actionList->at(1))
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

      distribution = new RealVector(actionList->dimension());
      for (int i = 0; i < distribution->dimension(); i++)
        distribution->at(i) = 1.0 / actionList->dimension();

      acting = new ConstantPolicy<double>(actionList, distribution);

      int stateIndex = 0;
      for (std::vector<GraphState*>::iterator i = states->begin(); i != states->end(); ++i)
      {
        RealVector* v = new RealVector;
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
      delete actionList;
      delete distribution;
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

    const DenseVector<double>& expectedDiscountedSolution() const
    {
      return *solution;
    }

    ActionList* actions() const
    {
      return actionList;
    }

    void enableOnlyLeftPolicy()
    {
      if (acting)
        delete acting;
      actionList->erase(Right->id());
      acting = new SingleActionPolicy<double>(actionList);
      setPolicy(acting);
    }

    void enableOnlyRightPolicy()
    {
      if (acting)
        delete acting;
      actionList->erase(Left->id());
      acting = new SingleActionPolicy<double>(actionList);
      setPolicy(acting);
    }

    Policy<double>* getBehaviorPolicy(const double& behaviourLeftProbability)
    {
      if (acting)
        delete acting;
      distribution->at(0) = behaviourLeftProbability;
      distribution->at(1) = 1.0 - behaviourLeftProbability;
      acting = new ConstantPolicy<double>(actionList, distribution);
      return acting;
    }
};

template<class T = double, class O = double>
class FSGAgentState: public Projector<T, O>
{
  protected:
    FiniteStateGraph* graph;
    SparseVector<T>* featureState;
    std::map<GraphState*, int>* stateIndexes;
  public:
    FSGAgentState(FiniteStateGraph* graph) :
        graph(graph), featureState(0), stateIndexes(new std::map<GraphState*, int>)
    {
      std::vector<GraphState*>& states = graph->states();
      for (std::vector<GraphState*>::iterator iter = states.begin(); iter != states.end(); ++iter)
      {
        if ((*iter)->hasNextState())
          stateIndexes->insert(std::make_pair(*iter, stateIndexes->size()));
      }
      featureState = new SparseVector<T>(stateIndexes->size(), stateIndexes->size());
    }

    ~FSGAgentState()
    {
      delete featureState;
      delete stateIndexes;
    }

    const SparseVector<T>& project(const DenseVector<O>& x)
    {
      featureState->clear();
      if (x.empty())
        return *featureState;
      GraphState* sg = graph->state(x);
      if (!sg->hasNextState())
        return *featureState;
      featureState->setEntry(stateIndexes->at(sg), 1.0);
      return *featureState;
    }

    const SparseVector<T>& project(const DenseVector<O>& x, int h1)
    {
      return project(x);
    }

    double vectorNorm() const
    {
      return 1.0;
    }

    int dimension() const
    {
      return stateIndexes->size();
    }
};

}  // namespace RLLib

#endif /* STATEGRAPH_H_ */
