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

    GraphState* nextState(const Action<double>& action)
    {
      return transitions[action.id()];
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
        O(new GraphState("NULL", std::numeric_limits<int>::min())), X(
            new Action<double>(std::numeric_limits<int>::max())), stepTime(-1), s_0(0), a_t(0), s_t(
            0), acting(0), graphStates(0)
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

    GraphState* state(const Vector<double>* s)
    {
      return graphStates->at(s->getEntry(0));
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
        s_t = s_tm1->nextState(*a_tm1);
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
    virtual ActionList<double>* actions() const =0;
};

class LineProblem: public FiniteStateGraph
{
  public:
    const double Gamma;
    ActionList<double>* actionList;
    Policy<double>* acting;
    std::vector<GraphState*>* states;
    DenseVector<double>* solution;
    const Action<double>* Move;

    GraphState* A;
    GraphState* B;
    GraphState* C;
    GraphState* D;

    LineProblem() :
        FiniteStateGraph(), Gamma(0.9), actionList(new GeneralActionList<double>(1)), acting(
            new SingleActionPolicy<double>(actionList)), states(new std::vector<GraphState*>()), solution(
            new PVector<double>(3)), Move(actionList->at(0))
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

    const Vector<double>* expectedDiscountedSolution() const
    {
      return solution;
    }

    ActionList<double>* actions() const
    {
      return actionList;
    }

};

class RandomWalk: public FiniteStateGraph
{
  public:
    const double Gamma;
    ActionList<double>* actionList;
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

    DenseVector<double>* distribution;
    Policy<double>* acting;

    RandomWalk() :
        FiniteStateGraph(), Gamma(0.9), actionList(new GeneralActionList<double>(2)), states(
            new std::vector<GraphState*>()), solution(new PVector<double>(5)), Left(
            actionList->at(0)), Right(actionList->at(1))
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

      distribution = new PVector<double>(actionList->dimension());
      for (int i = 0; i < distribution->dimension(); i++)
        distribution->at(i) = 1.0 / actionList->dimension();

      acting = new ConstantPolicy<double>(actionList, distribution);

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

    const Vector<double>* expectedDiscountedSolution() const
    {
      return solution;
    }

    ActionList<double>* actions() const
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

class FSGAgentState: public StateToStateAction<double>
{
  protected:
    FiniteStateGraph* graph;
    Vector<double>* featureState;
    std::map<GraphState*, int>* stateIndexes;
    Representations<double>* phis;
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
      featureState = new PVector<double>(stateIndexes->size());
      phis = new Representations<double>(stateIndexes->size() * graph->actions()->dimension(),
          graph->actions());
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
          featureState->setEntry(stateIndexes->at(sg), 1.0);
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
      phis->at(a)->setEntry(a->id() * stateIndexes->size() + stateIndexes->at(sg), 1);
      return phis->at(a);
    }

    const Representations<double>* stateActions(const Vector<double>* x)
    {
      phis->clear();
      if (x->empty())
        return phis;
      for (ActionList<double>::const_iterator a = graph->actions()->begin();
          a != graph->actions()->end(); ++a)
        stateAction(x, *a);
      return phis;
    }

    const ActionList<double>* getActionList() const
    {
      return graph->actions();
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
};

}  // namespace RLLib

#endif /* STATEGRAPH_H_ */
