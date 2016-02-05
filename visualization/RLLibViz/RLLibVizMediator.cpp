#include "RLLibVizMediator.h"
#include "ui_RLLibVizForm.h"
//
#include "MountainCarModel.h"
#include "MountainCarModel2.h"
#include "MountainCarModel3.h"
#include "ContinuousGridworldModel.h"
#include "SwingPendulumModel.h"
#include "SwingPendulumModel2.h"
#include "SwingPendulumModel3.h"
#include "SwingPendulumModel4.h"
#include "AcrobotModel.h"
#include "CartPoleModel.h"
//
#include "MountainCarView.h"
#include "ContinuousGridworldView.h"
#include "SwingPendulumView.h"
#include "AcrobotView.h"
#include "CartPoleView.h"
//
#include "ValueFunctionView.h"
#include "NULLView.h"
#include "PlotView.h"
//
#include "LearningThread.h"
#include "EvaluationThread.h"

//-- Our access
RLLibVizMediator::RLLibVizMediator(QWidget *parent) :
    QMainWindow(parent), ui(new Ui::RLLibVizMediator), currentWindow(0), currentModelBase(0)
{
  ui->setupUi(this);
  connect(ui->pushButtonExec, SIGNAL(clicked()), this, SLOT(execClicked()));
  connect(ui->pushButtonStop, SIGNAL(clicked()), this, SLOT(stopClicked()));
  connect(ui->horizontalSliderSimulationSpeed, SIGNAL(valueChanged(int)),
      ui->progressBarSimulationSpeed, SLOT(setValue(int)));
  connect(ui->horizontalSliderSimulationSpeed, SIGNAL(valueChanged(int)), this,
      SLOT(updateSimulationSpeed(int)));

  ui->pushButtonStop->setEnabled(false);

  threadMap.insert(std::make_pair("doLearning", new RLLibViz::LearningThread));
  threadMap.insert(std::make_pair("doEvaluation", new RLLibViz::EvaluationThread));

  for (ThreadMap::iterator iter = threadMap.begin(); iter != threadMap.end(); ++iter)
    iter->second->start();

  problemMap.insert(
      std::make_pair(ui->radioButtonMountainCarOffPAC,
          std::make_pair(new RLLibViz::Window, new RLLibViz::MountainCarModel)));
  ProblemMap::iterator iter = problemMap.find(ui->radioButtonMountainCarOffPAC);
  iter->second.first->addProblemView(new RLLibViz::MountainCarView);
  iter->second.first->addProblemView(new RLLibViz::MountainCarView);
  iter->second.first->addPlotView(new RLLibViz::PlotView("Behavior Policy"));
  iter->second.first->addPlotView(new RLLibViz::PlotView("Target Policy"));
  iter->second.first->addValueFunctionView(new RLLibViz::NULLView);
  iter->second.first->addValueFunctionView(new RLLibViz::ValueFunctionView);
  iter->second.first->initialize(iter->second.second);

  problemMap.insert(
      std::make_pair(ui->radioButtonContinuousGridworldOffPAC,
          std::make_pair(new RLLibViz::Window, new RLLibViz::ContinuousGridworldModel)));
  iter = problemMap.find(ui->radioButtonContinuousGridworldOffPAC);
  iter->second.first->addProblemView(new RLLibViz::ContinuousGridworldView);
  iter->second.first->addProblemView(new RLLibViz::ContinuousGridworldView);
  iter->second.first->addPlotView(new RLLibViz::PlotView("Behavior Policy"));
  iter->second.first->addPlotView(new RLLibViz::PlotView("Target Policy"));
  iter->second.first->addValueFunctionView(new RLLibViz::NULLView);
  iter->second.first->addValueFunctionView(new RLLibViz::ValueFunctionView);
  iter->second.first->initialize(iter->second.second);

  problemMap.insert(
      std::make_pair(ui->radioButtonSwingPendulumAverageRewardActorCritic,
          std::make_pair(new RLLibViz::Window, new RLLibViz::SwingPendulumModel)));
  iter = problemMap.find(ui->radioButtonSwingPendulumAverageRewardActorCritic);
  iter->second.first->addProblemView(new RLLibViz::SwingPendulumView);
  iter->second.first->addPlotView(new RLLibViz::PlotView("Target Policy"));
  iter->second.first->addValueFunctionView(new RLLibViz::ValueFunctionView);
  iter->second.first->initialize(iter->second.second);

  problemMap.insert(
      std::make_pair(ui->radioButtonSwingPendulumOffPAC,
          std::make_pair(new RLLibViz::Window, new RLLibViz::SwingPendulumModel2)));
  iter = problemMap.find(ui->radioButtonSwingPendulumOffPAC);
  iter->second.first->addProblemView(new RLLibViz::SwingPendulumView);
  iter->second.first->addProblemView(new RLLibViz::SwingPendulumView);
  iter->second.first->addPlotView(new RLLibViz::PlotView("Behavior Policy"));
  iter->second.first->addPlotView(new RLLibViz::PlotView("Target Policy"));
  iter->second.first->addValueFunctionView(new RLLibViz::NULLView);
  iter->second.first->addValueFunctionView(new RLLibViz::ValueFunctionView);
  iter->second.first->initialize(iter->second.second);

  problemMap.insert(
      std::make_pair(ui->radioButtonMountainCarSarsaAlphaBound,
          std::make_pair(new RLLibViz::Window, new RLLibViz::MountainCarModel2)));
  iter = problemMap.find(ui->radioButtonMountainCarSarsaAlphaBound);
  iter->second.first->addProblemView(new RLLibViz::MountainCarView);
  iter->second.first->addPlotView(new RLLibViz::PlotView("Target Policy"));
  iter->second.first->addValueFunctionView(new RLLibViz::ValueFunctionView);
  iter->second.first->initialize(iter->second.second);

  problemMap.insert(
      std::make_pair(ui->radioButtonSwingPendulumSarsaTrue,
          std::make_pair(new RLLibViz::Window, new RLLibViz::SwingPendulumModel3)));
  iter = problemMap.find(ui->radioButtonSwingPendulumSarsaTrue);
  iter->second.first->addProblemView(new RLLibViz::SwingPendulumView);
  iter->second.first->addPlotView(new RLLibViz::PlotView("Target Policy"));
  iter->second.first->addValueFunctionView(new RLLibViz::ValueFunctionView);
  iter->second.first->initialize(iter->second.second);

  problemMap.insert(
      std::make_pair(ui->radioButtonSwingPendulumSarsaAlphaBoundFourierBasis,
          std::make_pair(new RLLibViz::Window, new RLLibViz::SwingPendulumModel4)));
  iter = problemMap.find(ui->radioButtonSwingPendulumSarsaAlphaBoundFourierBasis);
  iter->second.first->addProblemView(new RLLibViz::SwingPendulumView);
  iter->second.first->addPlotView(new RLLibViz::PlotView("Target Policy"));
  iter->second.first->addValueFunctionView(new RLLibViz::ValueFunctionView);
  iter->second.first->initialize(iter->second.second);

  problemMap.insert(
      std::make_pair(ui->radioButtonMountainCarSarsaAlphaBoundFourierBasis,
          std::make_pair(new RLLibViz::Window, new RLLibViz::MountainCarModel3)));
  iter = problemMap.find(ui->radioButtonMountainCarSarsaAlphaBoundFourierBasis);
  iter->second.first->addProblemView(new RLLibViz::MountainCarView);
  iter->second.first->addPlotView(new RLLibViz::PlotView("Target Policy"));
  iter->second.first->addValueFunctionView(new RLLibViz::ValueFunctionView);
  iter->second.first->initialize(iter->second.second);

  problemMap.insert(
      std::make_pair(ui->radioButtonAcrobotSarasaAlphaBoundFourierBasis,
          std::make_pair(new RLLibViz::Window, new RLLibViz::AcrobotModel)));
  iter = problemMap.find(ui->radioButtonAcrobotSarasaAlphaBoundFourierBasis);
  iter->second.first->addProblemView(new RLLibViz::AcrobotView);
  iter->second.first->addPlotView(new RLLibViz::PlotView("Target Policy"));
  //iter->second.first->addValueFunctionView(new RLLibViz::ValueFunctionView);
  iter->second.first->initialize(iter->second.second);

  problemMap.insert(
      std::make_pair(ui->radioButtonCartPoleSarsaAlphaBoundFourierBasis,
          std::make_pair(new RLLibViz::Window, new RLLibViz::CartPoleModel)));
  iter = problemMap.find(ui->radioButtonCartPoleSarsaAlphaBoundFourierBasis);
  iter->second.first->addProblemView(new RLLibViz::CartPoleView);
  iter->second.first->addPlotView(new RLLibViz::PlotView("Target Policy"));
  //iter->second.first->addValueFunctionView(new RLLibViz::ValueFunctionView);
  iter->second.first->initialize(iter->second.second);

  // Add more problems

}

RLLibVizMediator::~RLLibVizMediator()
{
  delete ui;
  // FixMe: Sam
}

void RLLibVizMediator::execClicked()
{
  if (currentWindow) // remove the old one
  {
    currentWindow->setVisible(false);
    if (ui->mainWidget->layout())
      ui->mainWidget->layout()->removeWidget(currentWindow);
    delete ui->mainWidget->layout();
    currentWindow = 0;
    currentModelBase = 0;
  }

  ProblemMap::iterator iter = problemMap.begin();
  for (; iter != problemMap.end(); ++iter)
  {
    if (iter->first->isChecked())
      break;
  }

  if (iter != problemMap.end())
  {
    currentWindow = iter->second.first;
    currentModelBase = iter->second.second;
    currentWindow->setVisible(false);
    currentWindow->newLayout();
    ui->mainWidget->setLayout(new QGridLayout);
    ui->mainWidget->layout()->addWidget(currentWindow);
    currentWindow->setVisible(true);
    for (ThreadMap::iterator iter = threadMap.begin(); iter != threadMap.end(); ++iter)
    {
      iter->second->setModel(currentModelBase);
      iter->second->setWindow(currentWindow);
      iter->second->setActive(true);
    }
    ui->pushButtonExec->setEnabled(false);
    ui->pushButtonStop->setEnabled(true);
  }

}

void RLLibVizMediator::stopClicked()
{
  for (ThreadMap::iterator iter = threadMap.begin(); iter != threadMap.end(); ++iter)
    iter->second->setActive(false);
  ui->pushButtonExec->setEnabled(true);
  ui->pushButtonStop->setEnabled(false);
}

void RLLibVizMediator::updateSimulationSpeed(int value)
{
  for (ThreadMap::iterator iter = threadMap.begin(); iter != threadMap.end(); ++iter)
    iter->second->setSimulationSpeed(value);
}
