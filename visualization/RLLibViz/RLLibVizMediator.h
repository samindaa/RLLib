#ifndef RLLIBVIZ_H
#define RLLIBVIZ_H

#include <map>
#include <QMainWindow>
#include <QRadioButton>

#include "Window.h"
#include "ThreadBase.h"

namespace Ui
{
class RLLibVizMediator;
}

class RLLibVizMediator: public QMainWindow
{
  Q_OBJECT

  public:
    explicit RLLibVizMediator(QWidget *parent = 0);
    ~RLLibVizMediator();

  private slots:
    void execClicked();
    void stopClicked();

  private:
    Ui::RLLibVizMediator *ui;
    typedef std::map<QRadioButton*, std::pair<RLLibViz::Window*, RLLibViz::ModelBase*> > DemoProblems;
    typedef std::map<std::string, RLLibViz::ThreadBase*> DemoThreads;
    DemoProblems demoProblems;
    DemoThreads demoThreads;
    RLLibViz::Window* currentWindow;
    RLLibViz::ModelBase* currentModelBase;
};

#endif // RLLIBVIZ_H
