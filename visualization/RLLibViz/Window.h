#ifndef WINDOW_H
#define WINDOW_H

#include <QWidget>
#include <QGridLayout>

#include "ViewBase.h"
#include <vector>

namespace RLLibViz
{

class Window: public QWidget
{
  Q_OBJECT

  public:
    typedef std::vector<ViewBase*> Views;
    typedef std::vector<ViewBase*> Plots;
    typedef std::vector<ViewBase*> VFuns;
    Views views;
    Plots plots;
    VFuns vfuns;
    QGridLayout* grid;
    int colsA, colsB, colsC;

  public:
    explicit Window(QWidget *parent = 0);
    virtual ~Window();
    void addView(ViewBase* view);
    void addPlot(ViewBase* view);
    void addValueFunctionView(ViewBase* valueFunctionView);

  protected:
    void keyPressEvent(QKeyEvent* event);

};

}  // namespace RLLibViz

#endif // WINDOW_H
