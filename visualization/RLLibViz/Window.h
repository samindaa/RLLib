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
  Views views;
  Plots plots;
  ViewBase* valueFunctionView;
  QLayout* grid;
  int colsA, colsB;

public:
  explicit Window(QWidget *parent = 0);
  virtual ~Window();
  void addView(ViewBase* view);
  void addPlot(ViewBase* view);
  void setValueFunctionView(ViewBase* valueFunctionView);

protected:
  void keyPressEvent(QKeyEvent* event);

};

}  // namespace RLLibViz

#endif // WINDOW_H
