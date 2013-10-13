#ifndef WINDOW_H
#define WINDOW_H

#include <QWidget>
#include <QHBoxLayout>
#include "ViewBase.h"
#include <vector>

namespace RLLibViz
{

class Window: public QWidget
{
Q_OBJECT

public:
  typedef std::vector<ViewBase*> Views;
  Views views;
  QHBoxLayout* mainLayout;
public:
  explicit Window(QWidget *parent = 0);
  virtual ~Window();
  void addView(ViewBase* view);

protected:
  void keyPressEvent(QKeyEvent *);
};

}  // namespace RLLibViz

#endif // WINDOW_H
