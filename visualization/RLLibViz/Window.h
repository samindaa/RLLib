#ifndef WINDOW_H
#define WINDOW_H

#include <QWidget>
#include <QHBoxLayout>
#include "RenderArea.h"
#include <vector>

class Window : public QWidget
{
  Q_OBJECT

public:

  enum { NB_AREA = 2};
  typedef std::vector<RenderArea*> Renders;
  Renders renders;

public:
  explicit Window(QWidget *parent = 0);
  
protected:
   void keyPressEvent(QKeyEvent *);
};

#endif // WINDOW_H
