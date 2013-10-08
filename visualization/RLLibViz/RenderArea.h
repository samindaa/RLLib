#ifndef RENDERAREA_H
#define RENDERAREA_H

#include <QWidget>
#include <QBrush>
#include <QPen>
#include <QPainter>
#include <QKeyEvent>
#include <vector>

class RenderArea : public QWidget
{
  Q_OBJECT

private:
  int offsetLeft, offsetRight, offsetTop, offsetBottom;
  
public:
  typedef std::vector<std::pair<float, float> > Poses;
  Poses poses;
public:
  RenderArea(QWidget *parent = 0);
  ~RenderArea();

  QSize minimumSizeHint() const;
  QSize sizeHint() const;


protected:
  void paintEvent(QPaintEvent *);

};

#endif // RENDERAREA_H
