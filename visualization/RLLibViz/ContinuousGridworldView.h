/*
 * ContinuousGridworldView.h
 *
 *  Created on: Oct 11, 2013
 *      Author: sam
 */

#ifndef CONTINUOUSGRIDWORLDVIEW_H_
#define CONTINUOUSGRIDWORLDVIEW_H_

#include "ViewBase.h"
#include "Framebuffer.h"

#include <QResizeEvent>

namespace RLLibViz
{

class ContinuousGridworldView: public ViewBase
{
  Q_OBJECT

  public:
    Vec vecE;
    Vec vecX;
    Vec vecY;
    Mat T;

  protected:
    Framebuffer* buffers[2];
    Framebuffer* current;
    Framebuffer* next;

  public:
    ContinuousGridworldView(QWidget *parent = 0);
    virtual ~ContinuousGridworldView();
    void initialize();

  public slots:
    void add(QWidget* that, const Vec& p1, const Vec& p2);
    void draw(QWidget* that);
    void add(QWidget* that, const MatrixXd& mat);

  protected:
    void swap();
    void resizeEvent(QResizeEvent *);
    void paintEvent(QPaintEvent *);
};

}  // namespace RLLibViz

#endif /* CONTINUOUSGRIDWORLDVIEW_H_ */
