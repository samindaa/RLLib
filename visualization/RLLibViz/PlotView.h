/*
 * PlotView.h
 *
 *  Created on: Oct 13, 2013
 *      Author: sam
 */

#ifndef PLOTVIEW_H_
#define PLOTVIEW_H_

#include "ViewBase.h"
#include "Mat.h"
#include <vector>

namespace RLLibViz
{

class PlotView: public ViewBase
{
  Q_OBJECT
  private:
    std::vector<Vec> points;
  public:
    PlotView(QWidget *parent = 0);
    virtual ~PlotView();

    void initialize();
    void add(const Vec& p);

    void draw();
    void paintEvent(QPaintEvent* event);
};

}  // namespace RLLibViz

#endif /* PLOTVIEW_H_ */
