/*
 * NULLView.h
 *
 *  Created on: Oct 17, 2013
 *      Author: sam
 */

#ifndef NULLVIEW_H_
#define NULLVIEW_H_

#include "ViewBase.h"

namespace RLLibViz
{

// NULLView class represent and empty view
class NULLView: public ViewBase
{
  public:
    NULLView(QWidget *parent = 0) :
        ViewBase(parent)
    {
    }

    virtual ~NULLView()
    {
    }

    void initialize()
    {
    }

  public slots:
    void add(QWidget* that, const Vec&, const Vec&)
    {
      if (this != that)
        return;
    }

    void add(QWidget* that, const MatrixXd&)
    {
      if (this != that)
        return;
    }

    void draw(QWidget* that)
    {
      if (this != that)
        return;
    }
};

}  // namespace RLLibViz

#endif /* NULLVIEW_H_ */
