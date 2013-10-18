/*
 * ViewBase.cpp
 *
 *  Created on: Oct 11, 2013
 *      Author: sam
 */

#include "ViewBase.h"

using namespace RLLibViz;

ViewBase::ViewBase(QWidget *parent) :
    QWidget(parent)
{
}

ViewBase::~ViewBase()
{
}

QSize ViewBase::minimumSizeHint() const
{
  return QSize(200, 200);
}

QSize ViewBase::sizeHint() const
{
  return QSize(400, 400);
}
