/*
 * WindowLayout.cpp
 *
 *  Created on: Feb 3, 2016
 *      Author: sabeyruw
 */

#include "WindowLayout.h"

WindowLayout::WindowLayout() :
    topColumns(0), centerColumns(0), bottomColumns(0)
{
}

WindowLayout::~WindowLayout()
{
}

void WindowLayout::addTopWidget(QWidget *w)
{
  this->addWidget(w, 0, topColumns++);
}

void WindowLayout::addCenterWidget(QWidget *w)
{
  this->addWidget(w, 1, centerColumns++);
}

void WindowLayout::addBottomWidget(QWidget *w)
{
  this->addWidget(w, 2, bottomColumns++);
}
