/*
 * ContinuousGridworldView.h
 *
 *  Created on: Oct 11, 2013
 *      Author: sam
 */

#ifndef CONTINUOUSGRIDWORLDVIEW_H_
#define CONTINUOUSGRIDWORLDVIEW_H_

#include "ViewBase.h"

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

public:
  ContinuousGridworldView(QWidget *parent = 0);
  virtual ~ContinuousGridworldView();

  void initialize();
  void add(const Vec& p);
};

}  // namespace RLLibViz

#endif /* CONTINUOUSGRIDWORLDVIEW_H_ */
