/*
 * WindowLayout.h
 *
 *  Created on: Feb 3, 2016
 *      Author: sabeyruw
 */

#ifndef WINDOWLAYOUT_H_
#define WINDOWLAYOUT_H_

#include <QGridLayout>

class WindowLayout: public QGridLayout
{
  Q_OBJECT

  private:
    int topColumns, centerColumns, bottomColumns;

  public:
    WindowLayout();
    virtual ~WindowLayout();

    void addTopWidget(QWidget *w);
    void addCenterWidget(QWidget *w);
    void addBottomWidget(QWidget *w);
};

#endif /* WINDOWLAYOUT_H_ */
