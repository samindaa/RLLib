/*
 * ValueFunctionView.h
 *
 *  Created on: Oct 15, 2013
 *      Author: sam
 */

#ifndef VALUEFUNCTIONVIEW_H_
#define VALUEFUNCTIONVIEW_H_

#include <QImage>
#include <QLabel>
#include <QScrollArea>
#include <QHBoxLayout>
#include <QPixmap>

#include "Math.h"
#include "ViewBase.h"
#include "Mat.h"

#include <vector>
#include <cmath>

namespace RLLibViz
{

// http://www.andrewnoske.com/wiki/Code_-_heatmaps_and_color_gradients
class ColorGradient
{
  private:
    struct ColorPoint  // Internal class used to store colors at different points in the gradient.
    {
        float r, g, b;      // Red, green and blue values of our color.
        float val;        // Position of our color along the gradient (between 0 and 1).
        ColorPoint(float red, float green, float blue, float value) :
            r(red), g(green), b(blue), val(value)
        {
        }
    };
    std::vector<ColorPoint> color;      // An array of color points in ascending value.

  public:
    //-- Default constructor:
    ColorGradient()
    {
      createDefaultHeatMapGradient();
    }

    //-- Inserts a new color point into its correct position:
    void addColorPoint(float red, float green, float blue, float value)
    {
      for (int i = 0; i < (int) color.size(); i++)
      {
        if (value < color[i].val)
        {
          color.insert(color.begin() + i, ColorPoint(red, green, blue, value));
          return;
        }
      }
      color.push_back(ColorPoint(red, green, blue, value));
    }

    //-- Inserts a new color point into its correct position:
    void clearGradient()
    {
      color.clear();
    }

    //-- Places a 5 color heapmap gradient into the "color" vector:
    void createDefaultHeatMapGradient()
    {
      color.clear();
      color.push_back(ColorPoint(0, 0, 1, 0.0f));      // Blue.
      color.push_back(ColorPoint(0, 1, 1, 0.25f));     // Cyan.
      color.push_back(ColorPoint(0, 1, 0, 0.5f));      // Green.
      color.push_back(ColorPoint(1, 1, 0, 0.75f));     // Yellow.
      color.push_back(ColorPoint(1, 0, 0, 1.0f));      // Red.
    }

    //-- Inputs a (value) between 0 and 1 and outputs the (red), (green) and (blue)
    //-- values representing that position in the gradient.
    QRgb getColorAtValue(const float value)
    {
      if (color.size() == 0)
        return qRgb(255, 255, 255);

      for (int i = 0; i < (int) color.size(); i++)
      {
        ColorPoint &currC = color[i];
        if (value < currC.val)
        {
          ColorPoint &prevC = color[std::max(0, i - 1)];
          float valueDiff = (prevC.val - currC.val);
          float fractBetween = (valueDiff == 0) ? 0 : (value - currC.val) / valueDiff;
          float red = (prevC.r - currC.r) * fractBetween + currC.r;
          float green = (prevC.g - currC.g) * fractBetween + currC.g;
          float blue = (prevC.b - currC.b) * fractBetween + currC.b;
          red *= 255.0;
          green *= 255.0;
          blue *= 255.0;
          return qRgb(red, green, blue);
        }
      }
      float red = color.back().r;
      float green = color.back().g;
      float blue = color.back().b;
      red *= 255.0;
      green *= 255.0;
      blue *= 255.0;
      return qRgb(red, green, blue);
    }
};

class ValueFunctionView: public ViewBase
{
  Q_OBJECT
  private:
    QImage* image;
    QLabel* imageLabel;
    QHBoxLayout* mainLayour;
    ColorGradient heatMapGradient;

  public:
    ValueFunctionView(QWidget *parent = 0);
    virtual ~ValueFunctionView();
    void initialize();

  public slots:
    void add(QWidget*, const Vec&, const Vec&);
    void draw(QWidget* that);
    void add(QWidget* that, const MatrixXd& mat);
};

}  // namespace RLLibViz

#endif /* VALUEFUNCTIONVIEW_H_ */
