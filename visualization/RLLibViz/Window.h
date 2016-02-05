#ifndef WINDOW_H
#define WINDOW_H

#include <vector>
#include <QWidget>

#include "ModelBase.h"
#include "ViewBase.h"
#include "WindowLayout.h"

namespace RLLibViz
{

  class ModelBase;

  class Window: public QWidget
  {
    Q_OBJECT

    private:
      WindowLayout* windowLayout;

    public:
      typedef std::vector<ViewBase*> WindowVector;
      WindowVector problemVector;
      WindowVector plotVector;
      WindowVector valueFunctionVector;

      explicit Window(QWidget* parent = 0);
      virtual ~Window();

      void initialize(ModelBase* modelBase);
      bool empty() const;
      void newLayout();
      void addProblemView(ViewBase* view);
      void addPlotView(ViewBase* view);
      void addValueFunctionView(ViewBase* valueFunctionView);

    protected:
      void keyPressEvent(QKeyEvent* event);

  };

}  // namespace RLLibViz

#endif // WINDOW_H
