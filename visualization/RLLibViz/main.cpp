#include <QApplication>
#include "RLLibVizMediator.h"

#include "Matrix.h"
#include "Vec.h"

using namespace RLLibViz;
using namespace RLLib;

Q_DECLARE_METATYPE(Vec)
Q_DECLARE_METATYPE(Matrix)

int main(int argc, char *argv[])
{
  QApplication a(argc, argv);

  qRegisterMetaType<Vec>();
  qRegisterMetaType<Matrix>();

  RLLibVizMediator w;
  w.show();

  return a.exec();
}
