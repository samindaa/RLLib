#include <QApplication>
#include "RLLibVizMediator.h"

#include "Eigen/Dense"
#include "Vec.h"

using namespace RLLibViz;
using namespace RLLib;
using namespace Eigen;

Q_DECLARE_METATYPE(Vec)
Q_DECLARE_METATYPE(MatrixXd)

int main(int argc, char *argv[])
{
  QApplication a(argc, argv);

  qRegisterMetaType<Vec>();
  qRegisterMetaType<MatrixXd>();

  RLLibVizMediator w;
  w.show();

  return a.exec();
}
