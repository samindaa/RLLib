#include "BasisFunction.h"

//<< Sam
using namespace PoliFitted;

BasisFunction::BasisFunction()
{
}

BasisFunction::~BasisFunction()
{
}

namespace PoliFitted
{
ofstream& operator<<(ofstream& out, PoliFitted::BasisFunction& bf)
{
  bf.WriteOnStream(out);
  return out;
}

ifstream& operator>>(ifstream& in, PoliFitted::BasisFunction& bf)
{
  bf.ReadFromStream(in);
  return in;
}
}
