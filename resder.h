
#ifndef RESDER_H
#define RESDER_H

#include "libmesh/libmesh_common.h"
#include "libmesh/elem.h"
#include "libmesh/fe_base.h"
#include "libmesh/fem_context.h"
#include "libmesh/point.h"
#include "libmesh/quadrature.h"
#include "libmesh/diff_qoi.h"

// Bring in everything from the libMesh namespace
using namespace libMesh;

class TopOptSystem;

class ResidualDerivative : public DifferentiableQoI
{
public:
	ResidualDerivative(){};

  virtual ~ResidualDerivative(){}

  virtual void 	init_context (DiffContext &){};

  void init_context(DiffContext &context, TopOptSystem & sys);

  virtual void postprocess( ){}

  virtual void element_res_derivative(DiffContext &context, TopOptSystem & sys);

  virtual AutoPtr<DifferentiableQoI> clone( ) {
    return AutoPtr<DifferentiableQoI> ( new ResidualDerivative(*this) );
  }

protected:



};

#endif // RESDER_H

