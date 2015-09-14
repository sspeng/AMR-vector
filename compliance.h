
#ifndef L_QOI_H
#define L_QOI_H

#include "libmesh/libmesh_common.h"
#include "libmesh/elem.h"
#include "libmesh/fe_base.h"
#include "libmesh/fem_context.h"
#include "libmesh/point.h"
#include "libmesh/quadrature.h"
#include "libmesh/diff_qoi.h"

// Bring in everything from the libMesh namespace
using namespace libMesh;

class Compliance : public DifferentiableQoI
{
public:
  Compliance(){};
  virtual ~Compliance(){}

  virtual void init_qoi( std::vector<Number>& sys_qoi );

  virtual void postprocess( ){}

  virtual void element_qoi_derivative(DiffContext &context, const QoISet & qois);

  virtual void element_qoi (DiffContext &context, const QoISet & qois);

  void attach_body_force (Gradient fptr(const Point& p,const std::string& var_name));

  virtual UniquePtr<DifferentiableQoI> clone( ) {
    return UniquePtr<DifferentiableQoI> ( new Compliance(*this) );
  }

protected:
  Gradient (* _body_force) ( const Point& p, const std::string& var_name);

  bool body_force_included = false;

  Gradient f;

  unsigned int compliance_index;


};
#endif // L_QOI_H
