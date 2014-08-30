
#ifndef COMP_TRACTION_H
#define COMP_TRACTION_H

#include "libmesh/libmesh_common.h"
#include "libmesh/elem.h"
#include "libmesh/fe_base.h"
#include "libmesh/fem_context.h"
#include "libmesh/point.h"
#include "libmesh/quadrature.h"
#include "libmesh/diff_qoi.h"

// Bring in everything from the libMesh namespace
using namespace libMesh;

class ComplianceTraction : public DifferentiableQoI
{
public:
	ComplianceTraction(){};
  virtual ~ComplianceTraction(){}

  virtual void init_qoi( std::vector<Number>& sys_qoi );

  virtual void postprocess( ){}

  virtual void side_qoi_derivative(DiffContext &context, const QoISet & qois);

  virtual void side_qoi (DiffContext &context, const QoISet & qois);

  void attach_flux_bc_function (std::pair<bool,Gradient> fptr(const System& ,
                                                                          const Point& ,
                                                                          const std::string&));

  virtual AutoPtr<DifferentiableQoI> clone( ) {
    return AutoPtr<DifferentiableQoI> ( new ComplianceTraction(*this) );
  }

protected:
  /**
   * Pointer to function that returns BC information.
   */
  std::pair<bool,Gradient> (* _bc_function) (const System& system,
                                         const Point& p,
                                         const std::string& var_name);

  bool integrate_boundary_sides;

  unsigned int compliance_traction_index;


};
#endif // L_QOI_H
