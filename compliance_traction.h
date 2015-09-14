
#ifndef COMP_TRACTION_H
#define COMP_TRACTION_H

#include "libmesh/libmesh_common.h"
#include "libmesh/elem.h"
#include "libmesh/fe_base.h"
#include "libmesh/fem_context.h"
#include "libmesh/point.h"
#include "libmesh/quadrature.h"
#include "libmesh/diff_qoi.h"
#include "libmesh/auto_ptr.h"
#include "libmesh/numeric_vector.h"
#include "TopOpt.h"
#include "libmesh/explicit_system.h"

// Bring in everything from the libMesh namespace
using namespace libMesh;

class ComplianceTraction : public DifferentiableQoI
{
public:


	ComplianceTraction(ExplicitSystem * densities, TopOptSystem * femsystem);
  virtual ~ComplianceTraction(){}

  virtual void init_qoi( std::vector<Number>& sys_qoi );

  virtual void postprocess( ){}

  virtual void side_qoi_derivative(DiffContext &context, const QoISet & qois);

  virtual void side_qoi (DiffContext &context, const QoISet & qois);

  void attach_flux_bc_function (std::pair<bool,Gradient> fptr(const TopOptSystem& ,
                                                                          const Point& ,
                                                                          const std::string&));

  virtual void element_qoi_for_FD (UniquePtr<NumericVector<Number> > & local_solution,
											UniquePtr<NumericVector<Number> >& densities_vector,
													const Elem * elem,
													DiffContext & context, Number & qoi_computed);

  virtual UniquePtr<DifferentiableQoI> clone( ) {
    return UniquePtr<DifferentiableQoI> ( new ComplianceTraction(*this) );
  }

protected:
  /**
   * Pointer to function that returns BC information.
   */
  std::pair<bool,Gradient> (* _bc_function) (const TopOptSystem& system,
                                         const Point& p,
                                         const std::string& var_name);

  bool integrate_boundary_sides;

  unsigned int compliance_traction_index;

  ExplicitSystem * _densities;

  TopOptSystem * _femsystem;

};
#endif // L_QOI_H
