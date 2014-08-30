#include "compliance.h"

using namespace libMesh;

void Compliance::init_qoi( std::vector<Number>& sys_qoi )
{
	if (sys_qoi.size() == 0){
		sys_qoi.resize(1);
		compliance_index = 0;
	}
	else if(sys_qoi.size() == 1){
		sys_qoi.resize(2);
		compliance_index = 1;
	}
	else if(sys_qoi.size() == 2){
		sys_qoi.resize(3);
		compliance_index = 2;
	}
  return;
}

// We only have one QoI, so we don't bother checking the qois argument
// to see if it was requested from us
void Compliance::element_qoi (DiffContext &context,
                              const QoISet & /* qois */ )

{
  FEMContext &c = libmesh_cast_ref<FEMContext&>(context);

  FEVectorBase* elem_fe = NULL;
  c.get_element_fe( 0, elem_fe );

  // Element Jacobian * quadrature weights for interior integration
  const std::vector<Real> &JxW = elem_fe->get_JxW();

  unsigned int n_qpoints = c.get_element_qrule().n_points();

  Number dQoI_0 = 0.;

  // Loop over quadrature points


  for (unsigned int qp = 0; qp != n_qpoints; qp++)
    {
	  // Get the solution value at the quadrature point
	  Gradient U;
	  c.interior_value(0, qp, U);

	  f = _body_force(Point(0,0,0), "u");

	  // Update the elemental increment dR for each qp
	  dQoI_0 += JxW[qp] * U * f;
    }

  // Update the computed value of the global functional R, by adding the contribution from this element

  c.get_qois()[compliance_index] = c.get_qois()[compliance_index] + dQoI_0;

}

// We only have one QoI, so we don't bother checking the qois argument
// to see if it was requested from us
void Compliance::element_qoi_derivative (DiffContext &context,
                                         const QoISet & /* qois */)
{
  FEMContext &c = libmesh_cast_ref<FEMContext&>(context);

  // The dimension that we are running
  const unsigned int dim = c.get_dim();

  // First we get some references to cell-specific data that
  // will be used to assemble the linear system.
  FEVectorBase* elem_fe = NULL;
  c.get_element_fe( 0, elem_fe );

  // Element Jacobian * quadrature weights for interior integration
  const std::vector<Real> &JxW = elem_fe->get_JxW();

  // The basis functions for the element
  const std::vector<std::vector<RealGradient> > &phi = elem_fe->get_phi();

  // The number of local degrees of freedom in each variable
  const unsigned int n_T_dofs = c.get_dof_indices(0).size();
  unsigned int n_qpoints = c.get_element_qrule().n_points();

  // Fill the QoI RHS corresponding to this QoI. Since this is the 0th QoI
  // we fill in the [0][i] subderivatives, i corresponding to the variable index.
  // Our system has only one variable, so we only have to fill the [0][0] subderivative
  // DO NOT CONFUSE variable with design variable, here variable means displacement, velocity or other
  // implicit quantity
  DenseSubVector<Number> &Q = c.get_qoi_derivatives(compliance_index,0);


  f = _body_force( Point(0,0,0), "u");

  // Loop over the qps
  for (unsigned int qp=0; qp != n_qpoints; qp++)
    {
          for (unsigned int i=0; i != n_T_dofs; i++)
            Q(i) += JxW[qp] *phi[i][qp](i%dim)*f(i%dim) ;

    } // end of the quadrature point qp-loop
}

void Compliance::attach_body_force (Gradient fptr(const Point& , const std::string&))
{
	_body_force = fptr;
	  // We may be turning boundary side integration on or off
	  if (fptr)
		  body_force_included = true;
	  else
		  body_force_included = false;
}

