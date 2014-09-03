#include "compliance_traction.h"
#include "libmesh/elasticity_tools.h"
#include "libmesh/dof_map.h"

using namespace libMesh;

ComplianceTraction::ComplianceTraction(ExplicitSystem * densities, TopOptSystem * femsystem){

	_densities = densities;

	_femsystem = femsystem;
}

void ComplianceTraction::init_qoi( std::vector<Number>& sys_qoi )
{
	if (sys_qoi.size() == 0){
		sys_qoi.resize(1);
		compliance_traction_index = 0;
	}
	else if(sys_qoi.size() == 1){
		sys_qoi.resize(2);
		compliance_traction_index = 1;
	}
	else if(sys_qoi.size() == 2){
		sys_qoi.resize(3);
		compliance_traction_index = 2;
	}
  return;
}

void ComplianceTraction::attach_flux_bc_function (std::pair<bool,Gradient> fptr(const System& ,
                                                                        const Point& ,
                                                                        const std::string&))
{
  _bc_function = fptr;

  // We may be turning boundary side integration on or off
  if (fptr)
    integrate_boundary_sides = true;
  else
    integrate_boundary_sides = false;
}

// We only have one QoI, so we don't bother checking the qois argument
// to see if it was requested from us
void ComplianceTraction::side_qoi (DiffContext &context,
                              const QoISet & /* qois */ )

{
  FEMContext &c = libmesh_cast_ref<FEMContext&>(context);

  FEVectorBase* elem_fe_face = NULL;
  c.get_side_fe( 0, elem_fe_face );

  // Element Jacobian * quadrature weights for interior integration
  const std::vector<Real> &JxW_side = elem_fe_face->get_JxW();

  unsigned int n_qpoints = c.get_side_qrule().n_points();

  Number dQoI_0 = 0.;

  // Because the traction is constant along the face, we call it now
  System * dummy;
  std::pair<bool,Gradient>  flux_pair = _bc_function(*dummy,Point(0,0,0), "u");

  // Only integrate on the boundary where the traction is apploed

  // Grab boundary ids in side element
  std::vector<boundary_id_type> side_BdId = c.side_boundary_ids();
  //short int bc_id = mesh.boundary_info->boundary_id	(elem,side);



  if (!side_BdId.empty() && side_BdId[0] == 1 && integrate_boundary_sides)
  {
	  // Loop over quadrature points
	  for (unsigned int qp = 0; qp != n_qpoints; qp++)
		{
		  // Get the solution value at the quadrature point
		  Gradient U;
		  c.side_value(0,qp,U);

		  // Update the elemental increment dR for each qp
		  dQoI_0 += JxW_side[qp] * U * flux_pair.second;
		}
  }

  // Update the computed value of the global functional R, by adding the contribution from this element

  c.get_qois()[compliance_traction_index] = c.get_qois()[compliance_traction_index] + dQoI_0;

}

// We only have one QoI, so we don't bother checking the qois argument
// to see if it was requested from us
void ComplianceTraction::side_qoi_derivative (DiffContext &context,
                                         const QoISet & /* qois */)
{
  FEMContext &c = libmesh_cast_ref<FEMContext&>(context);

  // The dimension that we are running
  const unsigned int dim = c.get_dim();

  // First we get some references to cell-specific data that
  // will be used to assemble the linear system.
  FEVectorBase* elem_fe_face = NULL;
  c.get_side_fe( 0, elem_fe_face );

  // Element Jacobian * quadrature weights for interior integration
  const std::vector<Real> &JxW_side = elem_fe_face->get_JxW();


  // The basis functions for the element
  const std::vector<std::vector<RealGradient> > &phi = elem_fe_face->get_phi();

  // The number of local degrees of freedom in each variable
  const unsigned int n_T_dofs = c.get_dof_indices(0).size();
  unsigned int n_qpoints = c.get_side_qrule().n_points();

  // Fill the QoI RHS corresponding to this QoI. Since this is the 0th QoI
  // we fill in the [0][i] subderivatives, i corresponding to the variable index.
  // Our system has only one variable, so we only have to fill the [0][0] subderivative
  // DO NOT CONFUSE variable with design variable, here variable means displacement, velocity or other
  // implicit quantity
  DenseSubVector<Number> &Q = c.get_qoi_derivatives(compliance_traction_index,0);


  // Because the traction is constant along the face, we call it now
  System * dummy;
  std::pair<bool,Gradient>  flux_pair = _bc_function(*dummy,Point(0,0,0), "u");

  // Grab boundary ids in side element
  std::vector<boundary_id_type> side_BdId = c.side_boundary_ids();
  //short int bc_id = mesh.boundary_info->boundary_id	(elem,side);
  if (!side_BdId.empty() && side_BdId[0] == 1 && integrate_boundary_sides)
  {
	  // Loop over the qps
//	  for (unsigned int qp=0; qp != n_qpoints; qp++)
//		{
//			  for (unsigned int i=0; i != n_T_dofs; i++)
//				Q(i) += JxW_side[qp] *phi[i][qp](i%dim)*flux_pair.second(i%dim) ;
//		} // end of the quadrature point qp-loop

	  // Loop over the qps
	  DenseMatrix<Number> Nhat;
	  Nhat.resize(phi.size(),dim);
	  DenseVector<Number> Qtemp, flux;
	  flux.resize(dim);
	  for (unsigned int qp=0; qp != n_qpoints; qp++)
		{
	 	  ElasticityTools::NHatMatrix( Nhat, dim,  phi, qp);
	 	  flux(0) = flux_pair.second(0);
	 	  flux(1) = flux_pair.second(1);
	 	  Nhat.vector_mult(Qtemp, flux);

		  for (unsigned int i=0; i != n_T_dofs; i++)
			Q(i) += JxW_side[qp] *Qtemp(i) ;


		} // end of the quadrature point qp-loop
  }
}


void ComplianceTraction::element_qoi_for_FD (AutoPtr<NumericVector<Number> > & local_solution,
											AutoPtr<NumericVector<Number> >& densities_vector,
													const Elem * elem,
													DiffContext & context, Number & qoi_computed)
{
	FEMContext &c = libmesh_cast_ref<FEMContext&>(context);
	// Grab boundary ids in side element
	std::vector<boundary_id_type> side_BdId = c.side_boundary_ids();

	  if (!side_BdId.empty() && side_BdId[0] == 1 && integrate_boundary_sides)
	  {


		FEVectorBase* elem_fe_face = NULL;
		c.get_side_fe( 0, elem_fe_face );

		// Element Jacobian * quadrature weights for interior integration
		const std::vector<Real> &JxW_side = elem_fe_face->get_JxW();

		// Derivatives of the element basis functions
		const std::vector<std::vector<RealGradient> > phi = elem_fe_face->get_phi();

		// Number of gauss points
		unsigned int n_qpoints = c.get_side_qrule().n_points();

		const unsigned int dim = c.get_dim();

		Number VonMises = 0.;


		// Calculate stresses first, we only pick one gauss point, because the gradient is constant
		DenseMatrix<Number>  Nhat;
		DenseVector<Number> U, U_sol;
		Nhat.resize((elem_fe_face->get_phi().size()),dim);
		U_sol.resize((elem_fe_face->get_phi().size()));




		// Because the traction is constant along the face, we call it now
		System * dummy;
		std::pair<bool,Gradient>  flux_pair = _bc_function(*dummy,Point(0,0,0), "u");

		// Get the gradient
		// Grab the dof indices for this element (global degree of freedom)
		std::vector<dof_id_type> indices;
		unsigned int u_var = _femsystem->variable_number ("u");

		_femsystem->get_dof_map().dof_indices(elem,indices,u_var);
		// Grab the values for these indices
		std::vector<Number> values;

		local_solution->get(indices,values);

			  // Loop over quadrature points
			  for (unsigned int qp = 0; qp != n_qpoints; qp++)
				{

					ElasticityTools::NHatMatrix(Nhat,dim,phi,qp);

					U_sol(0) = values[0];
					U_sol(1) = values[1];
					U_sol(2) = values[2];
					U_sol(3) = values[3];

					U_sol(4) = values[4];
					U_sol(5) = values[5];
					U_sol(6) = values[6];
					U_sol(7) = values[7];


					Nhat.vector_mult_transpose(U,U_sol);
				  // Update the elemental increment dR for each qp
				  qoi_computed += JxW_side[qp] * (U(0) * flux_pair.second(0) + U(1) * flux_pair.second(1));
				}
		  }


}

