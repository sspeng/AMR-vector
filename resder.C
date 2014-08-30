#include "resder.h"
#include "TopOpt.h"
#include "libmesh/explicit_system.h"
#include "libmesh/equation_systems.h"
#include "libmesh/elasticity_tools.h"

// Define the DofMap, which handles degree of freedom
// indexing.
#include "libmesh/dof_map.h"


using namespace libMesh;

void ResidualDerivative::init_context(DiffContext &context, TopOptSystem & sys)
{
	  FEMContext &c = libmesh_cast_ref<FEMContext&>(context);
	  sys.init_context(context);
	  // Get the Residual Derivative vector, we have added it calling ResDerContributions
	  NumericVector<Number> &res_der_vector = sys.get_resder_rhs(0);
	  // Add this vector to the vectors that diff context should localize
	  c.add_localized_vector(res_der_vector, sys);
}

// We only have one QoI, so we don't bother checking the qois argument
// to see if it was requested from us
void ResidualDerivative::element_res_derivative (DiffContext &context, TopOptSystem & sys)
{
	FEMContext &c = libmesh_cast_ref<FEMContext&>(context);

	// First we get some references to cell-specific data that
	// will be used to assemble the linear system.
	FEVectorBase* elem_fe = NULL;
	c.get_element_fe( 0, elem_fe );

	// Grab element stiffness matrix
	DenseMatrix<Number> & K  = c.get_elem_jacobian();
	// Grab element solution vector
	const DenseVector<Number> & U  = c.get_elem_solution();

	// Get a pointer to the residual derivative vector
	const NumericVector<Number> &res_der_vector = sys.get_resder_rhs(0);
	// Grab element contribution to the residual derivative vector
	DenseSubVector<Number> & dRdp = c.get_localized_subvector(res_der_vector,0);




	// Get system
	ExplicitSystem& aux_system = sys.get_equation_systems().get_system<ExplicitSystem>("Densities");
	// Array to hold the dof indices
	std::vector<dof_id_type> densities_index;
	// Array to hold the values
	std::vector<Number> density;
	// Get dof indices
	aux_system.get_dof_map().dof_indices(&c.get_elem(), densities_index, 0);
	// Get the value, stored in u_undefo
	aux_system.current_local_solution->get(densities_index, density);
	// Apply ramp
	Number density_orig = density[0];
	sys.ramp_deriv(density[0]);
	// The stiffness matrix K comes factored by the density "ramped", we calculate
	// it again to "unramp it"
	sys.ramp(density_orig);

	for (unsigned int i=0; i<K.n(); i++)
	  for (unsigned int j=0; j<K.m(); j++){
		  dRdp(i) += 1.0/density_orig * density[0]*K(i,j)*U(j);
	  }

//	// Element Jacobian * quadrature weights for interior integration
//	const std::vector<Real> &JxW = elem_fe->get_JxW();
//
//	// The element shape functions evaluated at the quadrature points.
//	// Notice the shape functions are a vector rather than a scalar.
//	const std::vector<std::vector<RealTensor> >& dphi = elem_fe->get_dphi();
//	unsigned int n_qpoints = c.get_element_qrule().n_points();
//	const unsigned int dim = c.get_dim();
//	DenseMatrix<Number> DNhat, CMat;
//	CMat.resize(dim*dim, dim*dim);
//	DNhat.resize((elem_fe->get_phi().size()),dim*dim);
//	DenseVector<Real> gradU, stress, stress_temp;
//	gradU.resize(dim*dim);
//	unsigned int n_t_dofs = dRdp.size();
//
//
//	for (unsigned int qp=0; qp<n_qpoints; qp++){
//	  // Write shape function matrices
//	  ElasticityTools::DNHatMatrix( DNhat, dim, dphi, qp);
//	  // Evaluate elasticity tensor
//	  sys.EvalElasticity(CMat);
//
//	  // Get gradient, we need it for the residual
//	  Tensor grad_u;
//
//	 c.interior_gradient( 0, qp, grad_u );
//
//	 gradU(0) = grad_u(0,0);
//	 gradU(1) = grad_u(1,0);
//	 gradU(2) = grad_u(0,1);
//	 gradU(3) = grad_u(1,1);
//
//	 // Calculate stress
//	 CMat.vector_mult(stress_temp,gradU);
//
//	 // Factor stress with the density
//	 stress_temp *= density[0];
//
//	 // Multiplied with the shape derivative matrix because of the
//	 // contribution of the test variable inthe residual form
//	 DNhat.vector_mult(stress,stress_temp);
//
//
//	 for (unsigned int i=0; i<n_t_dofs; i++)
//		 dRdp(i) += JxW[qp]*stress(i);
//	 //dRdp.add(JxW[qp],stress);
//
//
//	}

}
