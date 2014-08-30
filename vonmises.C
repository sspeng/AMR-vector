#include "vonmises.h"
#include "libmesh/elasticity_tools.h"
#include <math.h>
#include "TopOpt.h"
#include "libmesh/dof_map.h"
using namespace libMesh;

VonMisesPnorm::VonMisesPnorm(ExplicitSystem * densities, TopOptSystem * femsystem, QoISet * qois){

	_densities = densities;
	_femsystem = femsystem;

	_qois = qois;

	unsigned int dim = _femsystem->get_mesh().mesh_dimension();

	// Declare and resize the matrices involced in this product
	DenseMatrix<Number> Identity, Identity_cross, CMat;
	CMat.resize(dim*dim,dim*dim);
	Identity.resize(dim*dim,dim*dim);
	Identity_cross.resize(dim*dim,dim*dim);

	Identity(0,0) = Identity(1,1) = Identity(2,2) = Identity(3,3) = 1.0;
	Identity_cross(0,0) = Identity_cross(3,0) = Identity_cross(0,3) = Identity_cross(3,3) = 1.0;

	// Build the Pdev
	Identity.add(-1.0/3.0,Identity_cross);

	// Multiply
	_femsystem->EvalElasticity(CMat);
	Identity.right_multiply(CMat);

	// Copy to keep it in PdevCmat
	PdevCMat = Identity;

	von_mises_index = 0;
}

void VonMisesPnorm::init_qoi( std::vector<Number>& sys_qoi )
{
  //Only 1 qoi to worry about
	if (sys_qoi.size() == 0){
		sys_qoi.resize(1);
		von_mises_index = 0;
	}
	else if(sys_qoi.size() == 1){
		sys_qoi.resize(2);
		von_mises_index = 1;
	}
	else if(sys_qoi.size() == 2){
		sys_qoi.resize(3);
		von_mises_index = 2;
	}
  return;
}


// We only have one QoI, so we don't bother checking the qois argument
// to see if it was requested from us
void VonMisesPnorm::element_qoi (DiffContext &context,
                              const QoISet & /* qois */ )

{
	FEMContext &c = libmesh_cast_ref<FEMContext&>(context);

	FEVectorBase* elem_fe = NULL;
	c.get_element_fe( 0, elem_fe );

	const unsigned int dim = c.get_dim();

	Number VonMises = 0.;

	/*
	* Calculating Von Mises stress
	*/

	// Calculate stresses first, we only pick one gauss point, because the gradient is constant
	DenseMatrix<Number> CMat;
	DenseVector<Number> gradU, stress_vector;
	CMat.resize(dim*dim,dim*dim);
	gradU.resize(dim*dim);
	_femsystem->EvalElasticity(CMat);

	unsigned int qp = 0;
	// Get the solution value at the quadrature point
	Tensor grad_u;
	c.interior_gradient( 0, qp, grad_u );

	gradU(0) = grad_u(0,0);
	gradU(1) = grad_u(1,0);
	gradU(2) = grad_u(0,1);
	gradU(3) = grad_u(1,1);

	CMat.vector_mult(stress_vector,gradU);

	// Before adding the contribution, we need to calculate the SIMP function.
	Elem & elem = c.get_elem();
	std::vector<dof_id_type> densities_index;
	std::vector<Number> density;
	unsigned int density_var = _densities->variable_number ("rho");
	// Get dof indices
	const DofMap& dof_map_densities = _densities->get_dof_map();
	dof_map_densities.dof_indices(&elem, densities_index, density_var);
	// Get the value, stored in density
	_densities->current_local_solution->get(densities_index, density);

	Number density_parameter = density[0];
	// Apply SIMP function
	SIMP_function(density_parameter);

	stress_vector *= density_parameter;

	// Calculate deviatoric stress
	Number stress_trace = stress_vector(0) + stress_vector(3);

	DenseVector<Number> stress_deviatoric = stress_vector;
	stress_deviatoric(0) = stress_deviatoric(0) - 1.0/3.0 * stress_trace;
	stress_deviatoric(3) = stress_deviatoric(3) - 1.0/3.0 * stress_trace;

	VonMises = sqrt(1.5*stress_deviatoric.dot(stress_deviatoric));

	c.get_qois()[von_mises_index] = c.get_qois()[von_mises_index] + c.get_elem().volume()*pow(VonMises,_femsystem->pnorm_parameter);

}

// We only have one QoI, so we don't bother checking the qois argument
// to see if it was requested from us
void VonMisesPnorm::element_qoi_derivative (DiffContext &context,
                                         const QoISet & /* qois */)
{
	FEMContext &c = libmesh_cast_ref<FEMContext&>(context);

	FEVectorBase* elem_fe = NULL;
	c.get_element_fe( 0, elem_fe );

	const unsigned int dim = c.get_dim();


	// Derivatives of the rlement basis functions
	const std::vector<std::vector<RealTensor> > &dphi = elem_fe->get_dphi();

	Number VonMises = 0.;

	/*
	* Calculating Von Mises stress
	*/

	// Calculate stresses first, we only pick one gauss point, because the gradient is constant for bilinear elements
	DenseMatrix<Number> CMat, DNhat, TempMatrix;
	DenseVector<Number> gradU, stress_vector, temp_vector;
	CMat.resize(dim*dim,dim*dim);
	DNhat.resize((elem_fe->get_phi().size()),dim*dim);
	gradU.resize(dim*dim);
	this->_femsystem->EvalElasticity(CMat);

	unsigned int qp = 0;
	// Get the solution value at the quadrature point
	Tensor grad_u;
	c.interior_gradient( 0, qp, grad_u );

	gradU(0) = grad_u(0,0);
	gradU(1) = grad_u(1,0);
	gradU(2) = grad_u(0,1);
	gradU(3) = grad_u(1,1);

	// Get stresses
	CMat.vector_mult(stress_vector,gradU);

	// Before adding the contribution, we need to calculate the SIMP function.
	Elem & elem = c.get_elem();
	std::vector<dof_id_type> densities_index;
	std::vector<Number> density;
	unsigned int density_var = _densities->variable_number ("rho");
	// Get dof indices
	const DofMap& dof_map_densities = _densities->get_dof_map();
	dof_map_densities.dof_indices(&elem, densities_index, density_var);
	// Get the value, stored in density
	_densities->current_local_solution->get(densities_index, density);

	Number density_parameter = density[0];
	// Apply SIMP function
	SIMP_function(density_parameter);

	stress_vector *= density_parameter;

	// Calculate deviatoric stress
	Number stress_trace = stress_vector(0) + stress_vector(3);

	DenseVector<Number> stress_deviatoric = stress_vector;
	stress_deviatoric(0) = stress_vector(0) - 1.0/3.0 * stress_trace;
	stress_deviatoric(3) = stress_vector(3) - 1.0/3.0 * stress_trace;

	VonMises = sqrt(1.5*stress_deviatoric.dot(stress_deviatoric));

	stress_deviatoric *= 1.0/VonMises;

	ElasticityTools::DNHatMatrix(DNhat,dim,dphi,qp);

	// Fill the QoI RHS corresponding to this QoI. Since this is the 0th QoI
	// we fill in the [0][i] subderivatives, i corresponding to the variable index.
	// Our system has only one variable, so we only have to fill the [0][0] subderivative
	// DO NOT CONFUSE variable with design variable, here variable means displacement, velocity or other
	// implicit quantity
	std::vector<DenseVector<Number> > &Q = c.get_qoi_derivatives();

	TempMatrix = PdevCMat;

	TempMatrix.right_multiply_transpose(DNhat);

	TempMatrix.vector_mult_transpose(temp_vector, stress_deviatoric);

	temp_vector *= 1.5;

	temp_vector *= density_parameter;

	//Grab the QoI because we need it for the computation of the derivatives
	/*
	 * Need to check that the QoI has been calculated previously
	 */
    Number pnorm_parameter = _femsystem->pnorm_parameter;
    // Coefficient in front to take into account all the elements
    Number VonMisesQoI = _femsystem->get_QoI_value(von_mises_index);
    std::cout<<"VonMisesQoI = "<<VonMisesQoI<<std::endl;
    Number von_mises_sum = 1.0 / pnorm_parameter * pow(VonMisesQoI, 1.0/pnorm_parameter - 1.0);
    // Coefficients from the derivatives of the same element
    von_mises_sum *= c.get_elem().volume()*pnorm_parameter*pow(VonMises,pnorm_parameter - 1.0);

    // Add the contribution
    Q[von_mises_index].add(von_mises_sum, temp_vector);
}

void VonMisesPnorm::element_qoi_derivative_parameter (DiffContext &context, Number & parameter_derivative, Number density_element){

	FEMContext &c = libmesh_cast_ref<FEMContext&>(context);

	FEVectorBase* elem_fe = NULL;
	c.get_element_fe( 0, elem_fe );

	const unsigned int dim = c.get_dim();


	// Derivatives of the rlement basis functions
	const std::vector<std::vector<RealTensor> > &dphi = elem_fe->get_dphi();

	Number VonMises = 0.;

	/*
	* Calculating Von Mises stress
	*/

	// Calculate stresses first, we only pick one gauss point, because the gradient is constant for bilinear elements
	DenseMatrix<Number> CMat, DNhat, TempMatrix;
	DenseVector<Number> gradU, stress_vector, temp_vector;
	CMat.resize(dim*dim,dim*dim);
	DNhat.resize((elem_fe->get_phi().size()),dim*dim);
	gradU.resize(dim*dim);
	this->_femsystem->EvalElasticity(CMat);

	unsigned int qp = 0;
	// Get the solution value at the quadrature point
	Tensor grad_u;
	c.interior_gradient( 0, qp, grad_u );

	gradU(0) = grad_u(0,0);
	gradU(1) = grad_u(1,0);
	gradU(2) = grad_u(0,1);
	gradU(3) = grad_u(1,1);

	// Get stresses
	CMat.vector_mult(stress_vector,gradU);

	Number density_element_function = density_element;
	// Apply SIMP function
	SIMP_function(density_element_function);

	stress_vector *= density_element_function;

	// Calculate deviatoric stress
	Number stress_trace = stress_vector(0) + stress_vector(3);

	DenseVector<Number> stress_deviatoric = stress_vector;
	stress_deviatoric(0) = stress_vector(0) - 1.0/3.0 * stress_trace;
	stress_deviatoric(3) = stress_vector(3) - 1.0/3.0 * stress_trace;

	VonMises = sqrt(1.5*stress_deviatoric.dot(stress_deviatoric));

	stress_deviatoric *= 1.0/VonMises;

	// Fill the QoI RHS corresponding to this QoI. Since this is the 0th QoI
	// we fill in the [0][i] subderivatives, i corresponding to the variable index.
	// Our system has only one variable, so we only have to fill the [0][0] subderivative
	// DO NOT CONFUSE variable with design variable, here variable means displacement, velocity or other
	// implicit quantity
	std::vector<DenseVector<Number> > &Q = c.get_qoi_derivatives();

	PdevCMat.vector_mult_transpose(temp_vector, stress_deviatoric);

	temp_vector *= 1.5;

	Number derivative = temp_vector.dot(gradU);

	//Grab the QoI because we need it for the computation of the derivatives
	/*
	 * Need to check that the QoI has been calculated previously
	 */
    Number pnorm_parameter = _femsystem->pnorm_parameter;
    // Coefficient in front to take into account all the elements
    Number von_mises_sum = 1.0 / pnorm_parameter * pow(_femsystem->get_QoI_value(von_mises_index), 1.0/pnorm_parameter - 1.0);
    // Coefficients from the derivatives of the same element
    von_mises_sum *= c.get_elem().volume()*pnorm_parameter*pow(VonMises,pnorm_parameter - 1.0);


    SIMP_function_derivative(density_element);

    parameter_derivative = von_mises_sum * density_element * derivative;


}
//
//void VonMisesPnorm::SIMP_function(Number & density){
//
//	Number phi = density/(0.3*(1 - density) + density);
//
//	density = phi;
//}
//
//void VonMisesPnorm::SIMP_function_derivative(Number & density){
//
//	Number phi = 0.3/((0.3*(1 - density) + density)*(0.3*(1 - density) + density));
//
//	density = phi;
//}
