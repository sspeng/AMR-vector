/* The libMesh Finite Element Library. */
/* Copyright (C) 2003  Benjamin S. Kirk */

/* This library is free software; you can redistribute it and/or */
/* modify it under the terms of the GNU Lesser General Public */
/* License as published by the Free Software Foundation; either */
/* version 2.1 of the License, or (at your option) any later version. */

/* This library is distributed in the hope that it will be useful, */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU */
/* Lesser General Public License for more details. */

/* You should have received a copy of the GNU Lesser General Public */
/* License along with this library; if not, write to the Free Software */
/* Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA */


// <h1>Vector Finite Element Example 1 - Solving an uncoupled Elasticity Problem</h1>
//
// This is the first vector FE example program.  It builds on
// the introduction_ex3 example program by showing how to solve a simple
// uncoupled TopOpt system using vector Lagrange elements.

// C++ include files that we need
#include <iostream>
#include <algorithm>
#include <math.h>
#include <time.h>
#include <utility>
// Basic include files needed for the mesh functionality.
#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/mesh_modification.h"
#include "libmesh/linear_implicit_system.h"
#include "libmesh/explicit_system.h"
#include "libmesh/equation_systems.h"
#include "libmesh/exodusII_io.h"
#include "libmesh/gmv_io.h"

// Define the Finite Element object.
#include "libmesh/fe.h"

// Define Gauss quadrature rules.
#include "libmesh/quadrature_gauss.h"

// Define useful datatypes for finite element
// matrix and vector components.
#include "libmesh/sparse_matrix.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/vector_value.h"
#include "libmesh/tensor_value.h"
#include "libmesh/elem.h"

// Define the DofMap, which handles degree of freedom
// indexing.
#include "libmesh/dof_map.h"

#include "libmesh/newton_solver.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/steady_solver.h"
#include "libmesh/system_norm.h"

// Data structures to handle Dirichlet BC applied in the regular way
#include "libmesh/boundary_info.h"


// AMR Data Structures
#include "libmesh/error_vector.h"
#include "libmesh/kelly_error_estimator_elasticity.h"
#include "libmesh/mesh_refinement.h"
#include "libmesh/uniform_refinement_estimator.h"
#include "libmesh/patch_recovery_error_estimator.h"
#include "libmesh/patch_recovery_error_estimator_elasticity.h"

// Adjoint Related includes
#include "libmesh/adjoint_residual_error_estimator.h"

// Exact solution
#include "libmesh/exact_solution.h"

#include "libmesh/petsc_vector.h"

//#include "solution_function.h"
//#include "elasticity_exact_solution.h"

// Functions to build the shape function matrices
#include "libmesh/elasticity_tools.h"

// Sensitivity Calculation related includes
#include "libmesh/parameter_vector.h"
#include "libmesh/sensitivity_data.h"

// Local includes
#include "femparameters.h"
#include "TopOpt.h"

#include "compliance.h"
#include "vonmises.h"
#include "compliance_traction.h"

// Include optimizer
#include "MMA.h"


#include <iostream>
#include <fstream>

// Bring in everything from the libMesh namespace
using namespace libMesh;


void write_output(EquationSystems &es,
                  unsigned int a_step, // The adaptive step count
                  std::string solution_type) // primal or adjoint solve
{
#ifdef LIBMESH_HAVE_GMV
  MeshBase &mesh = es.get_mesh();

  std::ostringstream file_name_gmv;
  file_name_gmv << solution_type
                << ".out.gmv."
                << std::setw(2)
                << std::setfill('0')
                << std::right
                << a_step;

  GMVIO(mesh).write_equation_systems
    (file_name_gmv.str(), es);


#endif
}

// Function to evaluate the entire constitutive response matrix. It is a matrix of 4-by-4 for 2-D only now.

void EvalElasticity(DenseMatrix<Real> & CMat) {

	Number _lambda = 12;
	Number _mu = 8;
	CMat(0,0) = _lambda + 2*_mu;
	CMat(3,3) = _lambda + 2*_mu;
	CMat(1,1) = _mu;
	CMat(1,2) = _mu;
	CMat(2,1) = _mu;
	CMat(2,2) = _mu;
	CMat(3,0) = _lambda;
	CMat(0,3) = _lambda;
}

std::pair<bool,Tensor> stress_function (const System& ,
                                       	 const Point& ,
                                       	 const std::string&,
                                       	 const dof_id_type & ,
                                       	 const ParameterVector & ,
                                       	 const DenseVector<Real> & Du,
                                       	 const unsigned int & dim ){
	DenseVector<Real> stress_vec;

	// Create CMat Matrices
	DenseMatrix<Real> CMat;
	CMat.resize(dim*dim,dim*dim);

	EvalElasticity(CMat);

	CMat.vector_mult(stress_vec,Du);

	Tensor stress;

	for (unsigned int i=0; i<dim; i++){
		for (unsigned int j=0; j<dim; j++){
			stress(i,j) = stress_vec(i+j*dim);
		}
	}

	std::pair<bool,Tensor> result(true,stress);
	return result;

}


// Set the parameters for the nonlinear and linear solvers to be used during the simulation

void set_system_parameters(TopOptSystem &system, FEMParameters &param)
{
  // Use analytical jacobians?
  system.analytic_jacobians() = param.analytic_jacobians;

  // Verify analytic jacobians against numerical ones?
  system.verify_analytic_jacobians = param.verify_analytic_jacobians;

  // Use the prescribed FE type
  system.fe_family() = param.fe_family[0];
  system.fe_order() = param.fe_order[0];

  // More desperate debugging options
  system.print_solution_norms = param.print_solution_norms;
  system.print_solutions      = param.print_solutions;
  system.print_residual_norms = param.print_residual_norms;
  system.print_residuals      = param.print_residuals;
  system.print_jacobian_norms = param.print_jacobian_norms;
  system.print_jacobians      = param.print_jacobians;

  // No transient time solver
  system.time_solver =
		  UniquePtr<TimeSolver>(new SteadySolver(system));

  // Nonlinear solver options
  {
    NewtonSolver *solver = new NewtonSolver(system);
    system.time_solver->diff_solver() = UniquePtr<DiffSolver>(solver);

    solver->quiet                       = param.solver_quiet;
    solver->max_nonlinear_iterations    = param.max_nonlinear_iterations;
    solver->minsteplength               = param.min_step_length;
    solver->relative_step_tolerance     = param.relative_step_tolerance;
    solver->relative_residual_tolerance = param.relative_residual_tolerance;
    solver->require_residual_reduction  = param.require_residual_reduction;
    solver->linear_tolerance_multiplier = param.linear_tolerance_multiplier;
    if (system.time_solver->reduce_deltat_on_diffsolver_failure)
      {
        solver->continue_after_max_iterations = true;
        solver->continue_after_backtrack_failure = true;
      }

    // And the linear solver options
    solver->max_linear_iterations       = param.max_linear_iterations;
    solver->initial_linear_tolerance    = param.initial_linear_tolerance;
    solver->minimum_linear_tolerance    = param.minimum_linear_tolerance;
  }

  // Set RAMP parameters;
  system.set_ramp_parameter(param.ramp_parameter);
  // Set filter parameter
  system.epsilon = param.filter_parameter;
  system.pnorm_parameter = param.pnorm_parameter;

  system.volume_fraction_constraint = param.volume_fraction_constraint;

  system.read_solution_from_file = param.read_solution_from_file;

  system.output_solution_to_file = param.output_solution_to_file;

  // Set elasticity constants
  system.set_elasticity_modules(param.lambda, param.mu);

  // Objective scaling function
  system.opt_scaling = param.opt_scaling;

  // Traction vertical force
  system.traction_force = param.traction_force;
}


void compute_von_mises(EquationSystems & es){


	  const MeshBase& mesh = es.get_mesh();

	  const unsigned int dim = mesh.mesh_dimension();

	  TopOptSystem& system = es.get_system<TopOptSystem>("Elasticity");

	  const unsigned int u_var = system.variable_number ("u");

	  const DofMap& dof_map = system.get_dof_map();
	  FEType fe_type = dof_map.variable_type(u_var);
	  UniquePtr<FEVectorBase> fe (FEVectorBase::build(dim, fe_type));
	  QGauss qrule (dim, fe_type.default_quadrature_order());
	  fe->attach_quadrature_rule (&qrule);

	  //const std::vector<Real>& JxW = fe->get_JxW();
	  //const std::vector<std::vector<RealTensor> >& dphi = fe->get_dphi();

	  // Also, get a reference to the vonmises. DofMap to set the value
	  ExplicitSystem& vonmises = es.get_system<ExplicitSystem>("VonMises");
	  const DofMap& stress_dof_map = vonmises.get_dof_map();
	  unsigned int vonMises_var = vonmises.variable_number ("vonmises");

	  // Storage for the stress dof indices on each element
	  std::vector<dof_id_type> dof_indices_vonmises;

	  // Same thing for the densities, we need to grab them
	  ExplicitSystem& densities = es.get_system<ExplicitSystem>("Densities");
	  const DofMap& densities_dof_map = densities.get_dof_map();
	  unsigned int density_var = densities.variable_number ("rho");

	  // Storage for the stress dof indices on each element
	  std::vector<dof_id_type> dof_indices_density;
	  std::vector<Number> density_value;

	  // To store the stress tensor on each element and the elasticity tensor
	  DenseMatrix<Number> CMat;
	  DenseVector<Number> stress_tensor, gradU, stress_deviatoric;
	  CMat.resize(dim*dim,dim*dim);
	  gradU.resize(dim*dim);
	  // Deviatoric stress has to be in three dimensions
	  stress_deviatoric.resize(3*3);

	  MeshBase::const_element_iterator       el     = mesh.active_local_elements_begin();
	  const MeshBase::const_element_iterator end_el = mesh.active_local_elements_end();

	  UniquePtr<DiffContext> con = system.build_context();
	  FEMContext &_femcontext = libmesh_cast_ref<FEMContext&>(*con);
	  system.init_context(_femcontext);

	  for ( ; el != end_el; ++el)
	    {
	      const Elem* elem = *el;
	      // We need the algebraic data
	      _femcontext.pre_fe_reinit(system, *el);
	      // And when asserts are on, we also need the FE so
	      // we can assert that the mesh data is of the right type.
	#ifndef NDEBUG
	      _femcontext.elem_fe_reinit();
	#endif

	      fe->reinit (elem);

	      // Reset deviatoric stress
	      stress_deviatoric.zero();

	      // Grab the gradient at qp = 0. It is constant so it doesn't matter
		  unsigned int qp = 0;
	      Tensor grad_u;
	      _femcontext.interior_gradient(u_var,qp,grad_u);

	      // Grab elasticity tensor
	      system.EvalElasticity(CMat);

	      // Get stress
	      CMat.vector_mult(stress_tensor, gradU);

	      // Get the value for the density field
	      densities_dof_map.dof_indices(elem, dof_indices_density, density_var);
	      // Get the value, stored in density
	      densities.current_local_solution->get(dof_indices_density, density_value);

	      // Apply the SIMP function
	      Number density_parameter = density_value[0];
	      // Apply SIMP function
	      system.SIMP_function(density_parameter);
	      stress_tensor *= density_parameter;

	      // Calculate deviatoric stress
	      Number stress_trace = stress_tensor(0) + stress_tensor(3);

	      // Copy values for the deviatoric stress
	      stress_deviatoric(0) = stress_tensor(0) - 1.0/3.0 * stress_trace;
	      stress_deviatoric(1) = stress_tensor(1);
	      stress_deviatoric(2) = 0;
	      stress_deviatoric(3) = stress_tensor(2);
	      stress_deviatoric(5) = stress_tensor(3) - 1.0/3.0 * stress_trace;
	      stress_deviatoric(6) = 0;
	      stress_deviatoric(7) = 0;
	      stress_deviatoric(8) = - 1.0/3.0 * stress_trace;

	      Number VonMises = sqrt(1.5*stress_deviatoric.dot(stress_deviatoric));

	      // Grab the index for the vonmises system and set the new value
	      stress_dof_map.dof_indices (elem, dof_indices_vonmises, vonMises_var);
	      dof_id_type dof_index = dof_indices_vonmises[0];
	      if( (vonmises.solution->first_local_index() <= dof_index) &&
	          (dof_index < vonmises.solution->last_local_index()) )
	        {
	    	  vonmises.solution->set(dof_index, VonMises);
	        }

	    }


	  // Should call close and update when we set vector entries directly
	  vonmises.solution->close();
	  vonmises.update();


}

#ifdef LIBMESH_ENABLE_AMR

UniquePtr<MeshRefinement> build_mesh_refinement(MeshBase &mesh,
                                              FEMParameters &param)
{
	UniquePtr<MeshRefinement> mesh_refinement(new MeshRefinement(mesh));
  mesh_refinement->coarsen_by_parents() = true;
  mesh_refinement->absolute_global_tolerance() = param.global_tolerance;
  mesh_refinement->nelem_target()      = param.nelem_target;
  mesh_refinement->refine_fraction()   = param.refine_fraction;
  mesh_refinement->coarsen_fraction()  = param.coarsen_fraction;
  mesh_refinement->coarsen_threshold() = param.coarsen_threshold;
  mesh_refinement->max_h_level()      = param.max_h_level;



  return mesh_refinement;
}

#endif // LIBMESH_ENABLE_AMR


// This is where we declare the error estimators to be built and used for
// mesh refinement. The adjoint residual estimator needs two estimators.
// One for the forward component of the estimate and one for the adjoint
// weighting factor. Here we use the Patch Recovery indicator to estimate both the
// forward and adjoint weights. The H1 seminorm component of the error is used
// as dictated by the weak form the Laplace equation.

UniquePtr<ErrorEstimator> build_error_estimator(FEMParameters &param, TopOptSystem &)
{
  UniquePtr<ErrorEstimator> error_estimator;

  if (param.indicator_type == "kelly")
    {
      std::cout<<"Using Kelly Error Estimator"<<std::endl;

      error_estimator.reset(new KellyErrorEstimatorElasticity);
    }
  else if (param.indicator_type == "adjoint_residual")
    {
      std::cout<<"Using Adjoint Residual Error Estimator with Patch Recovery Weights"<<std::endl<<std::endl;

      AdjointResidualErrorEstimator *adjoint_residual_estimator = new AdjointResidualErrorEstimator;

      error_estimator.reset (adjoint_residual_estimator);

      adjoint_residual_estimator->error_plot_suffix = "error.gmv";

      PatchRecoveryErrorEstimatorElasticity *p1 =
        new PatchRecoveryErrorEstimatorElasticity;
      p1->attach_elasticity_tensor(&EvalElasticity);
      p1->target_patch_size = param.target_patch_size;
      adjoint_residual_estimator->primal_error_estimator().reset(p1);

      PatchRecoveryErrorEstimatorElasticity *p2 =
        new PatchRecoveryErrorEstimatorElasticity;
      p2->attach_elasticity_tensor(&EvalElasticity);
      adjoint_residual_estimator->dual_error_estimator().reset(p2);

      adjoint_residual_estimator->primal_error_estimator()->error_norm.set_type(0, H1_SEMINORM);

      adjoint_residual_estimator->dual_error_estimator()->error_norm.set_type(0, H1_SEMINORM);
    }
  else
    libmesh_error_msg("Unknown indicator_type = " << param.indicator_type);

  return error_estimator;
}

std::pair<bool,Gradient> bc_function (const TopOptSystem& system,
                                        const Point&,
                                        const std::string& ){
	Gradient stress_flux(0,-1*system.traction_force);

	std::pair<bool,Gradient> result(true,stress_flux);
	return result;
}

Gradient body_force (const Point&,
					const std::string& ){

	Gradient bodyForce(0,-5);

	return bodyForce;
}




double myfunc(const std::vector<double> & x, std::vector<double> & , void * f_data){

	EquationSystems * equation_systems = (EquationSystems *) f_data;

	// Grab mesh
	//const MeshBase & mesh = equation_systems->get_mesh();
	// Grab systems
	TopOptSystem & system = equation_systems->get_system<TopOptSystem>("Elasticity");
	ExplicitSystem & densities = equation_systems->get_system<ExplicitSystem>("Densities");
    system.comm().barrier();

	// Transfer the value of the variables to the density field
	system.transfer_densities(densities, x, true);

	// Create QoIs set and indices
	QoISet qois;
	std::vector<unsigned int> qoi_indices;
	qoi_indices.push_back(0);
	qois.add_indices(qoi_indices);
	qois.set_weight(0, 1.0);

	// Copy solution to a file
	if (system.output_solution_to_file){
		std::ofstream myfile;
		myfile.open ("variables.txt", std::ofstream::out | std::ofstream::trunc);
		for (unsigned int i = 0; i < x.size(); i++)
			myfile << x[i]<<"\n";

		myfile.close();
	}

	// Solve system
	system.solve();

	std::cout << "System has: " << equation_systems->n_active_dofs()
			<< " degrees of freedom."
			<< std::endl;

	std::cout << "Linear solver converged at step: "
			<< (system.time_solver->diff_solver())->total_inner_iterations()
			<< std::endl;

	// Get a pointer to the primal solution vector
	//NumericVector<Number> &primal_solution = *system.solution;

	SensitivityData sensitivities(qois, system, system.get_parameter_vector());

    // We are about to solve the adjoint system, but before we do this we see the same preconditioner
    // flag to reuse the preconditioner from the forward solver
	system.get_linear_solver()->reuse_preconditioner(true);

	// Here we solve the adjoint problem inside the adjoint_qoi_parameter_sensitivity
	// function, so we have to set the adjoint_already_solved boolean to false
	system.set_adjoint_already_solved(false);

	// Compute the sensitivities
	system.adjoint_qoi_parameter_sensitivity(qois, system.get_parameter_vector(), sensitivities);

	// Now that we have solved the adjoint, set the adjoint_already_solved boolean to true, so we dont solve unneccesarily in the error estimator
	system.set_adjoint_already_solved(true);

    // Compute the approximate QoIs and write them out to the console
    system.calculate_qoi(qois);
    Number QoI_0_computed = system.get_QoI_value(0);

    // Transfer the gradient, apply filter and scale it
    system.filter_gradient();

    // Scale objective function
    QoI_0_computed *= system.opt_scaling;
    system.contador++;

    std::cout<<"Iteration # "<<system.contador<<" Function value = "<<QoI_0_computed<<std::endl;

    if (system.contador % 20 == 0){
    	std::cout<<"Printing densities"<<std::endl;
        std::ostringstream densities_filt;
        densities_filt << "dens_filt";
        NumericVector<Number> & densities_filtered = densities.get_vector(densities_filt.str());
    	densities_filtered.print();
    }

    system.comm().barrier();
	return QoI_0_computed;
}


double myvolumeconstraint(const std::vector<double> & x, std::vector<double> &, void * f_data){

	EquationSystems * equation_systems = (EquationSystems *) f_data;

	// Grab mesh
	//const MeshBase & mesh = equation_systems->get_mesh();
	// Grab systems
	TopOptSystem & system = equation_systems->get_system<TopOptSystem>("Elasticity");
	ExplicitSystem & densities = equation_systems->get_system<ExplicitSystem>("Densities");

    system.comm().barrier();

	// Transfer the value of the variables to the density field
	system.transfer_densities(densities, x, true);


	Number volume_constraint = system.calculate_filtered_volume_constraint(densities);

    system.comm().barrier();
    return volume_constraint;
}

int main (int argc, char** argv)
{

	// Initialize libraries.
	LibMeshInit init (argc, argv);


	// Skip adaptive examples on a non-adaptive libMesh build
	#ifndef LIBMESH_ENABLE_AMR
	libmesh_example_requires(false, "--enable-amr");
	#endif

	std::cout << "Started " << argv[0] << std::endl;

	// Make sure the general input file exists, and parse it
	{
	 std::ifstream i("general.in");
	 if (!i)
	   libmesh_error_msg('[' << init.comm().rank() << "] Can't find general.in; exiting early.");
	}
	GetPot infile("general.in");

	// Read in parameters from the input file
	FEMParameters param;
	param.read(infile);

	// Skip this default-2D example if libMesh was compiled as 1D-only.
	libmesh_example_requires(2 <= LIBMESH_DIM, "2D support");
	// Brief message to the user regarding the program name
	// and command line arguments.
	std::cout << "Running " << argv[0];
	// Printing put input options
	for (int i=1; i<argc; i++)
		std::cout << " " << argv[i];

	std::cout << std::endl << std::endl;

	// Skip this 2D example if libMesh was compiled as 1D-only.
	libmesh_example_requires(2 <= LIBMESH_DIM, "2D support");

	std::cout << "Reading in and building the mesh" << std::endl;
	const unsigned int dim = 2;

	// Create a mesh, with dimension to be overridden later, on the
	// default MPI communicator.
	Mesh mesh(init.comm(),dim);

	// And an object to refine it
	UniquePtr<MeshRefinement> mesh_refinement = build_mesh_refinement(mesh, param);

	// Read in the mesh
	mesh.read(param.domainfile.c_str());

	//bool p_norm_objectivefunction = false;

	// Create an equation systems object.
	EquationSystems equation_systems (mesh);


	// Build an auxiliary explicit system for the densities
	ExplicitSystem & densities = equation_systems.add_system<ExplicitSystem> ("Densities");

	// Add a zeroth order monomial variable that will represent the densities
	densities.add_variable("rho", CONSTANT, MONOMIAL);

	// Declare the Elasticity system and its variables.
	TopOptSystem& system = equation_systems.add_system<TopOptSystem> ("Elasticity");

	// New system to compute the Von Mises stress
//	ExplicitSystem & vonmises_system = equation_systems.add_system<ExplicitSystem> ("VonMises");
//	vonmises_system.add_variable("vonmises", CONSTANT, MONOMIAL);

	// Add objective function in the boundary
	ComplianceTraction compliancetractionqoi(&densities, &system);
	compliancetractionqoi.attach_flux_bc_function(&bc_function);
	compliancetractionqoi.assemble_qoi_sides = true;
	system.attach_advanced_qoi(&compliancetractionqoi);
	// Make sure we get the contributions to the adjoint RHS from the sides
	system.assemble_qoi_sides = true;
	// Add a Neumann condition at the right side of the second element,
	// the other tip of the L. Side ordering starts from zero on the lower side and goes counter clockwise
	// Attach Traction.
	system.attach_flux_bc_function(&bc_function);

	// Create a mesh refinement object to do the initial uniform refinements
	// on the coarse grid read in from lshaped.xda
	MeshRefinement initial_uniform_refinements(mesh);
	initial_uniform_refinements.uniformly_refine(param.coarserefinements);

	// Set its parameters
	set_system_parameters(system, param);



	// Attach body force.
//	system.attach_body_force(&body_force);
//	Compliance complianceqoi;
//
//	complianceqoi.attach_body_force(&body_force);
//
//	system.attach_advanced_qoi( &complianceqoi );



	// Create QoIs set and indices
	QoISet qois(system);
	std::vector<unsigned int> qoi_indices;
	qoi_indices.push_back(0);
	qois.add_indices(qoi_indices);
	qois.set_weight(0, 1.0);



//	VonMisesPnorm vonmises(&densities,&system,&qois);
//	system.p_norm_objectivefunction = true;
//	system.attach_advanced_qoi(&vonmises);
//	p_norm_objectivefunction = true;
//	system.assemble_qoi_sides = false;



	// Initialize the data structures for the equation system.
	equation_systems.init();

	mesh.print_info();


	// Give the system a pointer to the matrix assembly
	// function.  This will be called when needed by the
	// library.
	//system.attach_assemble_function (assemble_elasticity);

	std::cout<<"Print boundary info now"<<std::endl;
	mesh.boundary_info->print_info();

	// Set linear solver max iterations
	const int max_linear_iterations   = 2500;
	equation_systems.parameters.set<unsigned int>("linear solver maximum iterations")
	= max_linear_iterations;

	// Linear solver tolerance.
	equation_systems.parameters.set<Real>("linear solver tolerance") = 1e-7;

	// Prints information about the system to the screen.
	equation_systems.print_info();

	ParameterVector design_variables;


	const std::string indicator_type = "kelly";


	// Vector of variables
	std::vector<double> x(mesh.n_active_elem());

	// Copy solution to a file
	if (system.read_solution_from_file){
		  std::ifstream myfile ("variables.txt");

		  for (unsigned int i = 0; i<mesh.n_active_elem(); i++){
			  myfile >> x[i];
		  }
	}
	else{
		// Initial estimation
		for (std::vector<double>::iterator it = x.begin(); it != x.end(); it++)
			*it = param.initial_density;
	}


	// A refinement loop.
	for (unsigned int r_step=0; r_step<param.max_adaptivesteps; r_step++)
	{
		mesh.print_info();
		equation_systems.print_info();
		std::cout << "Beginning Solve " << r_step << std::endl;

		// Update the kernel with the new mesh
		std::cout<<"filtro bueno?"<<std::endl;
		system.update_kernel();
		std::cout<<"filtro bueno"<<std::endl;

		// For the first iteration, we copy the initial estimation "x" on to the
		// densities field
		if (r_step == 0) {
			bool filter = false;
			// Transfer the value of the variables to the density field
			// filter them if the boolean filter is true
			system.transfer_densities(densities, x, filter);
		}
		// Once we have done the refinement, we need to rewrite the vector "x" with
		// the densities field
		else{
			x.resize(mesh.n_active_elem());
			densities.solution->localize(x);
		}



		if (param.finite_difference){

			// Solve system
			system.solve();

			std::cout << "System has: " << equation_systems.n_active_dofs()
					<< " degrees of freedom."
					<< std::endl;

			std::cout << "Linear solver converged at step: "
					<< (system.time_solver->diff_solver())->total_inner_iterations()
					<< std::endl;

			// Get a pointer to the primal solution vector
			//NumericVector<Number> &primal_solution = *system.solution;

			SensitivityData sensitivities(qois, system, system.get_parameter_vector());

			// Make sure we get the contributions to the adjoint RHS from the sides
			system.assemble_qoi_sides = true;

			// We are about to solve the adjoint system, but before we do this we see the same preconditioner
			// flag to reuse the preconditioner from the forward solver
			system.get_linear_solver()->reuse_preconditioner(true);

			// Here we solve the adjoint problem inside the adjoint_qoi_parameter_sensitivity
			// function, so we have to set the adjoint_already_solved boolean to false
			system.set_adjoint_already_solved(false);

			// Here we calculate the QoIs before the sensitivity analysis because we need it for the
			// VonMises objective function. This QoI doesn't take into account the p-norm exponent.
			system.calculate_qoi(qois);

			// Compute the sensitivities
			system.adjoint_qoi_parameter_sensitivity(qois, system.get_parameter_vector(), sensitivities);

			// Now that we have solved the adjoint, set the adjoint_already_solved boolean to true, so we dont solve unneccesarily in the error estimator
			system.set_adjoint_already_solved(true);

			system.finite_differences_check(&densities,qois,sensitivities,system.get_parameter_vector());
			std::cout << "Terminamos FD check" << std::endl;
			//system.finite_differences_partial_derivatives_check(qois,system.get_parameter_vector(),sensitivities, p_norm_objectivefunction);
		}
		else{
			// Number of variables for the optimizer, equal to the number of active elements, changes with each refinement
			dof_id_type n_variables = mesh.n_active_elem();



			// Cast PetscVectors in order to obtain the raw Vec vectors
			PetscVector<Number> * x = dynamic_cast<PetscVector<Number> *>(densities.solution.get());
			// Create Optimizer
			MMA *mma = new MMA(n_variables,1.0,x->vec());
	        // Allocate after input
			PetscScalar * gx = new PetscScalar[1];


			PetscInt itr=0;
			PetscScalar ch = 1.0;
			// Error code for debugging
			PetscErrorCode ierr;
			double t1,t2;
			system.init_opt_vectors();
			while ((unsigned int) itr < param.maxeval && ch > param.xtol_rel){
					// Update iteration counter
					itr++;

					t1 = MPI_Wtime();

					// Filter densities

					system.kernel_filter_parallel.vector_mult(densities.get_vector("dens_filt"), *densities.solution.get());

					// Primal Analysis
					system.solve();

					// Get a pointer to the primal solution vector
					//NumericVector<Number> &primal_solution = *system.solution;

					// We are about to solve the adjoint system, but before we do this we see the same preconditioner
					// flag to reuse the preconditioner from the forward solver
					system.get_linear_solver()->reuse_preconditioner(true);

					// Here we solve the adjoint problem inside the adjoint_qoi_parameter_sensitivity
					// function, so we have to set the adjoint_already_solved boolean to false
					system.set_adjoint_already_solved(false);

					// Here we calculate the QoIs before the sensitivity analysis because we need it for the
					// VonMises objective function. This QoI doesn't take into account the p-norm exponent.
					system.calculate_qoi(qois);

					// Compute the sensitivities
					SensitivityData sensitivities(qois, system, system.get_parameter_vector());
					system.adjoint_qoi_parameter_sensitivity(qois, system.get_parameter_vector(), sensitivities);

					// Now that we have solved the adjoint, set the adjoint_already_solved boolean to true, so we dont solve unneccesarily in the error estimator
					system.set_adjoint_already_solved(true);

					// Filter sensitivities, now the sensitivities is in get_vector("grad_filt")
				    system.filter_gradient();

					// Get constraint value and the filtered sensitivities, stored in get_vector("vol_grad_filt")
					Number volume_constraint = system.calculate_filtered_volume_constraint(densities);


					// Cast PetscVectors in order to obtain the raw Vec vectors
					PetscVector<Number> & xmin 			= dynamic_cast<PetscVector<Number> & >(densities.get_vector("xmin"));
					PetscVector<Number> & xmax 			= dynamic_cast<PetscVector<Number> & >(densities.get_vector("xmax"));
					PetscVector<Number> & xold 			= dynamic_cast<PetscVector<Number> & >(densities.get_vector("xold"));
					PetscVector<Number> & gradient 		= dynamic_cast<PetscVector<Number> & >(densities.get_vector("grad"));
					PetscVector<Number> & vol_gradient 	= dynamic_cast<PetscVector<Number> & >(densities.get_vector("vol_grad_filt"));
					gx[0] = volume_constraint;
					// Sets outer movelimits on design variables
					ierr = mma->SetOuterMovelimit(param.minimum_density,
													param.maximum_density,
													param.movlim,
													x->vec(),xmin.vec(),xmax.vec()); CHKERRQ(ierr);

					// I need to copy the content of vol_gradient into dgdx
					VecCopy(vol_gradient.vec(), system.dgdx[0]);

					// Update design by MMA
					ierr = mma->Update(x->vec(),gradient.vec(),gx,system.dgdx,xmin.vec(),xmax.vec()); CHKERRQ(ierr);
					// Update the solution field in densities
					densities.update();
					// Inf norm on the design change
					ch = mma->DesignChange(x->vec(),xold.vec());


					// stop timer
					t2 = MPI_Wtime();
					// Print to screen
					PetscPrintf(PETSC_COMM_WORLD,"It.: %i, obj.: %f, g[0]: %f, ch.: %f, time: %f\n",
								itr,system.get_QoI_value(0),volume_constraint, ch,t2-t1);

					if (itr % 10 == 0) {
						densities.solution->print_global();
					}

			  }

			  // Clean up
			  delete mma;
			  delete [] gx ;
			  // Reinit the array Vec dgdx
			  if (system.dgdx!=NULL){ VecDestroyVecs(1.0,&system.dgdx); }

		}

		if (r_step+1 != param.max_adaptivesteps)
		{
		  std::cout << "  Refining the mesh..." << std::endl;

		   //Adaptively refine based on reaching an error tolerance
		  if(param.global_tolerance >= 0. && param.nelem_target == 0.)
		  {
			  // Now we construct the data structures for the mesh refinement process
			  ErrorVector error;

			  // Build an error estimator object
			  UniquePtr<ErrorEstimator> error_estimator =
				build_error_estimator(param,system);


			  // Estimate the error in each element using the Adjoint Residual or Kelly error estimator
			  error_estimator->estimate_error(system, error);

			  mesh_refinement->flag_elements_by_error_tolerance (error);

			  mesh_refinement->refine_and_coarsen_elements();
		  }

		  // This call reinitializes the \p EquationSystems object for
		  // the newly refined mesh.  One of the steps in the
		  // reinitialization is projecting the \p solution,
		  // \p old_solution, etc... vectors from the old mesh to
		  // the current one.
		  equation_systems.reinit();
		  std::cout << "Refined mesh to "
					<< mesh.n_active_elem()
					<< " active elements and "
					<< equation_systems.n_active_dofs()
					<< " active dofs." << std::endl;

		  system.init_opt_vectors();
		}
	}

	#ifdef LIBMESH_HAVE_EXODUS_API
	// Use single precision in this case (reduces the size of the exodus file)
	ExodusII_IO exo_io(mesh, /*single_precision=*/true);

	// First plot the displacement field using a nodal plot
	std::set<std::string> system_names;
	system_names.insert("Elasticity");
		exo_io.write_equation_systems("displacement_and_densities.e",equation_systems,&system_names);

	// then append element-based discontinuous plots of the stresses
		exo_io.write_element_data(equation_systems);
	#endif


	// All done.
	return 0;
}

