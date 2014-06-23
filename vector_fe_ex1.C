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


// <h1>Vector Finite Element Example 1 - Solving an uncoupled Poisson Problem</h1>
//
// This is the first vector FE example program.  It builds on
// the introduction_ex3 example program by showing how to solve a simple
// uncoupled Poisson system using vector Lagrange elements.

// C++ include files that we need
#include <iostream>
#include <algorithm>
#include <math.h>
#include <time.h>
// Basic include files needed for the mesh functionality.
#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/linear_implicit_system.h"
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

// Data structures to handle Dirichlet BC applied in the regular way
#include "libmesh/boundary_info.h"
#include "libmesh/zero_function.h"
#include "libmesh/dirichlet_boundaries.h"

// AMR Data Structures
#include "libmesh/error_vector.h"
#include "libmesh/kelly_error_estimator.h"
#include "libmesh/mesh_refinement.h"
#include "libmesh/uniform_refinement_estimator.h"

// Exact solution
#include "libmesh/exact_solution.h"

// Bring in everything from the libMesh namespace
using namespace libMesh;

// Function prototype.  This is the function that will assemble
// the linear system for our Poisson problem.  Note that the
// function will take the  EquationSystems object and the
// name of the system we are assembling as input.  From the
//  EquationSystems object we have access to the  Mesh and
// other objects we might need.
void assemble_poisson(EquationSystems& es,
                      const std::string& system_name);

//Functions to assemble the matrices DNHat and Nhat. These matrices contain the shape function information. They are built using the kronecker product of either the shape
// function N or its derivatives DN and the identity matrix
void DNHatMatrix(DenseMatrix<Real> & DNhat, const unsigned int & dim, int & phiSize, const std::vector<std::vector<RealTensor> > & dphi, unsigned int & qp);

void NHatMatrix(DenseMatrix<Real> & Nhat, const unsigned int & dim, int & phiSize, const std::vector<std::vector<RealGradient> >& phi, unsigned int & qp);

// Function to evaluate the entire constitutive response matrix. It is a matrix of 4-by-4 for 2-D only now.
void EvalElasticity(DenseMatrix<Real> & CMat, const std::vector<Point>& q_point, const unsigned int & dim, unsigned int & qp);


Number exact_solution(const Point& p,
                      const Parameters&,   // EquationSystem parameters, not needed
                      const std::string&,  // sys_name, not needed
                      const std::string&); // unk_name, not needed);

// Prototype for calculation of the gradient of the exact solution.
Gradient exact_derivative(const Point& p,
                          const Parameters&,   // EquationSystems parameters, not needed
                          const std::string&,  // sys_name, not needed
                          const std::string&); // unk_name, not needed);


int main (int argc, char** argv)
{
  // Initialize libraries.
  LibMeshInit init (argc, argv);



  // Brief message to the user regarding the program name
  // and command line arguments.
  std::cout << "Running " << argv[0];
  // Printing put input options
  for (int i=1; i<argc; i++)
    std::cout << " " << argv[i];

  std::cout << std::endl << std::endl;

  // Skip this 2D example if libMesh was compiled as 1D-only.
  libmesh_example_requires(2 <= LIBMESH_DIM, "2D support");


  const unsigned int dim = 2;
  // Create a mesh, with dimension to be overridden later, on the
  // default MPI communicator.
  Mesh mesh(init.comm(),dim);

  // Use the MeshTools::Generation mesh generator to create a uniform
  // 2D grid on the rectangle [0 , 1] x [0 , 0.2].  We instruct the mesh generator
  // to build a mesh of 50x10 QUAD9 elements.

  MeshTools::Generation::build_square (mesh,
                                              50, 10,
                                              0., 1.,
                                              0., 0.2,
                                              QUAD9);

  // Print information about the mesh to the screen.
  mesh.print_info();

  // Create an equation systems object.
  EquationSystems equation_systems (mesh);

  // Declare the Poisson system and its variables.
  // The Poisson system is another example of a steady system.
  LinearImplicitSystem& system = equation_systems.add_system<LinearImplicitSystem> ("Poisson");

  // Adds the variable "u" to "Poisson".  "u"
  // will be approximated using second-order approximation
  // using vector Lagrange elements. Since the mesh is 2-D, "u" will
  // have two components.
  unsigned int u_var = system.add_variable("u", SECOND, LAGRANGE_VEC);

  // Give the system a pointer to the matrix assembly
  // function.  This will be called when needed by the
  // library.
  system.attach_assemble_function (assemble_poisson);

  const unsigned int n_components =
		  system.variable(0).n_components();

  std::cout<<"Number of components for the first and only variable "<<n_components<<std::endl;


  // Construct a Dirichlet boundary condition object
  // We impose a "clamped" boundary condition on the
  // "left" boundary, i.e. bc_id = 3
  std::set<boundary_id_type> boundary_ids;
  boundary_ids.insert(3);

  // Create a vector storing the variable numbers which the BC applies to
  std::vector<unsigned int> variables(2);
  variables[0] = u_var;

  // Create a ZeroFunction to initialize dirichlet_bc
  ZeroFunction<> zf;

  DirichletBoundary dirichlet_bc(boundary_ids,
                                 variables,
                                 &zf);

  // We must add the Dirichlet boundary condition _before_
  // we call equation_systems.init()
  system.get_dof_map().add_dirichlet_boundary(dirichlet_bc);


  // Define the mesh refinement object that takes care of adaptively
  // refining the mesh.
  MeshRefinement mesh_refinement(mesh);

  // These parameters determine the proportion of elements that will
  // be refined and coarsened. Any element within 30% of the maximum
  // error on any element will be refined, and any element within 30%
  // of the minimum error on any element might be coarsened
  mesh_refinement.refine_fraction()  = 0.7;
  mesh_refinement.coarsen_fraction() = 0.3;
  // We won't refine any element more than 5 times in total
  mesh_refinement.max_h_level()      = 5;


  // Initialize the data structures for the equation system.
  equation_systems.init();

  // Set linear solver max iterations
  const int max_linear_iterations   = 2500;
  equation_systems.parameters.set<unsigned int>("linear solver maximum iterations")
    = max_linear_iterations;

  // Linear solver tolerance.
  equation_systems.parameters.set<Real>("linear solver tolerance") = 1e-7;


  // Refinement parameters
  const unsigned int max_r_steps = 5; // Refine the mesh 5 times

  // Prints information about the system to the screen.
  equation_systems.print_info();

//   Construct ExactSolution object and attach solution functions
//   ExactSolution exact_sol(equation_systems);
//   exact_sol.attach_exact_value(exact_solution);
//   exact_sol.attach_exact_deriv(exact_derivative);

  // Solve the system "Poisson".  Note that calling this
  // member will assemble the linear system and invoke
  // the default numerical solver.  With PETSc the solver can be
  // controlled from the command line.  For example,
  // you can invoke conjugate gradient with:
  //
  // ./vector_fe_ex1 -ksp_type cg
  //
  // You can also get a nice X-window that monitors the solver
  // convergence with:
  //
  // ./vector_fe_ex1 -ksp_xmonitor
  //
  // if you linked against the appropriate X libraries when you
  // built PETSc.


  const std::string indicator_type = "kelly";

  // A refinement loop.
  for (unsigned int r_step=0; r_step<max_r_steps; r_step++)
  {
      std::cout << "Beginning Solve " << r_step << std::endl;

	  system.solve();

      std::cout << "System has: " << equation_systems.n_active_dofs()
                << " degrees of freedom."
                << std::endl;

      std::cout << "Linear solver converged at step: "
                << system.n_linear_iterations()
                << ", final residual: "
                << system.final_linear_residual()
                << std::endl;

//      // Compute the error.
//      exact_sol.compute_error("Poisson", "u");
//
//      // Print out the error values
//      std::cout << "L2-Error is: "
//                << exact_sol.l2_error("Poisson", "u")
//                << std::endl;
//      std::cout << "H1-Error is: "
//                << exact_sol.h1_error("Poisson", "u")
//                << std::endl;


      // Possibly refine the mesh
      if (r_step+1 != max_r_steps)
        {
          std::cout << "  Refining the mesh..." << std::endl;
		  // The \p ErrorVector is a particular \p StatisticsVector
		  // for computing error information on a finite element mesh.
		  ErrorVector error;
		  if (indicator_type == "uniform")
			{
			  // Error indication based on uniform refinement
			  // is reliable, but very expensive.
			  UniformRefinementEstimator error_estimator;

			  error_estimator.estimate_error (system, error);
			}
		  else
			{
			  libmesh_assert_equal_to (indicator_type, "kelly");

			  // The Kelly error estimator is based on
			  // an error bound for the Poisson problem
			  // on linear elements, but is useful for
			  // driving adaptive refinement in many problems
			  KellyErrorEstimator error_estimator;

			  error_estimator.estimate_error (system, error);
			}

          // This takes the error in \p error and decides which elements
          // will be coarsened or refined.  Any element within 20% of the
          // maximum error on any element will be refined, and any
          // element within 10% of the minimum error on any element might
          // be coarsened. Note that the elements flagged for refinement
          // will be refined, but those flagged for coarsening _might_ be
          // coarsened.
          mesh_refinement.flag_elements_by_error_fraction (error);

          // This call actually refines and coarsens the flagged
          // elements.
          mesh_refinement.refine_and_coarsen_elements();

          // This call reinitializes the \p EquationSystems object for
          // the newly refined mesh.  One of the steps in the
          // reinitialization is projecting the \p solution,
          // \p old_solution, etc... vectors from the old mesh to
          // the current one.
    	  equation_systems.reinit();
        }
  }


#ifdef LIBMESH_HAVE_EXODUS_API
  ExodusII_IO(mesh).write_equation_systems( "out_parallel.e", equation_systems);
#endif

#ifdef LIBMESH_HAVE_GMV
  GMVIO(mesh).write_equation_systems( "out.gmv", equation_systems);
#endif

  // All done.
  return 0;
}

Number exact_solution(const Point& p,
                           const Parameters&,  // parameters, not needed
                           const std::string&, // sys_name, not needed
                           const std::string&) // unk_name, not needed
  {
    const Real x = p(0);
    const Real y = p(1);

    Gradient solution;
    solution(0) = 256.*(x-x*x)*(x-x*x)*(y-y*y)*(y-y*y);
    solution(1) = 256.*(x-x*x)*(x-x*x)*(y-y*y)*(y-y*y);

    return 25600.*(x-x*x)*(x-x*x)*(y-y*y)*(y-y*y);
  }

Gradient exact_derivative(const Point& p,
                              const Parameters&,  // parameters, not needed
                              const std::string&, // sys_name, not needed
                              const std::string&) // unk_name, not needed
 {
   const Real x = p(0);
   const Real y = p(1);

   Tensor gradu;
   gradu(0,0) = 256.*2.*(x-x*x)*(1-2*x)*(y-y*y)*(y-y*y);
   gradu(0,1) = 256.*2.*(x-x*x)*(1-2*x)*(y-y*y)*(y-y*y);
   gradu(1,0) = 256.*2.*(x-x*x)*(x-x*x)*(y-y*y)*(1-2*y);
   gradu(1,1) = 256.*2.*(x-x*x)*(x-x*x)*(y-y*y)*(1-2*y);

   //FIXME We're trying to see what happens when we calculate the error with a exact solution if we have the default exact_solution and exact_derivative
   Gradient graduTemp;
   graduTemp(0) = 256.*2.*(x-x*x)*(1-2*x)*(y-y*y)*(y-y*y);
   graduTemp(1) = 256.*2.*(x-x*x)*(x-x*x)*(y-y*y)*(1-2*y);

   return graduTemp;
 }


void DNHatMatrix(DenseMatrix<Real> & DNhat, const unsigned int & dim, int & phiSize, const std::vector<std::vector<RealTensor> >& dphi, unsigned int & qp) {
	DNhat.resize(phiSize,dim*dim);
	for (unsigned int ii = 0; ii<phiSize/dim; ii++) {
		for (unsigned int jj = 0; jj<dim; jj++) {
			for (unsigned int i = 0; i<dim; i++) {
				DNhat(dim*ii + i,dim*jj + i) = dphi[dim*ii][qp](0,jj);
			}
		}
	}
}

void NHatMatrix(DenseMatrix<Real> & Nhat, const unsigned int & dim, int & phiSize, const std::vector<std::vector<RealGradient> >& phi, unsigned int & qp) {
	Nhat.resize(phiSize,dim);
	for (unsigned int ii = 0; ii<phiSize/dim; ii++) {
		for (unsigned int i = 0; i<dim; i++) {
			Nhat(dim*ii + i,i) = phi[dim*ii][qp](0);
		}
	}
}

void EvalElasticity(DenseMatrix<Real> & CMat, const std::vector<Point>& q_point, const unsigned int & dim, unsigned int & qp) {

	// Mesh is [0 , 1] x [0 , 0.2]
	// Different materials for [0, 0.5] and [0.5 , 1]
	// This can be optimized, and it should be!
	Real lambda, mu;
    const Real x = q_point[qp](0);
    if ( x <= 0.5){
    	lambda = 5;
    	mu = 2;
    	// FIXME Now it's the fourth order identity tensor because of the manufactured solution
    	if (dim == 2) {
    	  CMat(0,0) = lambda + 2*mu;
    	  CMat(3,3) = lambda + 2*mu;
    	  CMat(1,1) = mu;
    	  CMat(1,2) = mu;
    	  CMat(2,1) = mu;
    	  CMat(2,2) = mu;
    	  CMat(3,0) = lambda;
    	  CMat(0,3) = lambda;
    	}
    	else {
    	  CMat(0,0) = 1;
    	  CMat(1,1) = 1;
    	  CMat(2,2) = 1;
    	  CMat(3,3) = 1;
    	}
    }
    else {
    	lambda = 10;
    	mu = 4;
    	if (dim == 2) {
    	  CMat(0,0) = lambda + 2*mu;
    	  CMat(3,3) = lambda + 2*mu;
    	  CMat(1,1) = mu;
    	  CMat(1,2) = mu;
    	  CMat(2,1) = mu;
    	  CMat(2,2) = mu;
    	  CMat(3,0) = lambda;
    	  CMat(0,3) = lambda;
    	}
    	else {
    	  CMat(0,0) = 1;
    	  CMat(1,1) = 1;
    	  CMat(2,2) = 1;
    	  CMat(3,3) = 1;
    	}
    }
}

// We now define the matrix assembly function for the
// Poisson system.  We need to first compute element
// matrices and right-hand sides, and then take into
// account the boundary conditions, which will be handled
// via a penalty method.

void assemble_poisson(EquationSystems& es,
                      const std::string& system_name)
{

  // It is a good idea to make sure we are assembling
  // the proper system.
  libmesh_assert_equal_to (system_name, "Poisson");


  // Get a constant reference to the mesh object.
  const MeshBase& mesh = es.get_mesh();

  // The dimension that we are running
  const unsigned int dim = mesh.mesh_dimension();

  // Get a reference to the LinearImplicitSystem we are solving
  LinearImplicitSystem& system = es.get_system<LinearImplicitSystem> ("Poisson");

  // A reference to the  DofMap object for this system.  The  DofMap
  // object handles the index translation from node and element numbers
  // to degree of freedom numbers.  We will talk more about the  DofMap
  // in future examples.
  const DofMap& dof_map = system.get_dof_map();

  // Get a constant reference to the Finite Element type
  // for the first (and only) variable in the system.
  FEType fe_type = dof_map.variable_type(0);

  // Build a Finite Element object of the specified type.
  // Note that FEVectorBase is a typedef for the templated FE
  // class.
  AutoPtr<FEVectorBase> fe (FEVectorBase::build(dim, fe_type));

  // A 5th order Gauss quadrature rule for numerical integration.
  QGauss qrule (dim, FIFTH);

  // Tell the finite element object to use our quadrature rule.
  fe->attach_quadrature_rule (&qrule);

//  // Declare a special finite element object for
//  // boundary integration.
//  AutoPtr<FEVectorBase> fe_face (FEVectorBase::build(dim, fe_type));
//
//  // Boundary integration requires one quadraure rule,
//  // with dimensionality one less than the dimensionality
//  // of the element.
//  QGauss qface(dim-1, FIFTH);
//
//  // Tell the finite element object to use our
//  // quadrature rule.
//  fe_face->attach_quadrature_rule (&qface);

  // Here we define some references to cell-specific data that
  // will be used to assemble the linear system.
  //
  // The element Jacobian * quadrature weight at each integration point.
  const std::vector<Real>& JxW = fe->get_JxW();

  // The physical XY locations of the quadrature points on the element.
  // These might be useful for evaluating spatially varying material
  // properties at the quadrature points.
  const std::vector<Point>& q_point = fe->get_xyz();

  // The element shape functions evaluated at the quadrature points.
  // Notice the shape functions are a vector rather than a scalar.
  const std::vector<std::vector<RealGradient> >& phi = fe->get_phi();

  // The element shape function gradients evaluated at the quadrature
  // points. Notice that the shape function gradients are a tensor.
  const std::vector<std::vector<RealTensor> >& dphi = fe->get_dphi();

  // Define data structures to contain the element matrix
  // and right-hand-side vector contribution.  Following
  // basic finite element terminology we will denote these
  // "Ke" and "Fe".  These datatypes are templated on
  //  Number, which allows the same code to work for real
  // or complex numbers.
  DenseMatrix<Number> Ke;
  DenseVector<Number> Fe;


  // This vector will hold the degree of freedom indices for
  // the element.  These define where in the global system
  // the element degrees of freedom get mapped.
  std::vector<dof_id_type> dof_indices;

  // Now we will loop over all the elements in the mesh.
  // We will compute the element matrix and right-hand-side
  // contribution.
  //
  // Element iterators are a nice way to iterate through all the
  // elements, or all the elements that have some property.  The
  // iterator el will iterate from the first to the last element on
  // the local processor.  The iterator end_el tells us when to stop.
  // It is smart to make this one const so that we don't accidentally
  // mess it up!  In case users later modify this program to include
  // refinement, we will be safe and will only consider the active
  // elements; hence we use a variant of the \p active_elem_iterator.
  MeshBase::const_element_iterator       el     = mesh.active_local_elements_begin();
  const MeshBase::const_element_iterator end_el = mesh.active_local_elements_end();

  // Loop over the elements.  Note that  ++el is preferred to
  // el++ since the latter requires an unnecessary temporary
  // object.

  // Create DNhat Matrices
  DenseMatrix<Real> DNhat, Nhat;
  // Create CMat Matrices
  DenseMatrix<Real> CMat;
  CMat.resize(dim*dim,dim*dim);
  DenseVector<Real> f;
  f.resize(dim);
  // Dummy parameters structure and string
  const Parameters _parameters;
  const std::string dummy;

  for ( ; el != end_el ; ++el)
    {
      // Store a pointer to the element we are currently
      // working on.  This allows for nicer syntax later.
      const Elem* elem = *el;

      // Get the degree of freedom indices for the
      // current element.  These define where in the global
      // matrix and right-hand-side this element will
      // contribute to.
      dof_map.dof_indices (elem, dof_indices);

      // Compute the element-specific data for the current
      // element.  This involves computing the location of the
      // quadrature points (q_point) and the shape functions
      // (phi, dphi) for the current element.
      fe->reinit (elem);


      // Zero the element matrix and right-hand side before
      // summing them.  We use the resize member here because
      // the number of degrees of freedom might have changed from
      // the last element.  Note that this will be the case if the
      // element type is different (i.e. the last element was a
      // triangle, now we are on a quadrilateral).

      // The  DenseMatrix::resize() and the  DenseVector::resize()
      // members will automatically zero out the matrix  and vector.
      Ke.resize (dof_indices.size(),
                 dof_indices.size());

      Fe.resize (dof_indices.size());
      // Now loop over the quadrature points.  This handles
      // the numeric integration.
      for (unsigned int qp=0; qp<qrule.n_points(); qp++)
        {
    	  int phiSize = phi.size();
    	  clock_t t;
    	  t = clock();
    	  DNHatMatrix( DNhat, dim, phiSize,  dphi, qp);
    	  EvalElasticity(CMat,q_point,dim,qp);
    	  DNhat.right_multiply(CMat);
    	  DNhat.right_multiply_transpose(DNhat);
    	  DNhat *= JxW[qp];
    	  Ke += DNhat;
    	  t = clock() - t;
          // This is the end of the matrix summation loop
          // Now we build the element right-hand-side contribution.
          // This involves a single loop in which we integrate the
          // "forcing function" in the PDE against the test functions.
          {
            const Real x = q_point[qp](0);
            const Real y = q_point[qp](1);
            const Real eps = 1.e-3;


            // "f" is the forcing function for the Poisson equation.
            // In this case we set f to be a finite difference
            // Laplacian approximation to the (known) exact solution.
            //
            // We will use the second-order accurate FD Laplacian
            // approximation, which in 2D is
            //
            // u_xx + u_yy = (u(i,j-1) + u(i,j+1) +
            //                u(i-1,j) + u(i+1,j) +
            //                -4*u(i,j))/h^2
            //
            // Since the value of the forcing function depends only
            // on the location of the quadrature point (q_point[qp])
            // we will compute it here, outside of the i-loop
//            const Number fx = -(exact_solution(Point(x,y-eps,0),_parameters,dummy,dummy) +
//                              exact_solution(Point(x,y+eps,0),_parameters,dummy,dummy) +
//                              exact_solution(Point(x-eps,y,0),_parameters,dummy,dummy) +
//                              exact_solution(Point(x+eps,y,0),_parameters,dummy,dummy) -
//                              4.*exact_solution(Point(x,y,0),_parameters,dummy,dummy))/eps/eps;
//
//            const Number fy = -(exact_solution(Point(x,y-eps,0),_parameters,dummy,dummy) +
//                    exact_solution(Point(x,y+eps,0),_parameters,dummy,dummy) +
//                    exact_solution(Point(x-eps,y,0),_parameters,dummy,dummy) +
//                    exact_solution(Point(x+eps,y,0),_parameters,dummy,dummy) -
//                    4.*exact_solution(Point(x,y,0),_parameters,dummy,dummy))/eps/eps;
//            f(0) = fx; f(1) = fy;

            //FIXME Temporal body force
            f(0) = 0;
            f(1) = 10;

            NHatMatrix( Nhat, dim, phiSize,  phi, qp);

            Nhat.vector_mult(Fe,f);
            Fe *= JxW[qp];


          }
        }
      }

      // We have now finished the quadrature point loop,
      // and have therefore applied all the boundary conditions.

      // If this assembly program were to be used on an adaptive mesh,
      // we would have to apply any hanging node constraint equations
      dof_map.constrain_element_matrix_and_vector (Ke, Fe, dof_indices);

      // The element matrix and right-hand-side are now built
      // for this element.  Add them to the global matrix and
      // right-hand-side vector.  The  SparseMatrix::add_matrix()
      // and  NumericVector::add_vector() members do this for us.
      system.matrix->add_matrix (Ke, dof_indices);
      system.rhs->add_vector    (Fe, dof_indices);


  // All done!
}
