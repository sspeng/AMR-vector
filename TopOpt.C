/* This is where we define the assembly of the Laplace system */

// General libMesh includes
#include "libmesh/getpot.h"
#include "libmesh/fe_base.h"
#include "libmesh/quadrature.h"
#include "libmesh/string_to_enum.h"
#include "libmesh/parallel.h"
#include "libmesh/fem_context.h"

#include "libmesh/boundary_info.h"

#include "libmesh/elasticity_tools.h"

#include "libmesh/zero_function.h"
#include "libmesh/dirichlet_boundaries.h"
#include "libmesh/dof_map.h"

#include "libmesh/parallel.h"
#include "libmesh/parallel_algebra.h"
#include "libmesh/parallel_ghost_sync.h"

#include <iostream>
#include <utility>
#include <queue>

#include "libmesh/sparse_matrix.h"

#include "libmesh/explicit_system.h"
#include "libmesh/equation_systems.h"



// Local includes
#include "TopOpt.h"
#include "resder.h"
#include <math.h>
#include "vonmises.h"

// Bring in everything from the libMesh namespace
using namespace libMesh;

namespace {
using namespace libMesh;

// give this guy some scope since there
// is underlying vector allocation upon
// creation/deletion
ConstElemRange elem_range;

typedef Threads::spin_mutex femsystem_mutex;
femsystem_mutex assembly_mutex;

void assemble_unconstrained_element_system
(const FEMSystem& _sys,
 const bool _get_jacobian,
 FEMContext &_femcontext)
{
  bool jacobian_computed =
    _sys.time_solver->element_residual(_get_jacobian, _femcontext);

  // Compute a numeric jacobian if we have to
  if (_get_jacobian && !jacobian_computed)
    {
      // Make sure we didn't compute a jacobian and lie about it
      libmesh_assert_equal_to (_femcontext.get_elem_jacobian().l1_norm(), 0.0);
      // Logging of numerical jacobians is done separately
      _sys.numerical_elem_jacobian(_femcontext);
    }

  // Compute a numeric jacobian if we're asked to verify the
  // analytic jacobian we got
  if (_get_jacobian && jacobian_computed &&
      _sys.verify_analytic_jacobians != 0.0)
    {
      DenseMatrix<Number> analytic_jacobian(_femcontext.get_elem_jacobian());

      _femcontext.get_elem_jacobian().zero();
      // Logging of numerical jacobians is done separately
      _sys.numerical_elem_jacobian(_femcontext);

      Real analytic_norm = analytic_jacobian.l1_norm();
      Real numerical_norm = _femcontext.get_elem_jacobian().l1_norm();

      // If we can continue, we'll probably prefer the analytic jacobian
      analytic_jacobian.swap(_femcontext.get_elem_jacobian());

      // The matrix "analytic_jacobian" will now hold the error matrix
      analytic_jacobian.add(-1.0, _femcontext.get_elem_jacobian());
      Real error_norm = analytic_jacobian.l1_norm();

      Real relative_error = error_norm /
        std::max(analytic_norm, numerical_norm);

      if (relative_error > _sys.verify_analytic_jacobians)
        {
          libMesh::err << "Relative error " << relative_error
                       << " detected in analytic jacobian on element "
                       << _femcontext.get_elem().id() << '!' << std::endl;

          std::streamsize old_precision = libMesh::out.precision();
          libMesh::out.precision(16);
          libMesh::out << "J_analytic " << _femcontext.get_elem().id() << " = "
                       << _femcontext.get_elem_jacobian() << std::endl;
          analytic_jacobian.add(1.0, _femcontext.get_elem_jacobian());
          libMesh::out << "J_numeric " << _femcontext.get_elem().id() << " = "
                       << analytic_jacobian << std::endl;

          libMesh::out.precision(old_precision);

          libmesh_error_msg("Relative error too large, exiting!");
        }
    }

  for (_femcontext.side = 0;
       _femcontext.side != _femcontext.get_elem().n_sides();
       ++_femcontext.side)
    {
      // Don't compute on non-boundary sides unless requested
      if (!_sys.get_physics()->compute_internal_sides &&
          _femcontext.get_elem().neighbor(_femcontext.side) != NULL)
        continue;

      // Any mesh movement has already been done (and restored,
      // if the TimeSolver isn't broken), but
      // reinitializing the side FE objects is still necessary
      _femcontext.side_fe_reinit();

      DenseMatrix<Number> old_jacobian;
      // If we're in DEBUG mode, we should always verify that the
      // user's side_residual function doesn't alter our existing
      // jacobian and then lie about it
#ifndef DEBUG
      // Even if we're not in DEBUG mode, when we're verifying
      // analytic jacobians we'll want to verify each side's
      // jacobian contribution separately.
      /* PB: We also need to account for the case when the user wants to
         use numerical Jacobians and not analytic Jacobians */
      if ( (_sys.verify_analytic_jacobians != 0.0 && _get_jacobian) ||
           (!jacobian_computed && _get_jacobian) )
#endif // ifndef DEBUG
        {
          old_jacobian = _femcontext.get_elem_jacobian();
          _femcontext.get_elem_jacobian().zero();
        }
      jacobian_computed =
        _sys.time_solver->side_residual(_get_jacobian, _femcontext);

      // Compute a numeric jacobian if we have to
      if (_get_jacobian && !jacobian_computed)
        {
          // In DEBUG mode, we've already set elem_jacobian == 0,
          // so we can make sure side_residual didn't compute a
          // jacobian and lie about it
          libmesh_assert_equal_to (_femcontext.get_elem_jacobian().l1_norm(), 0.0);

          // Logging of numerical jacobians is done separately
          _sys.numerical_side_jacobian(_femcontext);

          // Add back in element interior numerical Jacobian
          _femcontext.get_elem_jacobian() += old_jacobian;
        }

      // Compute a numeric jacobian if we're asked to verify the
      // analytic jacobian we got
      if (_get_jacobian && jacobian_computed &&
          _sys.verify_analytic_jacobians != 0.0)
        {
          DenseMatrix<Number> analytic_jacobian(_femcontext.get_elem_jacobian());

          _femcontext.get_elem_jacobian().zero();
          // Logging of numerical jacobians is done separately
          _sys.numerical_side_jacobian(_femcontext);

          Real analytic_norm = analytic_jacobian.l1_norm();
          Real numerical_norm = _femcontext.get_elem_jacobian().l1_norm();

          // If we can continue, we'll probably prefer the analytic jacobian
          analytic_jacobian.swap(_femcontext.get_elem_jacobian());

          // The matrix "analytic_jacobian" will now hold the error matrix
          analytic_jacobian.add(-1.0, _femcontext.get_elem_jacobian());
          Real error_norm = analytic_jacobian.l1_norm();

          Real relative_error = error_norm /
            std::max(analytic_norm, numerical_norm);

          if (relative_error > _sys.verify_analytic_jacobians)
            {
              libMesh::err << "Relative error " << relative_error
                           << " detected in analytic jacobian on element "
                           << _femcontext.get_elem().id()
                           << ", side "
                           << static_cast<unsigned int>(_femcontext.side) << '!' << std::endl;

              std::streamsize old_precision = libMesh::out.precision();
              libMesh::out.precision(16);
              libMesh::out << "J_analytic " << _femcontext.get_elem().id() << " = "
                           << _femcontext.get_elem_jacobian() << std::endl;
              analytic_jacobian.add(1.0, _femcontext.get_elem_jacobian());
              libMesh::out << "J_numeric " << _femcontext.get_elem().id() << " = "
                           << analytic_jacobian << std::endl;
              libMesh::out.precision(old_precision);

              libmesh_error_msg("Relative error too large, exiting!");
            }
          // Once we've verified a side, we'll want to add back the
          // rest of the accumulated jacobian
          _femcontext.get_elem_jacobian() += old_jacobian;
        }
      // In DEBUG mode, we've set elem_jacobian == 0, and we
      // may still need to add the old jacobian back
#ifdef DEBUG
      if (_get_jacobian && jacobian_computed &&
          _sys.verify_analytic_jacobians == 0.0)
        {
          _femcontext.get_elem_jacobian() += old_jacobian;
        }
#endif // ifdef DEBUG
    }
}

class ResDerivativeContributions
{
public:
  /**
   * constructor to set context
   */
	ResDerivativeContributions(TopOptSystem &sys, const QoISet& qoi_indices,
								ResidualDerivative & res_der, const dof_id_type & elem_id ) :
    _sys(sys), _qoi_indices(qoi_indices), _res_der(res_der), _elem_id(elem_id) {}

  /**
   * operator() for use with Threads::parallel_for().
   */
  void operator()(const ConstElemRange &range) const
  {
    AutoPtr<DiffContext> con = _sys.build_context();
    FEMContext &_femcontext = libmesh_cast_ref<FEMContext&>(*con);
    _res_der.init_context(_femcontext,_sys);



    for (ConstElemRange::const_iterator elem_it = range.begin();
         elem_it != range.end(); ++elem_it)
      {
        Elem *el = const_cast<Elem *>(*elem_it);

        _femcontext.pre_fe_reinit(_sys, el);
        _femcontext.elem_fe_reinit();

        // Get system
         ExplicitSystem& aux_system = _sys.get_equation_systems().get_system<ExplicitSystem>("Densities");
         // Array to hold the dof indices
         std::vector<dof_id_type> densities_index;
         // Get dof indices
         aux_system.get_dof_map().dof_indices(&_femcontext.get_elem(), densities_index, 0);

         // We are only interested in the element that has the density variable we are evaluating,
         // otherwise, it will have a zero contribution to the derivative of the residual
         if (_elem_id == densities_index[0]){
			// Because we need the stiffness matrix, we build it again
			  _sys.time_solver->element_residual(true, _femcontext);

			// Only build the element contribution from the element whose
			// id is equal to the parameter number, the other elements will
			// have a contribution of zero. This should be optimized obviously.

			_res_der.element_res_derivative(_femcontext, _sys);
         }
         else
        	 _femcontext.get_localized_vector(_sys.get_resder_rhs(0)).zero();


        // We need some unmodified indices to use for constraining
        // multiple vector
        // FIXME - there should be a DofMap::constrain_element_vectors
        // to do this more efficiently
#ifdef LIBMESH_ENABLE_CONSTRAINTS
        std::vector<dof_id_type> original_dofs = _femcontext.get_dof_indices();
#endif

        { // A lock is necessary around access to the global system
          femsystem_mutex::scoped_lock lock(assembly_mutex);

#ifdef LIBMESH_ENABLE_CONSTRAINTS
          // We'll need to see if any heterogenous constraints apply
          // to the QoI dofs on this element *or* to any of the dofs
          // they depend on, so let's get those dependencies
          _sys.get_dof_map().constrain_nothing(_femcontext.get_dof_indices());
#endif

#ifdef LIBMESH_ENABLE_CONSTRAINTS
                bool has_heterogenous_constraint = false;
                for (unsigned int d=0;
                     d != _femcontext.get_dof_indices().size(); ++d)
                  if (_sys.get_dof_map().has_heterogenous_adjoint_constraint
                      (0, _femcontext.get_dof_indices()[d]))
                    {
                      has_heterogenous_constraint = true;
                      break;
                    }

                _femcontext.get_dof_indices() = original_dofs;

                // If we're going to need K to impose a heterogenous
                // constraint then we either already have it or we
                // need to compute it
                if (has_heterogenous_constraint)
                  {
                    assemble_unconstrained_element_system
                      (_sys, true, _femcontext);

                    _sys.get_dof_map().heterogenously_constrain_element_vector
                      (_femcontext.get_elem_jacobian(),
                       _femcontext.get_localized_vector(_sys.get_resder_rhs(0)),
                       _femcontext.get_dof_indices(), false, 0);
                  }
                else
                  {
                    _sys.get_dof_map().constrain_element_vector
                      (_femcontext.get_localized_vector(_sys.get_resder_rhs(0)),
                       _femcontext.get_dof_indices(), false);
                  }
#endif

                _sys.get_resder_rhs(0).add_vector
                  (_femcontext.get_localized_vector(_sys.get_resder_rhs(0)), _femcontext.get_dof_indices());
        }
      }
  }

private:

  TopOptSystem& _sys;
  const QoISet& _qoi_indices;
  ResidualDerivative& _res_der;
  dof_id_type _elem_id;
};

}


void TopOptSystem::init_data ()
{
	u_var = this->add_variable ("u", static_cast<Order>(_fe_order),
					  Utility::string_to_enum<FEFamily>(_fe_family));

	// Set initial values of the parameters to 0.5


	this->init_bcs();

    densities_filtered.init(this->get_mesh().n_active_elem(), this->get_mesh().n_active_elem());

    gradient_filtered.init(this->get_mesh().n_active_elem(), this->get_mesh().n_active_elem());

    gradient_filtered_temp.init(this->get_mesh().n_active_elem(), this->get_mesh().n_active_elem());

	// Add the kernel matrix

	//kernel_filter = &(SparseMatrix<Number>::build(this->comm()).release());

	// Do the parent's initialization after variables are defined
	FEMSystem::init_data();

	this->time_evolving(0);
}

void TopOptSystem::init_bcs()
{


	// Get the set of boundary ids
	boundary_id_type hanger = 0;
	std::set<boundary_id_type> boundary_ids;
	boundary_ids.insert(hanger);

	// Create a vector storing the variable numbers which the BC applies to
	std::vector<unsigned int> vars;
	vars.push_back( u_var );

	// Note that for vector-valued variables, it is assumed each component is stored contiguously.
	// For 2-D elements in 3-D space, only two components should be returned.
	//SolutionFunction func( u_var );
	ZeroFunction<Number> zero;

	// We must add the Dirichlet boundary condition _before_
	// we call equation_systems.init()
	this->get_dof_map().add_dirichlet_boundary(libMesh::DirichletBoundary( boundary_ids, vars, &zero ));

	return;
}

void TopOptSystem::init_context(DiffContext &context)
{
  FEMContext &c = libmesh_cast_ref<FEMContext&>(context);

  // Now make sure we have requested all the data
  // we need to build the linear system.
  FEVectorBase* elem_fe = NULL;
  c.get_element_fe( 0, elem_fe );
  elem_fe->get_JxW();
  elem_fe->get_phi();
  elem_fe->get_dphi();

  FEVectorBase* side_fe = NULL;
  c.get_side_fe( 0, side_fe );

  side_fe->get_JxW();
  side_fe->get_phi();
  side_fe->get_dphi();

  // Initialize the matrices to build the stiffness matrix
  unsigned int dim = c.get_dim();
  DNhat.resize((elem_fe->get_phi().size()),dim*dim);
  Nhat.resize((elem_fe->get_phi().size()),dim);
  CMat.resize(dim*dim,dim*dim);
  stress_flux_vect.resize(dim);


  // Add Residual Derivative Object
  AutoPtr<ResidualDerivative> dummy(new ResidualDerivative);
  _res_der = dummy;


  FEMSystem::init_context(context);
}

#define optassert(X) {if (!(X)) libmesh_error_msg("Assertion " #X " failed.");}

// Assemble the element contributions to the stiffness matrix
bool TopOptSystem::element_time_derivative (bool request_jacobian,
                                             DiffContext &context)
{
  // Are the jacobians specified analytically ?
  bool compute_jacobian = request_jacobian && _analytic_jacobians;

  FEMContext &c = libmesh_cast_ref<FEMContext&>(context);

  // The dimension that we are running
  const unsigned int dim = c.get_dim();

  // First we get some references to cell-specific data that
  // will be used to assemble the linear system.
  FEVectorBase* elem_fe = NULL;
  c.get_element_fe( 0, elem_fe );

  // Resize here, for some reason we are not capable of doing so inside the initialization.
  DNhat.resize((elem_fe->get_phi().size()),dim*dim);
  Nhat.resize((elem_fe->get_phi().size()),dim);

  // Element Jacobian * quadrature weights for interior integration
  const std::vector<Real> &JxW = elem_fe->get_JxW();

  // The element shape functions evaluated at the quadrature points.
  // Notice the shape functions are a vector rather than a scalar.
  const std::vector<std::vector<RealGradient> >& phi = elem_fe->get_phi();

  // Derivatives of the rlement basis functions
  const std::vector<std::vector<RealTensor> > &dphi = elem_fe->get_dphi();



  // The subvectors and submatrices we need to fill:
  DenseMatrix<Number> &K = c.get_elem_jacobian();
  DenseVector<Number> &F = c.get_elem_residual();

  // Zero the matrices used to build the stiffness matrix
  K.zero();
  F.zero();
  Nhat.zero();
  DNhat.zero();
  CMat.zero();
  stress_flux_vect.zero();


  // Now we will build the element Jacobian and residual.
  // Constructing the residual requires the solution and its
  // gradient from the previous timestep.  This must be
  // calculated at each quadrature point by summing the
  // solution degree-of-freedom values by the appropriate
  // weight functions.
  unsigned int n_qpoints = c.get_element_qrule().n_points();

  DenseMatrix<Real> Temp;
  DenseVector<Real> gradU, stress, stress_temp, f_temp;
  gradU.resize(dim*dim);
  f_temp.resize(dim);

  // Get the density value for this element

  // Get system
  ExplicitSystem& aux_system = this->get_equation_systems().get_system<ExplicitSystem>("Densities");
  // Array to hold the dof indices
  std::vector<dof_id_type> densities_index;
  // Array to hold the values
  std::vector<Number> density;
  // Get dof indices
  aux_system.get_dof_map().dof_indices(&c.get_elem(), densities_index, 0);
  // Get the value, stored in u_undefo
  aux_system.current_local_solution->get(densities_index, density);
  // Apply ramp
  ramp(density[0]);

  for (unsigned int qp=0; qp<n_qpoints; qp++){
	  // Write shape function matrices
 	  ElasticityTools::DNHatMatrix( DNhat, dim, dphi, qp);
 	  ElasticityTools::NHatMatrix( Nhat, dim,  phi, qp);
 	  // Evaluate elasticity tensor
 	  EvalElasticity(CMat);
 	  // Evaluate element contribution
 	  Temp = DNhat;
 	  Temp.right_multiply(CMat);
 	  Temp.right_multiply_transpose(DNhat);
 	  Temp *= density[0];
 	  K.add(JxW[qp],Temp);

 	  // Get gradient, we need it for the residual
      Tensor grad_u;

     c.interior_gradient( u_var, qp, grad_u );

     gradU(0) = grad_u(0,0);
     gradU(1) = grad_u(1,0);
     gradU(2) = grad_u(0,1);
     gradU(3) = grad_u(1,1);

     // Calculate stress
     CMat.vector_mult(stress_temp,gradU);

     // Factor stress with the density
     stress_temp *= density[0];

     // Multiplied with the shape derivative matrix because of the
     // contribution of the test variable inthe residual form
     DNhat.vector_mult(stress,stress_temp);

     F.add(JxW[qp],stress);

     // Body force contribution
     if (integrate_body_force){
		 f = _body_force (Point(0,0,0), "u");
		 f_temp(0) = f(0); f_temp(1) = f(1);
		 // Negative because that's how the formulation is
		 f_temp *= -1;
		 Nhat.vector_mult_add(F,JxW[qp],f_temp);
     }

  }
  // Apply density
  //K *= density[0];

  return compute_jacobian;
}

// Set Dirichlet bcs, side contributions to global stiffness matrix
bool TopOptSystem::side_constraint (bool request_jacobian,
                                     DiffContext &context)
{
  // Are the jacobians specified analytically ?
  bool compute_jacobian = request_jacobian && _analytic_jacobians;

  FEMContext &c = libmesh_cast_ref<FEMContext&>(context);

  // The dimension that we are running
  const unsigned int dim = c.get_dim();

  // First we get some references to cell-specific data that
  // will be used to assemble the linear system.
  FEVectorBase * side_fe = NULL;
  c.get_side_fe( 0, side_fe );

  // Element Jacobian * quadrature weights for interior integration
  const std::vector<Real> &JxW = side_fe->get_JxW();

  // Side basis functions
  const std::vector<std::vector<RealGradient> > &phi = side_fe->get_phi();


  // The number of local degrees of freedom in each variable
  const unsigned int n_T_dofs = c.get_dof_indices(0).size();

  // The subvectors and submatrices we need to fill:
  DenseSubVector<Number> &F = c.get_elem_residual(0);

  unsigned int n_qpoints = c.get_side_qrule().n_points();

  // Resize here
  Nhat.resize(phi.size()*dim,dim);

	  // Grab boundary ids in side element
	  std::vector<boundary_id_type> side_BdId = c.side_boundary_ids();
	  //short int bc_id = mesh.boundary_info->boundary_id	(elem,side);
	  if (!side_BdId.empty() && side_BdId[0] == 1 && integrate_boundary_sides)
      {

		  for (unsigned int qp=0; qp<n_qpoints; qp++)
		  {
				std::pair<bool,Gradient> stress_flux = _bc_function (*this, Point(0,0,0), "u");

				ElasticityTools::NHatMatrix( Nhat, dim,  phi, qp);

				stress_flux_vect(0) = stress_flux.second(0);
				stress_flux_vect(1) = stress_flux.second(1);
				for (unsigned int i=0; i<n_T_dofs; i++)
					for (unsigned int j=0; j<dim; j++)
						F(i) -= JxW[qp]*Nhat(i,j)*stress_flux_vect(j);
		  }
      }

  return compute_jacobian;
}

void TopOptSystem::EvalElasticity(DenseMatrix<Real> & CMat) {
	CMat(0,0) = _lambda + 2*_mu;
	CMat(3,3) = _lambda + 2*_mu;
	CMat(1,1) = _mu;
	CMat(1,2) = _mu;
	CMat(2,1) = _mu;
	CMat(2,2) = _mu;
	CMat(3,0) = _lambda;
	CMat(0,3) = _lambda;
}

void
TopOptSystem::attach_flux_bc_function (std::pair<bool,Gradient> fptr(const System& ,
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

void
TopOptSystem::attach_body_force (Gradient fptr(const Point& ,const std::string& ))
{
  _body_force = fptr;


  // We may be turning boundary side integration on or off
  if (fptr)
	  integrate_body_force = true;
  else
	  integrate_body_force = false;

}

void TopOptSystem::adjoint_qoi_parameter_sensitivity
(const QoISet&          qoi_indices,
 const ParameterVector& parameters,
 SensitivityData&       sensitivities)
{
	const unsigned int Nq = libmesh_cast_int<unsigned int>
	(qoi.size());
	const unsigned int Np = this->get_mesh().n_active_elem();

	// Get system
	ExplicitSystem& aux_system = this->get_equation_systems().get_system<ExplicitSystem>("Densities");
	// Array to hold the dof indices
	std::vector<dof_id_type> densities_index;
	// Array to hold the values
	std::vector<Number> density;
	// Dof Map for the densities system
	const DofMap& dof_map_densities = aux_system.get_dof_map();
	// Variable number
	unsigned int density_var = aux_system.variable_number ("rho");

	// An introduction to the problem:
	//
	// Residual R(u(p),p) = 0
	// partial R / partial u = J = system matrix
	//
	// This implies that:
	// d/dp(R) = 0
	// (partial R / partial p) +
	// (partial R / partial u) * (partial u / partial p) = 0

	// We first do an adjoint solve:
	// J^T * z = (partial q / partial u)
	// if we havent already or dont have an initial condition for the adjoint

	if (!this->is_adjoint_already_solved())
	{
	  this->adjoint_solve(qoi_indices);
	}

	// Get ready to fill in senstivities:
	sensitivities.allocate_data(qoi_indices, *this, parameters);

	// We use the identities:
	// dq/dp = (partial q / partial p) + (partial q / partial u) *
	//         (partial u / partial p)
	// dq/dp = (partial q / partial p) + (J^T * z) *
	//         (partial u / partial p)
	// dq/dp = (partial q / partial p) + z * J *
	//         (partial u / partial p)

	// Leading to our final formula:
	// dq/dp = (partial q / partial p) - z * (partial R / partial p)

	// In the case of adjoints with heterogenous Dirichlet boundary
	// function phi, where
	// q := R(u,phi) + S(u)
	// the final formula works out to:
	// dq/dp = (partial S / partial p) - z * (partial R / partial p)
	// Because we currently have no direct access to
	// (partial S / partial p), we use the identity
	// (partial S / partial p) = (partial q / partial p) -
	//                           phi * (partial R / partial p)
	// to derive an equivalent equation:
	// dq/dp = (partial q / partial p) - (z+phi) * (partial R / partial p)

	MeshBase::const_element_iterator       el     = this->get_mesh().active_local_elements_begin();
	const MeshBase::const_element_iterator end_el = this->get_mesh().active_local_elements_end();

    AutoPtr<DiffContext> con = this->build_context();
    FEMContext &_femcontext = libmesh_cast_ref<FEMContext&>(*con);

	std::vector<Number> partialq_partialp(Np);
	std::cout<<"Adjoint Analysis"<<std::endl;
	for (; el != end_el; el++)
	{
		const Elem * elem = *el;
		_femcontext.pre_fe_reinit(*this, elem);
		// (partial q / partial p) ~= (q(p+dp)-q(p-dp))/(2*dp)
		// (partial R / partial p) ~= (rhs(p+dp) - rhs(p-dp))/(2*dp)
		// Get dof indices
		dof_map_densities.dof_indices(elem, densities_index, density_var);

		// Get the value, stored in density
		aux_system.current_local_solution->get(densities_index, density);

//		/*------------- THIS IS NO LONGER NECESSARY, DELETE AT SOME POINT -----------------------*/
//		// We currently get partial derivatives via central differencing
//		const Real delta_p = 1e-6;
//		Number old_parameter = density[0];
//		Number parameter_back = old_parameter - delta_p;
//		aux_system.current_local_solution->set(densities_index[0],parameter_back);
//		aux_system.current_local_solution->close();
//		this->assemble_qoi(qoi_indices);
//		std::vector<Number> qoi_minus = this->qoi;
//
//		this->assembly(true, false);
//		this->rhs->close();
//
//		// FIXME - this can and should be optimized to avoid the clone()
//		AutoPtr<NumericVector<Number> > partialR_partialp = this->rhs->clone();
//		*partialR_partialp *= -1;
//
//		Number parameter_forward = old_parameter + delta_p;
//		aux_system.current_local_solution->set(densities_index[0],parameter_forward);
//		aux_system.current_local_solution->close();
//		this->assemble_qoi(qoi_indices);
//		std::vector<Number>& qoi_plus = this->qoi;
//
//
//		for (unsigned int i=0; i != Nq; ++i)
//		if (qoi_indices.has_index(i))
//		partialq_partialp[i] = (qoi_plus[i] - qoi_minus[i]) / (2.*delta_p);
//
//		this->assembly(true, false);
//		this->rhs->close();
//		*partialR_partialp += *this->rhs;
//		*partialR_partialp /= (2.*delta_p);
//
//		// Don't leave the parameter changed
//		aux_system.current_local_solution->set(densities_index[0],old_parameter);
//		aux_system.current_local_solution->close();
		/*------------- THIS IS NO LONGER NECESSARY, DELETE AT SOME POINT -----------------------*/

		// Here we calculate (partial R / partial p) using our built in function and we'll compare later.
		this->assemble_res_derivative (qoi_indices, densities_index[0]);
		// Calculate dQ/dP for this parameter
		if (p_norm_objectivefunction){
			_femcontext.elem_fe_reinit();
			VonMisesPnorm * vonmises = (VonMisesPnorm *) this->get_qoi();
			Number parameter_derivative;
			vonmises->element_qoi_derivative_parameter(*con,parameter_derivative,density[0]);
			partialq_partialp[densities_index[0]] = parameter_derivative;
		}

//		std::cout<<"Partial derivative with analytical"<<std::endl;
//		this->get_resder_rhs(0).print();
//
//		std::cout<<"Partial derivative with FD"<<std::endl;
//		partialR_partialp->print();

		for (unsigned int i=0; i != Nq; ++i)
			if (qoi_indices.has_index(i))
			{

				if (this->get_dof_map().has_adjoint_dirichlet_boundaries(i))
				{
				AutoPtr<NumericVector<Number> > lift_func =
				  this->get_adjoint_solution(i).zero_clone();
				this->get_dof_map().enforce_constraints_exactly
				  (*this, lift_func.get(),
				   /* homogeneous = */ false);
//				sensitivities[i][densities_index[0]] = partialq_partialp[i] -
//				  partialR_partialp->dot(*lift_func) -
//				  partialR_partialp->dot(this->get_adjoint_solution(i));

				sensitivities[i][densities_index[0]] = partialq_partialp[densities_index[0]] -
					this->get_resder_rhs(0).dot(*lift_func) -
					this->get_resder_rhs(0).dot(this->get_adjoint_solution(i));
				}
				else{
//				sensitivities[i][densities_index[0]] = partialq_partialp[i] -
//				partialR_partialp->dot(this->get_adjoint_solution(i));
//
//				std::cout<<"Sensitivities with FD"<<std::endl;
//				std::cout<<sensitivities[i][densities_index[0]]<<std::endl;


				sensitivities[i][densities_index[0]] = partialq_partialp[densities_index[0]] -
				this->get_resder_rhs(0).dot(this->get_adjoint_solution(i));

//				std::cout<<"Sensitivities with analytical"<<std::endl;
//				std::cout<<sensitivities[i][densities_index[0]]<<std::endl;
				}
			}
	}

	// Sensitivities have been calculated
	sensitivities_calculated = true;

	// All parameters have been reset.
	// We didn't cache the original rhs or matrix for memory reasons,
	// but we can restore them to a state consistent solution -
	// principle of least surprise.


	//this->assembly(true, true);
	this->rhs->close();
	this->matrix->close();
	this->assemble_qoi(qoi_indices);
}

void TopOptSystem::finite_differences_partial_derivatives_check(const QoISet&          qoi_indices,
													   const ParameterVector& parameters,
													   SensitivityData&       sensitivities,
													   bool & p_norm_objectivefunction)
{
	const unsigned int Nq = libmesh_cast_int<unsigned int>
	(qoi.size());


	// Get system
	ExplicitSystem& aux_system = this->get_equation_systems().get_system<ExplicitSystem>("Densities");
	// Array to hold the dof indices
	std::vector<dof_id_type> densities_index;
	// Array to hold the values
	std::vector<Number> density;
	// Dof Map for the densities system
	const DofMap& dof_map_densities = aux_system.get_dof_map();
	// Variable number
	unsigned int density_var = aux_system.variable_number ("rho");

	// An introduction to the problem:
	//
	// Residual R(u(p),p) = 0
	// partial R / partial u = J = system matrix
	//
	// This implies that:
	// d/dp(R) = 0
	// (partial R / partial p) +
	// (partial R / partial u) * (partial u / partial p) = 0

	// We first do an adjoint solve:
	// J^T * z = (partial q / partial u)
	// if we havent already or dont have an initial condition for the adjoint

	if (!this->is_adjoint_already_solved())
	{
	  this->adjoint_solve(qoi_indices);
	}

	// Get ready to fill in senstivities:
	sensitivities.allocate_data(qoi_indices, *this, parameters);

	// We use the identities:
	// dq/dp = (partial q / partial p) + (partial q / partial u) *
	//         (partial u / partial p)
	// dq/dp = (partial q / partial p) + (J^T * z) *
	//         (partial u / partial p)
	// dq/dp = (partial q / partial p) + z * J *
	//         (partial u / partial p)

	// Leading to our final formula:
	// dq/dp = (partial q / partial p) - z * (partial R / partial p)

	// In the case of adjoints with heterogenous Dirichlet boundary
	// function phi, where
	// q := R(u,phi) + S(u)
	// the final formula works out to:
	// dq/dp = (partial S / partial p) - z * (partial R / partial p)
	// Because we currently have no direct access to
	// (partial S / partial p), we use the identity
	// (partial S / partial p) = (partial q / partial p) -
	//                           phi * (partial R / partial p)
	// to derive an equivalent equation:
	// dq/dp = (partial q / partial p) - (z+phi) * (partial R / partial p)

	MeshBase::const_element_iterator       el     = this->get_mesh().active_local_elements_begin();
	const MeshBase::const_element_iterator end_el = this->get_mesh().active_local_elements_end();

	dof_id_type n_variables = this->get_mesh().n_active_elem();
	std::vector<std::vector<Number> > partialq_partialp;
	partialq_partialp.resize(Nq);

	std::vector<std::vector<Number> > partialq_partialp_num;
	partialq_partialp_num.resize(Nq);

    AutoPtr<DiffContext> con = this->build_context();
    FEMContext &_femcontext = libmesh_cast_ref<FEMContext&>(*con);
    this->get_qoi()->init_context(_femcontext);

	for (unsigned int i = 0; i<Nq; i++ ){
		partialq_partialp[i].resize(n_variables);
		partialq_partialp_num[i].resize(n_variables);
	}


	for (; el != end_el; el++)
	{
		const Elem * elem = *el;

		_femcontext.pre_fe_reinit(*this, elem);

		// (partial q / partial p) ~= (q(p+dp)-q(p-dp))/(2*dp)
		// (partial R / partial p) ~= (rhs(p+dp) - rhs(p-dp))/(2*dp)
		// Get dof indices
		dof_map_densities.dof_indices(elem, densities_index, density_var);
		// Get the value, stored in density
		aux_system.current_local_solution->get(densities_index, density);

		/*------------- THIS IS NO LONGER NECESSARY, DELETE AT SOME POINT -----------------------*/
		// We currently get partial derivatives via central differencing
		const Real delta_p = 1e-6;
		Number old_parameter = density[0];
		Number parameter_back = old_parameter - delta_p;
		aux_system.current_local_solution->set(densities_index[0],parameter_back);
		aux_system.current_local_solution->close();
		this->assemble_qoi(qoi_indices);
		std::vector<Number> qoi_minus = this->qoi;
		if (p_norm_objectivefunction){
			for (unsigned int k=0 ; k!=qoi.size(); k++)
				qoi_minus[k]= pow(qoi_minus[k], 1.0/pnorm_parameter);
		}

		this->assembly(true, false);
		this->rhs->close();

		// FIXME - this can and should be optimized to avoid the clone()
		AutoPtr<NumericVector<Number> > partialR_partialp = this->rhs->clone();
		*partialR_partialp *= -1;

		Number parameter_forward = old_parameter + delta_p;
		aux_system.current_local_solution->set(densities_index[0],parameter_forward);
		aux_system.current_local_solution->close();
		this->assemble_qoi(qoi_indices);
		std::vector<Number>& qoi_plus = this->qoi;
		if (p_norm_objectivefunction){
			for (unsigned int k=0 ; k!=qoi.size(); k++)
				qoi_plus[k]= pow(qoi_plus[k], 1.0/pnorm_parameter);
		}


		for (unsigned int i=0; i != Nq; ++i)
		if (qoi_indices.has_index(i))
			partialq_partialp_num[i][densities_index[0]] = (qoi_plus[i] - qoi_minus[i]) / (2.*delta_p);

		this->assembly(true, false);
		this->rhs->close();
		*partialR_partialp += *this->rhs;
		*partialR_partialp /= (2.*delta_p);

		// Don't leave the parameter changed
		aux_system.current_local_solution->set(densities_index[0],old_parameter);
		aux_system.current_local_solution->close();
		/*------------- THIS IS NO LONGER NECESSARY, DELETE AT SOME POINT -----------------------*/

		// Calculate dQ/dP for this parameter
		if (p_norm_objectivefunction){
			_femcontext.elem_fe_reinit();
			VonMisesPnorm * vonmises = (VonMisesPnorm *) this->get_qoi();
			Number parameter_derivative;
			vonmises->element_qoi_derivative_parameter(*con,parameter_derivative,density[0]);
			partialq_partialp[0][densities_index[0]] = parameter_derivative;
		}

		// Here we calculate (partial R / partial p) using our built in function and we'll compare later.

		this->assemble_res_derivative (qoi_indices, densities_index[0]);

		std::cout<<"dR/dP with analytical for element = "<<densities_index[0]<<std::endl;
		this->get_resder_rhs(0).print();

		std::cout<<"dR/dP with FD for element ="<<densities_index[0]<<std::endl;
		partialR_partialp->print();
		std::cout<<"dQ/dP with FD  -----  dQ/dP with analytical for element ="<<densities_index[0]<<std::endl;
		for (unsigned int i=0; i != Nq; ++i) {
			if (qoi_indices.has_index(i))
			std::cout<<partialq_partialp[i][densities_index[0]]<<"   		     "<<partialq_partialp_num[i][densities_index[0]]<<std::endl;
		}
	}

	std::cout << "Terminamos este FD check" << std::endl;

	  // The quantity of interest derivative assembly accumulates on
	  // initially zero vectors
//	  for (unsigned int i=0; i != qoi.size(); ++i)
//	    if (qoi_indices.has_index(i))
//	      this->add_dqdu_fd(i).zero();
//
//	  std::ostringstream dqdu_fd_name;
//	  dqdu_fd_name << "dqdu_fd" << 0;
//	  this->set_vector_preservation(dqdu_fd_name.str(),true);

	  std::cout << "Empezamos este FD check" << std::endl;
	  MeshBase::node_iterator node_begin = this->get_mesh().local_nodes_begin();
	  MeshBase::node_iterator node_end = this->get_mesh().local_nodes_end();

		for (unsigned int j=0; j != qoi.size(); ++j)
			for (; node_begin != node_end; ++node_begin){
				const Node * nodo = *node_begin;
				unsigned int sys_num = this->number();
				unsigned int u_var = this->variable_number ("u");
				unsigned int n_components = nodo->n_comp(sys_num,u_var);

				for (unsigned int k = 0; k<n_components; k++){

					unsigned int i = nodo->dof_number(sys_num, u_var, k);
					std::cout << "dof = "<<i<< std::endl;
					const Real delta_p = 1e-6;

					Number old_parameter = this->solution->operator ()(i);
					Number parameter_back = old_parameter - delta_p;

//					this->solution->set(i,parameter_back);
//					this->solution->close();

					this->assemble_qoi(qoi_indices);

					std::vector<Number> qoi_minus = this->qoi;
					if (p_norm_objectivefunction){
						for (unsigned int k=0 ; k!=qoi.size(); k++)
							qoi_minus[k]= pow(qoi_minus[k], 1.0/pnorm_parameter);
					}


					Number parameter_forward = old_parameter + delta_p;
//					this->solution->set(i,parameter_forward);
//					this->solution->close();

					this->assemble_qoi(qoi_indices);
					std::vector<Number>  qoi_plus = this->qoi;
					if (p_norm_objectivefunction){
						for (unsigned int k=0 ; k!=qoi.size(); k++)
							qoi_plus[k]= pow(qoi_plus[k], 1.0/pnorm_parameter);
					}

					// Evaluate partial derivative
					Number dQdU = (qoi_plus[j] - qoi_minus[j]) / (2.*delta_p);

					// Set it
					//this->get_dqdu_fd(j).set(i,dQdU);

					// Modify solution back to its original
//					this->solution->set(i, old_parameter);
//					this->solution->close();

				}

			}
		std::cout << "Terminamos FD check partial" << std::endl;
	// Close vectors for reuse
//	for (unsigned int i=0; i != qoi.size(); ++i)
//		if (qoi_indices.has_index(i))
//			this->get_dqdu_fd(i).close();


	std::cout<<"dQ/dU with analytical"<<std::endl;
	this->get_adjoint_rhs(0).print();

	std::cout<<"dQ/dU with FD"<<std::endl;
//	for (unsigned int i=0; i != qoi.size(); ++i)
//		if (qoi_indices.has_index(i))
//			this->get_dqdu_fd(i).print();

	std::cout << "Terminamos FD check" << std::endl;

	// All parameters have been reset.
	// We didn't cache the original rhs or matrix for memory reasons,
	// but we can restore them to a state consistent solution -
	// principle of least surprise.
	this->assembly(true, true);
	this->rhs->close();
	this->matrix->close();
}

NumericVector<Number> & TopOptSystem::add_resder_rhs (unsigned int i)
{
  std::ostringstream resder_rhs_name;
  resder_rhs_name << "res_der" << i;

  return this->add_vector(resder_rhs_name.str(), false, GHOSTED);
}

NumericVector<Number> & TopOptSystem::get_resder_rhs (unsigned int i)
{
	  std::ostringstream resder_rhs_name;
	  resder_rhs_name << "res_der" << i;

  return this->get_vector(resder_rhs_name.str());
}

NumericVector<Number> & TopOptSystem::add_dqdu_fd (unsigned int i)
{
  std::ostringstream dqdu_fd_name;
  dqdu_fd_name << "dqdu_fd" << i;

  return this->add_vector(dqdu_fd_name.str(), false, GHOSTED);
}

NumericVector<Number> & TopOptSystem::get_dqdu_fd (unsigned int i)
{
  std::ostringstream dqdu_fd_name;
  dqdu_fd_name << "dqdu_fd" << i;

  return this->get_vector(dqdu_fd_name.str());
}

void TopOptSystem::assemble_res_derivative (const QoISet& qoi_indices, const dof_id_type & elem_id)
{
  START_LOG("assemble_qoi_derivative()", "FEMSystem");

  const MeshBase& mesh = this->get_mesh();

  this->update();

  // The quantity of interest derivative assembly accumulates on
  // initially zero vectors
  this->add_resder_rhs(0).zero();
  std::ostringstream resder_rhs_name;
  resder_rhs_name << "res_der" << 0;
  this->set_vector_preservation(resder_rhs_name.str(),true);

  // Loop over every active mesh element on this processor
  Threads::parallel_for(elem_range.reset(mesh.active_local_elements_begin(),
                                         mesh.active_local_elements_end()),
		  	  	  	  	ResDerivativeContributions(*this, qoi_indices,
                                                   *(this->_res_der.release()),elem_id));

  // Close vectors for reuse
 this->get_resder_rhs(0).close();

  STOP_LOG("assemble_qoi_derivative()", "FEMSystem");
}

void TopOptSystem::calculate_qoi(const QoISet &qoi_indices)
{
  // Reset the array holding the computed QoIs
  computed_QoI[0] = 0.0;

  FEMSystem::assemble_qoi(qoi_indices);
  std::vector<Number> qoi = this->qoi;

  computed_QoI[0] = qoi[0];

}

void TopOptSystem::finite_differences_check(ExplicitSystem * densities,
											const QoISet & qois,
											const SensitivityData&       sensitivities,
											const ParameterVector& parameters){

	libmesh_assert(sensitivities_calculated);

	const unsigned int Nq = libmesh_cast_int<unsigned int>
	(qoi.size());

	// We currently get partial derivatives via central differencing
	const Real delta_p = 1e-6;
	// Get system
	ExplicitSystem& aux_system = *densities;
	// Array to hold the dof indices
	std::vector<dof_id_type> densities_index;
	// Array to hold the values
	std::vector<Number> density;
	// Dof Map for the densities system
	const DofMap& dof_map_densities = aux_system.get_dof_map();
	// Variable number
	unsigned int density_var = aux_system.variable_number ("rho");

	// Allocate data for the Sensitivities calculated FD
	SensitivityData sensitivities_fd(qois, *this, this->get_parameter_vector());

	// Get ready to fill in sensitivities:
	sensitivities_fd.allocate_data(qois, *this, parameters);

	// Iterating through the element
	MeshBase::const_element_iterator       el     = this->get_mesh().active_local_elements_begin();
	const MeshBase::const_element_iterator end_el = this->get_mesh().active_local_elements_end();


	std::cout<<"Comparing global sensitivities with Finite Differences"<<std::endl;
	for (; el != end_el; el++)
	{
		const Elem * elem = *el;
		std::vector<Number> partialq_partialp(Nq, 0);
		// (partial q / partial p) ~= (q(p+dp)-q(p-dp))/(2*dp)
		// Get dof indices
		dof_map_densities.dof_indices(elem, densities_index, density_var);
		// Get the value, stored in density
		aux_system.current_local_solution->get(densities_index, density);

		Number old_parameter = density[0];
		Number parameter_back = old_parameter - delta_p;
		aux_system.current_local_solution->set(densities_index[0],parameter_back);
		aux_system.current_local_solution->close();
		// Solve system to obtain the new QoI
		this->solve();
		// Calculate perturbed QoI
		this->assemble_qoi(qois);
		std::vector<Number> qoi_minus = this->qoi;
		if (p_norm_objectivefunction){
			for (unsigned int k=0 ; k!=qoi.size(); k++)
				qoi_minus[k]= pow(qoi_minus[k], 1.0/pnorm_parameter);
		}

		Number parameter_forward = old_parameter + delta_p;
		aux_system.current_local_solution->set(densities_index[0],parameter_forward);
		aux_system.current_local_solution->close();
		// Solve system to obtain the new QoI
		this->solve();
		// Calculate perturbed QoI
		this->assemble_qoi(qois);
		std::vector<Number>& qoi_plus = this->qoi;
		if (p_norm_objectivefunction){
			for (unsigned int k=0 ; k!=qoi.size(); k++)
				qoi_plus[k]= pow(qoi_plus[k], 1.0/pnorm_parameter);
		}

		// Don't leave the parameter changed
		aux_system.current_local_solution->set(densities_index[0],old_parameter);
		aux_system.current_local_solution->close();


		std::cout<<"Sensitivities with FD  -----  Sensitivities with analytical"<<std::endl;
		for (unsigned int i=0; i != Nq; ++i) {
			if (qois.has_index(i))
				sensitivities_fd[i][densities_index[0]] = (qoi_plus[i] - qoi_minus[i]) / (2.*delta_p);

			std::cout<<sensitivities_fd[i][densities_index[0]]<<"   		     "<<sensitivities[i][densities_index[0]]<<std::endl;
		}

	}


}

void TopOptSystem::transfer_densities(ExplicitSystem & densities, const MeshBase & mesh, const std::vector<double> & x, const bool & filter){
	std::vector<dof_id_type> density_index;
	const DofMap& dof_map_densities = densities.get_dof_map();

	unsigned int density_var = densities.variable_number ("rho");

	MeshBase::const_element_iterator       el     = mesh.active_local_elements_begin();
	const MeshBase::const_element_iterator end_el = mesh.active_local_elements_end();

	for ( ; el != end_el; ++el){
		const Elem* elem = *el;
		dof_map_densities.dof_indices (elem, density_index, density_var);

		// We are using CONSTANT MONOMIAL basis functions, hence we only need to get
		// one dof index per variable
		dof_id_type dof_index = density_index[0];

		if( (densities.solution->first_local_index() <= dof_index) &&
				(dof_index < densities.solution->last_local_index()) )
			densities.solution->set(dof_index, x[dof_index]);
	}

	// Should call close and update when we set vector entries directly
	densities.solution->close();
	densities.update();

	if (filter){
		// Apply filter
		kernel_filter.vector_mult(densities_filtered, *densities.solution.get());

		// Copy them back to densities solution
		el     = mesh.active_local_elements_begin();
		for ( ; el != end_el; ++el){
			const Elem* elem = *el;
			dof_map_densities.dof_indices (elem, density_index, density_var);

			// We are using CONSTANT MONOMIAL basis functions, hence we only need to get
			// one dof index per variable
			dof_id_type dof_index = density_index[0];

			if( (densities.solution->first_local_index() <= dof_index) &&
					(dof_index < densities.solution->last_local_index()) )
				densities.solution->set(dof_index, densities_filtered(dof_index));
		}

		// Should call close and update when we set vector entries directly
		densities.solution->close();
		densities.update();
	}
}

void TopOptSystem::filter_gradient(const std::vector<Number> & gradient, ExplicitSystem & densities, std::vector<double> & grad){

	std::vector<dof_id_type> density_index;
	const DofMap& dof_map_densities = densities.get_dof_map();

	unsigned int density_var = densities.variable_number ("rho");

	// Copy sensitivities
	gradient_filtered = gradient;

	gradient_filtered_temp.zero();

	// Perform product of gradient_filtered_temp = (kernel_filter)^T * gradient_filtered (right now they're just the derivatives)
	gradient_filtered_temp.add_vector_transpose(gradient_filtered, kernel_filter);

	// The filtered gradient is gradient_filtered_temp, copy them back to the sensitivities
	gradient_filtered_temp.localize(grad);

}

Number TopOptSystem::calculate_volume_constraint(ExplicitSystem & densities, std::vector<double> & grad){


	// Sensitivities have been updated with the filter
	MeshBase::const_element_iterator       el     = this->get_mesh().active_local_elements_begin();
	const MeshBase::const_element_iterator end_el = this->get_mesh().active_local_elements_end();

	// Array to hold the dof indices
	std::vector<dof_id_type> densities_index;
	// Array to hold the values
	std::vector<Number> density;
	unsigned int density_var = densities.variable_number ("rho");


	Number total_volume(0.0), current_volume(0.0);
	for ( ; el != end_el ; ++el){
		const Elem * elem = *el;
		densities.get_dof_map().dof_indices (elem, densities_index, density_var);
		densities.current_local_solution->get(densities_index, density);
		Number elem_volume = elem->volume();
		total_volume += elem_volume;
		current_volume += density[0]*elem_volume;

		grad[densities_index[0]] = 1.0;
	}



	Number volume_fraction = current_volume / total_volume - volume_fraction_constraint;

	return volume_fraction;

}

void TopOptSystem::update_kernel(){

	// Number of processors
	unsigned int n_processors = this->n_processors();

	// Serialize the mesh
	MeshBase & mesh_parallel = this->get_mesh();
	mesh_parallel.allgather();

	MeshBase::const_element_iterator       el     = mesh_parallel.active_elements_begin();
	const MeshBase::const_element_iterator end_el = mesh_parallel.active_elements_end();

	std::cout<<"Printing every single element's centroid"<<std::endl;
	for ( ; el != end_el ; ++el){
		Elem * elem = *el;
		Point p = elem->centroid();
		std::cout<<p<<std::endl;
	}

	el     = mesh_parallel.active_elements_begin();

	std::queue<const Elem*> queue_elements;

	dof_id_type n_elements_kernel = mesh_parallel.n_active_elem();

	// Number of on-processor non-zeros per row
	std::vector< numeric_index_type > n_nz, n_oz;
	n_nz.resize(n_elements_kernel);
	n_oz.resize(n_elements_kernel);
	// Vector that contains an array of elements affected by the filter
	// We access it with the density_current index to see all the elements
	// affected by the filter on the density_current
	// This way, we can quickly build the matrix in the second pass
	std::vector<std::vector<const Elem*> > elements_array;
	elements_array.resize(n_elements_kernel);
	// Array to mark the visited elements
	std::vector<bool> marked_elements;
	marked_elements.resize(n_elements_kernel);

	// Get system
	ExplicitSystem& aux_system = this->get_equation_systems().get_system<ExplicitSystem>("Densities");
	// Array to hold the dof indices
	std::vector<dof_id_type> density_neighbor;
	std::vector<dof_id_type> density_current;

	for ( ; el != end_el ; ++el){
		std::fill(marked_elements.begin(),marked_elements.end(),false);
		Elem * elem = *el;
		// Get index, it will be the row number of our matrix
		aux_system.get_dof_map().dof_indices(elem, density_current, 0);
		// Add element to queue
		queue_elements.push(elem);
		// First, check the element by itself
		// Density index gives us the column for the current row in the kernel matrix
		// We have an additional non-zero for this row
		++n_nz[density_current[0]];
		// Mark it as visited
		marked_elements[density_current[0]] = true;
		elements_array[density_current[0]].push_back(elem);
		// Calculate centroid
		Point my_centroid = elem->centroid();
		while (!queue_elements.empty()){
			const Elem * elem_cola = queue_elements.front();
			queue_elements.pop();
            // Loop over the neighbors
            for (unsigned int n_p=0; n_p<elem_cola->n_neighbors(); n_p++)
            {
            	if (elem_cola->neighbor(n_p) != NULL){
					// Because the neighbor can have children next to our element,
					// we need to check for that and obtain those neighbors
					// Find the active neighbors in this direction
					std::vector<const Elem*> active_neighbors;
					elem_cola->neighbor(n_p)->active_family_tree_by_neighbor(active_neighbors,elem_cola);
					for (unsigned int a=0;
						 a != active_neighbors.size(); ++a)
					{
						const Elem *f = active_neighbors[a];
						// Get dof_index, which is like an element id
						aux_system.get_dof_map().dof_indices(f, density_neighbor, 0);
						if ( !marked_elements[density_neighbor[0]]) // not visited
						{
							// Calculate distance
							Point neighbor_centroid = f->centroid();
							neighbor_centroid.subtract(my_centroid);
							Number dist = neighbor_centroid.size();
							// Mark as visited
							marked_elements[density_neighbor[0]] = true;
							// Add to queue if within range
							if (dist<epsilon) {
								queue_elements.push(f);
								// Density index gives us the column for the current row in the kernel matrix
								// We have an additional non-zero for this row
								++n_nz[density_current[0]];
								// Mark it as visited
								elements_array[density_current[0]].push_back(f);
							}
						}
					}
            	}
            }
		}
	}


	// Allocate memory in matrix
	kernel_filter.init(n_elements_kernel,n_elements_kernel,n_elements_kernel,n_elements_kernel,n_nz,n_oz,1);
	// Second pass
	el     = mesh_parallel.active_elements_begin();

	kernel_filter.attach_dof_map(aux_system.get_dof_map());

	for ( ; el != end_el ; ++el){
		Elem * elem = *el;
		// Get index, it will be the row number of our matrix
		aux_system.get_dof_map().dof_indices(elem, density_current, 0);
		// Row where our DenseMatrix row_vector will be in the filter kernel
		std::vector<dof_id_type> rows, cols;
		std::vector<Number> values_row;
		rows.push_back(density_current[0]);

		// Iterators
		std::vector<const Elem *>::iterator el_kernel 		= elements_array[density_current[0]].begin();
		std::vector<const Elem *>::iterator el_kernel_end	= elements_array[density_current[0]].end();

		Point my_centroid = elem->centroid();
		Number normalization_coeffient = 0;
		// Build an auxiliary matrix to add an entire row to the Petsc Matrix at the same time
		DenseMatrix<Number> row_vector;
		for (; el_kernel!=el_kernel_end; ++el_kernel ){
        	const Elem *f = *el_kernel;
			// Calculate distance
			Point neighbor_centroid = f->centroid();
			neighbor_centroid.subtract(my_centroid);
			Number dist = neighbor_centroid.size_sq();
			Number value = (f->volume())*(epsilon - dist)/epsilon;
			// Get dof_index, which is like an element id
			aux_system.get_dof_map().dof_indices(f, density_neighbor, 0);
			// Update the normalization coefficient
			normalization_coeffient += value;
			// Update the columns vector
			values_row.push_back(value);
			cols.push_back(density_neighbor[0]);
		}
		row_vector.resize(1,cols.size());

		std::vector<Number>::iterator values_iterator 	= values_row.begin();

		for (unsigned i=0; i<cols.size(); i++){
			row_vector(0,i) = *values_iterator;
			++values_iterator;
		}
		/*
		 *  OJO, REVISAR COMO SE METE LA MATRIX PORQUE PUEDE HABER PROBLEMAS
		 */
		// Normalize the row vector
		row_vector *= 1.0/normalization_coeffient;
		kernel_filter.add_matrix(row_vector,rows,cols);
	}
	kernel_filter.close();

	mesh_parallel.delete_remote_elements();

	std::cout<<"Kernel finalized and built"<<std::endl;

	kernel_filter.print_personal();

	std::cout<<"Kernel printed"<<std::endl;




}
