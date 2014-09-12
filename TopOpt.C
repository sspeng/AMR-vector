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

#include "libmesh/mesh_function.h"



// Local includes
#include "TopOpt.h"
#include "resder.h"
#include <math.h>


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


	this->init_bcs();


	this->init_opt_vectors();

	// Add the kernel matrix


	// Do the parent's initialization after variables are defined
	FEMSystem::init_data();

	this->time_evolving(0);
}

void TopOptSystem::init_opt_vectors ()
{
	ExplicitSystem & densities = this->get_equation_systems().get_system<ExplicitSystem>("Densities");
	// These vectors are going to have a similar structure than the densities

    std::ostringstream densities_filt;
    densities_filt << "dens_filt";
    densities.add_vector(densities_filt.str(),false,PARALLEL);


    std::ostringstream grad;
    grad << "grad";
    densities.add_vector(grad.str(),false,PARALLEL);

    std::ostringstream grad_filt;
    grad_filt << "grad_filt";
    densities.add_vector(grad_filt.str(),false,PARALLEL);

    std::ostringstream vol_grad;
    vol_grad << "vol_grad";
    densities.add_vector(vol_grad.str(),false,PARALLEL);

    std::ostringstream vol_grad_filt;
    vol_grad_filt << "vol_grad_filt";
    densities.add_vector(vol_grad_filt.str(),false,PARALLEL);

    // Vectors with same structure than the densities field.
    // Necessary for MMA
    densities.add_vector("xmin", false, PARALLEL);
    densities.add_vector("xmax", false, PARALLEL);
    densities.add_vector("xold", false, PARALLEL);


	PetscErrorCode ierr;
    PetscVector<Number> & xold 			= dynamic_cast<PetscVector<Number> & >(densities.get_vector("xold"));
	ierr = VecDuplicateVecs(xold.vec(),1.0, &dgdx);


	// Add the kernel matrix
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

void TopOptSystem::attach_flux_bc_function (std::pair<bool,Gradient> fptr(const TopOptSystem& ,
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

void TopOptSystem::attach_body_force (Gradient fptr(const Point& ,const std::string& ))
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

	this->diff_qoi = advanced_qoi[0];
	if (!this->is_adjoint_already_solved())
	{
	  this->adjoint_solve(qoi_indices);
	}

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

	// Our vector that accounts for the residual derivative is serial (we need one per element)
	// but the adjoint vector is parallel, we need to get a serial copy to perform the dot product between them
	AutoPtr<NumericVector<Number> > local_copy_adjoint_soln = NumericVector<Number>::build(get_equation_systems().comm());
	std::vector<Number> adjoint_soln;
	adjoint_soln.resize (this->get_adjoint_solution(0).size());
	// Get copy into adjoint_soln vector
	this->get_adjoint_solution(0).localize(adjoint_soln);
	// Build serial petsc vector
	local_copy_adjoint_soln->init(this->solution->size(), true, SERIAL);
	// Copy the contents from adjoint_soln
	(*local_copy_adjoint_soln) = adjoint_soln;




	// Reference to the gradient vector where we store the values
    std::ostringstream grad;
    grad << "grad";
    NumericVector<Number> & gradient = aux_system.get_vector(grad.str());

	// Update system, localize solutions
	this->update();

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

		// Here we calculate (partial R / partial p) using our built in function and we'll compare later.
		this->assemble_res_derivative (qoi_indices, densities_index[0]);
		// Calculate dQ/dP for this parameter
		unsigned int indice = 0;
		Number parameter_derivative;
		_femcontext.elem_fe_reinit();
		advanced_qoi[indice]->element_qoi_derivative_parameter(*con,parameter_derivative,density[0]);

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

					Number sensitivity = parameter_derivative -
							this->get_resder_rhs(0).dot(*lift_func) -
							this->get_resder_rhs(0).dot(*local_copy_adjoint_soln);

					gradient.set(densities_index[0], sensitivity);
				}
				else{

					Number sensitivity = parameter_derivative -
							this->get_resder_rhs(0).dot(*local_copy_adjoint_soln);

					gradient.set(densities_index[0], sensitivity);


				}
			}
	}

	gradient.close();

	// Scale gradient
	gradient *= this->opt_scaling;

	// Sensitivities have been calculated
	sensitivities_calculated = true;

	// All parameters have been reset.
	// We didn't cache the original rhs or matrix for memory reasons,
	// but we can restore them to a state consistent solution -
	// principle of least surprise.


	//this->assembly(true, true);
	this->rhs->close();
	this->matrix->close();
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

	this->diff_qoi = advanced_qoi[0];

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
	MeshBase::const_element_iterator end_el = this->get_mesh().active_local_elements_end();

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

	for (unsigned int k=0; k<this->n_processors(); k++){
		for (; el != end_el; el++)
		{
			const Elem * elem = *el;

			_femcontext.pre_fe_reinit(*this, elem);

			// (partial q / partial p) ~= (q(p+dp)-q(p-dp))/(2*dp)
			// (partial R / partial p) ~= (rhs(p+dp) - rhs(p-dp))/(2*dp)
			// Get dof indices
			Number old_parameter;
			const Real delta_p = 1e-6;

			dof_map_densities.dof_indices(elem, densities_index, density_var);
			// Get the value, stored in density
			aux_system.current_local_solution->get(densities_index, density);
			if (k == this->processor_id()) {
				// We currently get partial derivatives via central differencing
				old_parameter = density[0];
				Number parameter_back = old_parameter - delta_p;
				aux_system.current_local_solution->set(densities_index[0],parameter_back);
			}
			this->comm().barrier();
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
			if (k == this->processor_id()) {
				aux_system.current_local_solution->set(densities_index[0],parameter_forward);
			}
			this->comm().barrier();
			aux_system.current_local_solution->close();
			this->assemble_qoi(qoi_indices);
			std::vector<Number>& qoi_plus = this->qoi;
			if (p_norm_objectivefunction){
				for (unsigned int k=0 ; k!=qoi.size(); k++)
					qoi_plus[k]= pow(qoi_plus[k], 1.0/pnorm_parameter);
			}


			if (k == this->processor_id()) {
				for (unsigned int i=0; i != Nq; ++i)
					if (qoi_indices.has_index(i))
						partialq_partialp_num[i][densities_index[0]] = (qoi_plus[i] - qoi_minus[i]) / (2.*delta_p);
			}
			this->comm().barrier();


			this->assembly(true, false);
			this->rhs->close();
			*partialR_partialp += *this->rhs;
			*partialR_partialp /= (2.*delta_p);

			// Don't leave the parameter changed
			if (k == this->processor_id()) {
				aux_system.current_local_solution->set(densities_index[0],old_parameter);
			}
			this->comm().barrier();
			aux_system.current_local_solution->close();

			// Calculate dQ/dP for this parameter
	//		if (p_norm_objectivefunction){
	//			_femcontext.elem_fe_reinit();
	//			VonMisesPnorm * vonmises = (VonMisesPnorm *) this->get_qoi();
	//			Number parameter_derivative;
	//			vonmises->element_qoi_derivative_parameter(*con,parameter_derivative,density[0]);
	//			partialq_partialp[0][densities_index[0]] = parameter_derivative;
	//		}

			// Here we calculate (partial R / partial p) using our built in function and we'll compare later.

			this->assemble_res_derivative (qoi_indices, densities_index[0]);


			this->comm().barrier();

			std::cout<<"dR/dP with analytical for element = "<<densities_index[0]<<std::endl;
			this->get_resder_rhs(0).print();

			std::cout<<"dR/dP with FD for element ="<<densities_index[0]<<std::endl;
			partialR_partialp->print_global();

			// Calculate dQ/dP for this parameter
			unsigned int indice = 0;
			Number parameter_derivative;
			_femcontext.set_elem(elem);
			_femcontext.elem_fe_reinit();
			advanced_qoi[indice]->element_qoi_derivative_parameter(*con,parameter_derivative,density[0]);


			std::cout<<"dQ/dP with FD  -----  dQ/dP with analytical for element ="<<densities_index[0]<<std::endl;
			if (k == this->processor_id()) {
				for (unsigned int i=0; i != Nq; ++i) {
					if (qoi_indices.has_index(i))
					std::cout<<partialq_partialp_num[i][densities_index[0]]<<"   		     "<<parameter_derivative<<std::endl;
				}
			}
			this->comm().barrier();
		}
		this->comm().barrier();
	}

	std::cout << "Terminamos este FD check" << std::endl;



	  std::cout << "Empezamos este FD check" << std::endl;

	  // Dof Map for the elasticity system
	  const DofMap& dof_map_elasticity = this->get_dof_map();


	  unsigned int sys_num = this->number();
	  unsigned int u_var = this->variable_number ("u");

	  // First we create a local copy of the entire solution vector
	  AutoPtr<NumericVector<Number> > local_copy_global_soln = NumericVector<Number>::build(get_equation_systems().comm());
	  AutoPtr<NumericVector<Number> > local_copy_global_density = NumericVector<Number>::build(get_equation_systems().comm());
	  AutoPtr<NumericVector<Number> > partialQ_partialU = NumericVector<Number>::build(get_equation_systems().comm());
	  // Transfer global solution to a vector
	  std::vector<Number> global_soln, global_densities;
	  this->update_global_solution(global_soln);
	  aux_system.update_global_solution(global_densities);
	  // Copy the contents into a NumericVector
	  local_copy_global_soln->init(this->solution->size(), true, SERIAL);
	  (*local_copy_global_soln) = global_soln;

	  local_copy_global_density->init(aux_system.solution->size(), true, SERIAL);
	  (*local_copy_global_density) = global_densities;

	  partialQ_partialU->init(this->solution->size(), true, SERIAL);

	  partialQ_partialU->zero();


	  this->get_mesh().allgather();
	  el     = this->get_mesh().active_elements_begin();
	  end_el = this->get_mesh().active_elements_end();

	  MeshBase::const_element_iterator       el_qoi     = this->get_mesh().active_elements_begin();
	  MeshBase::const_element_iterator       end_el_qoi = this->get_mesh().active_elements_end();

	  if (this->processor_id() == 0){
		for (unsigned int qoi_index=0; qoi_index != qoi.size(); ++qoi_index){
			for (; el != end_el; el++){


				const Elem * elem = *el;
				unsigned int elem_n_dofs = elem->n_dofs(sys_num,u_var);
				// Grab the dof indices for this element (global degree of freedom)
				std::vector<dof_id_type> indices;
				dof_map_elasticity.dof_indices(elem,indices,u_var);



				for (unsigned int i = 0; i < indices.size(); ++i){
					const Real delta_p = 1e-6;
					Number old_parameter = (*local_copy_global_soln)(indices[i]);
					Number parameter_back = old_parameter - delta_p;

					local_copy_global_soln->set(indices[i],parameter_back);
					local_copy_global_soln->close();


					Number qoi_minus = 0.0;

					el_qoi     = this->get_mesh().active_elements_begin();

					for (; el_qoi != end_el_qoi; el_qoi++){
						// Initialize context

						_femcontext.set_elem(*el_qoi);

						_femcontext.elem_fe_reinit();


						Number qoi_minus_elem = 0.0;
						this->advanced_qoi[qoi_index]->element_qoi_for_FD(local_copy_global_soln,
																			local_copy_global_density,
																			*el_qoi,
																			_femcontext,
																			qoi_minus_elem);

						Number qoi_minus_side = 0.0;
				        for (_femcontext.side = 0;
				        		_femcontext.side != _femcontext.get_elem().n_sides();
				             ++_femcontext.side)
				          {
				            // Don't compute on non-boundary sides unless requested
				            if (!this->advanced_qoi[qoi_index]->assemble_qoi_sides ||
				                (!this->advanced_qoi[qoi_index]->assemble_qoi_internal_sides &&
				                		_femcontext.get_elem().neighbor(_femcontext.side) != NULL))
				              continue;

				            _femcontext.side_fe_reinit();


							this->advanced_qoi[qoi_index]->element_qoi_for_FD(local_copy_global_soln,
																				local_copy_global_density,
																				*el_qoi,
																				_femcontext,
																				qoi_minus_side);
				          }
						qoi_minus += qoi_minus_elem + qoi_minus_side;
					}

					if (p_norm_objectivefunction){
							qoi_minus= pow(qoi_minus, 1.0/pnorm_parameter);
					}


					Number parameter_forward = old_parameter + delta_p;
					local_copy_global_soln->set(indices[i],parameter_forward);
					local_copy_global_soln->close();

					Number qoi_plus = 0.0;

					// Reinit the iterator
					el_qoi     = this->get_mesh().active_elements_begin();
					for (; el_qoi != end_el_qoi; el_qoi++){

						// Initialize context
						_femcontext.set_elem(*el_qoi);
						_femcontext.elem_fe_reinit();

						Number qoi_plus_elem = 0.0;
						this->advanced_qoi[qoi_index]->element_qoi_for_FD(local_copy_global_soln,
																			local_copy_global_density,
																			*el_qoi,
																			_femcontext,
																			qoi_plus_elem);

						Number qoi_plus_side = 0.0;
				        for (_femcontext.side = 0;
				        		_femcontext.side != _femcontext.get_elem().n_sides();
				             ++_femcontext.side)
				          {
				            // Don't compute on non-boundary sides unless requested
				            if (!this->advanced_qoi[qoi_index]->assemble_qoi_sides ||
				                (!this->advanced_qoi[qoi_index]->assemble_qoi_internal_sides &&
				                		_femcontext.get_elem().neighbor(_femcontext.side) != NULL))
				              continue;

				            _femcontext.side_fe_reinit();

							this->advanced_qoi[qoi_index]->element_qoi_for_FD(local_copy_global_soln,
																				local_copy_global_density,
																				*el_qoi,
																				_femcontext,
																				qoi_plus_side);
				          }
						qoi_plus += qoi_plus_elem + qoi_plus_side;
					}

					if (p_norm_objectivefunction){
						qoi_plus= pow(qoi_plus, 1.0/pnorm_parameter);
					}

					// Evaluate partial derivative
					Number dQdU = (qoi_plus - qoi_minus) / (2.*delta_p);


					//Set it
					partialQ_partialU->set(indices[i],dQdU);

					// Modify solution back to its original
					local_copy_global_soln->set(indices[i],old_parameter);
					local_copy_global_soln->close();
				}
			}
		}
	  }
	  this->get_mesh().delete_remote_elements();

//		for (unsigned int j=0; j != qoi.size(); ++j)
//			for (; node_begin != node_end; ++node_begin){
//				const Node * nodo = *node_begin;
//				unsigned int sys_num = this->number();
//				unsigned int u_var = this->variable_number ("u");
//				unsigned int n_components = nodo->n_comp(sys_num,u_var);
//
//				for (unsigned int k = 0; k<n_components; k++){
//
//					unsigned int i = nodo->dof_number(sys_num, u_var, k);
//					std::cout << "dof = "<<i<< std::endl;
//					const Real delta_p = 1e-6;
//
//					Number old_parameter = this->current_solution(i);
//					Number parameter_back = old_parameter - delta_p;
//
//					this->current_local_solution->set(i,parameter_back);
//					this->current_local_solution->close();
//
//					this->assemble_qoi(qoi_indices);
//
//					std::vector<Number> qoi_minus = this->qoi;
//					if (p_norm_objectivefunction){
//						for (unsigned int k=0 ; k!=qoi.size(); k++)
//							qoi_minus[k]= pow(qoi_minus[k], 1.0/pnorm_parameter);
//					}
//
//
//					Number parameter_forward = old_parameter + delta_p;
//					this->current_local_solution->set(i,parameter_forward);
//					this->current_local_solution->close();
//
//					this->assemble_qoi(qoi_indices);
//					std::vector<Number>  qoi_plus = this->qoi;
//					if (p_norm_objectivefunction){
//						for (unsigned int k=0 ; k!=qoi.size(); k++)
//							qoi_plus[k]= pow(qoi_plus[k], 1.0/pnorm_parameter);
//					}
//
//					// Evaluate partial derivative
//					Number dQdU = (qoi_plus[j] - qoi_minus[j]) / (2.*delta_p);
//
//					// Set it
//					//this->get_dqdu_fd(j).set(i,dQdU);
//
//					// Modify solution back to its original
//					this->current_local_solution->set(i, old_parameter);
//					this->current_local_solution->close();
//
//				}
//
//			}
		std::cout << "Terminamos FD check partial" << std::endl;
	// Close vectors for reuse
	partialQ_partialU->close();


	std::cout<<"dQ/dU with analytical"<<std::endl;

	this->get_adjoint_rhs(0).print_global();

	std::cout<<"dQ/dU with FD"<<std::endl;
	partialQ_partialU->print();

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

  return this->add_vector(resder_rhs_name.str(), false, SERIAL);
}

NumericVector<Number> & TopOptSystem::get_resder_rhs (unsigned int i)
{
	  std::ostringstream resder_rhs_name;
	  resder_rhs_name << "res_der" << i;

  return this->get_vector(resder_rhs_name.str());
}

void TopOptSystem::assemble_res_derivative (const QoISet& qoi_indices, const dof_id_type & elem_id)
{
  START_LOG("assemble_qoi_derivative()", "FEMSystem");

  const MeshBase& mesh = this->get_mesh();

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
                                                   this->_res_der,elem_id));

  // Close vectors for reuse
 this->get_resder_rhs(0).close();

  STOP_LOG("assemble_qoi_derivative()", "FEMSystem");
}

void TopOptSystem::calculate_qoi(const QoISet &qoi_indices)
{
  // Reset the array holding the computed QoIs
  computed_QoI[0] = 0.0;

  this->diff_qoi = advanced_qoi[0];
  FEMSystem::assemble_qoi(qoi_indices);
  std::vector<Number> qoi = this->qoi;


  computed_QoI[0] = qoi[0];

  // Scale obj function

  computed_QoI[0] *= this->opt_scaling;


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


	// Grab gradient vector

    std::ostringstream grad;
    grad << "grad";
    NumericVector<Number> & gradient = aux_system.get_vector(grad.str());
	std::cout<<"Comparing global sensitivities with Finite Differences"<<std::endl;

	for (unsigned int i=0; i<this->n_processors(); i++){

			for (; el != end_el; el++)
			{
				this->comm().barrier();
				const Elem * elem = *el;
				std::vector<Number> partialq_partialp(Nq, 0);
				// (partial q / partial p) ~= (q(p+dp)-q(p-dp))/(2*dp)
				// Get dof indices
				Number old_parameter;

				if (i == this->processor_id()) {
					dof_map_densities.dof_indices(elem, densities_index, density_var);
					// Get the value, stored in density
					aux_system.solution->get(densities_index, density);

					old_parameter = density[0];
					Number parameter_back = old_parameter - delta_p;
					aux_system.solution->set(densities_index[0],parameter_back);
				}
				this->comm().barrier();

				aux_system.solution->close();
				aux_system.update();

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


				if (i == this->processor_id()) {
					aux_system.solution->set(densities_index[0],parameter_forward);
				}
				this->comm().barrier();
				aux_system.solution->close();
				aux_system.update();
				// Solve system to obtain the new QoI
				this->solve();
				// Calculate perturbed QoI
				this->assemble_qoi(qois);
				std::vector<Number>& qoi_plus = this->qoi;

				if (p_norm_objectivefunction){
					for (unsigned int k=0 ; k!=qoi.size(); k++)
						qoi_plus[k]= pow(qoi_plus[k], 1.0/pnorm_parameter);
				}

				if (i == this->processor_id()) {
					// Don't leave the parameter changed
					aux_system.solution->set(densities_index[0],old_parameter);
				}
				this->comm().barrier();
				aux_system.solution->close();
				aux_system.update();

				std::cout<<"Sensitivities with FD  -----  Sensitivities with analytical"<<std::endl;
				if (i == this->processor_id()) {
					for (unsigned int i=0; i != Nq; ++i) {
						if (qois.has_index(i)){
							sensitivities_fd[i][densities_index[0]] = (qoi_plus[i] - qoi_minus[i]) / (2.*delta_p);
							// Scale it
							sensitivities_fd[i][densities_index[0]] *= this->opt_scaling;
						}
						std::cout<<sensitivities_fd[i][densities_index[0]]<<"   		     "<<gradient.el(densities_index[0])<<std::endl;
					}
				}
			}
	}


}

void TopOptSystem::transfer_densities(ExplicitSystem & densities, const std::vector<double> & x, const bool & filter){

	// First, we need to transfer the densities from the solution to the global solution
	// vector in the densities system. Because x hsa the same size than the global vector,
	// only the local components are copied.
	*(densities.solution) = x;
	// Should call close and update when we set vector entries directly
	densities.solution->close();
	densities.update();

	// Copy them back to the original densities field


    std::ostringstream densities_filt;
    densities_filt << "dens_filt";
    NumericVector<Number> & densities_filtered = densities.get_vector(densities_filt.str());

    densities_filtered.zero();

	if (filter){
		// Apply filter
		kernel_filter_parallel.vector_mult(densities_filtered, *densities.solution.get());

		// Copy them back to densities solution

		*(densities.solution) = densities_filtered;

		// Should call close and update when we set vector entries directly
		densities.solution->close();
		densities.update();
	}

}

void TopOptSystem::filter_gradient(){


	ExplicitSystem & densities =this->get_equation_systems().get_system<ExplicitSystem>("Densities");


    std::ostringstream gradiente;
    gradiente << "grad";
    NumericVector<Number> & gradient = densities.get_vector(gradiente.str());
    std::ostringstream grad_filt;
    grad_filt << "grad_filt";
    NumericVector<Number> & gradient_filtered = densities.get_vector(grad_filt.str());

	gradient_filtered.zero();

	// Perform product of gradient_filtered = (kernel_filter_parallel)^T * gradient (right now they're just the derivatives)
	gradient_filtered.add_vector_transpose(gradient, kernel_filter_parallel);

	gradient_filtered.close();

}

Number TopOptSystem::calculate_filtered_volume_constraint(ExplicitSystem & densities){


	// Sensitivities have been updated with the filter
	MeshBase::const_element_iterator       el     = this->get_mesh().active_local_elements_begin();
	const MeshBase::const_element_iterator end_el = this->get_mesh().active_local_elements_end();

	// Array to hold the dof indices
	std::vector<dof_id_type> densities_index;
	// Array to hold the values
	std::vector<Number> density;
	unsigned int density_var = densities.variable_number ("rho");


	Number total_volume(0.0), current_volume(0.0);
    std::ostringstream vol_grad, vol_grad_filt;
    vol_grad << "vol_grad";
    vol_grad_filt << "vol_grad_filt";

    NumericVector<Number> & volume_gradient = densities.get_vector(vol_grad.str());
    NumericVector<Number> & volume_gradient_filtered = densities.get_vector(vol_grad_filt.str());


	for ( ; el != end_el ; ++el){
		const Elem * elem = *el;
		densities.get_dof_map().dof_indices (elem, densities_index, density_var);
		densities.current_local_solution->get(densities_index, density);
		Number elem_volume = elem->volume();
		total_volume += elem_volume;
		current_volume += density[0]*elem_volume;
		volume_gradient.set(densities_index[0],elem_volume);
	}

	volume_gradient.close();

	// Perform product of gradient_filtered_temp = (kernel_filter_parallel)^T * gradient_filtered (right now they're just the derivatives)
	volume_gradient_filtered.zero();
	volume_gradient_filtered.add_vector_transpose(volume_gradient, kernel_filter_parallel);

	// We need to collect both the current_volume and total_volume from all processors
	this->comm().sum(current_volume);
	this->comm().sum(total_volume);
	Number volume_fraction = current_volume - volume_fraction_constraint*total_volume;

	return volume_fraction;

}

void TopOptSystem::update_kernel(){

	// Get system
	ExplicitSystem& aux_system = this->get_equation_systems().get_system<ExplicitSystem>("Densities");

	// Number of processors
	unsigned int n_processors = this->n_processors();

	// Serialize the mesh
	MeshBase & mesh_parallel = this->get_mesh();
	mesh_parallel.allgather();

	MeshBase::const_element_iterator       el     = mesh_parallel.active_local_elements_begin();
	const MeshBase::const_element_iterator end_el = mesh_parallel.active_local_elements_end();

	std::queue<const Elem*> queue_elements;

	dof_id_type n_elements_kernel_local = aux_system.n_local_dofs();
	dof_id_type n_elements_kernel = aux_system.n_dofs();

	// Number of on-processor non-zeros per row
	std::vector< numeric_index_type > n_nz, n_oz;
	n_nz.resize(n_elements_kernel_local);
	n_oz.resize(n_elements_kernel_local);
	// Vector that contains an array of elements affected by the filter
	// We access it with the density_current index to see all the elements
	// affected by the filter on the density_current
	// This way, we can quickly build the matrix in the second pass
	std::vector<std::vector<const Elem*> > elements_array;
	elements_array.resize(n_elements_kernel_local);
	// Array to mark the visited elements
	std::vector<bool> marked_elements;
	marked_elements.resize(n_elements_kernel);


	// Array to hold the dof indices
	std::vector<dof_id_type> density_neighbor;
	std::vector<dof_id_type> density_current;

	numeric_index_type first_local_index = aux_system.get_dof_map().first_dof();
	numeric_index_type last_local_index = aux_system.get_dof_map().last_dof();


	PetscVector<Number> * local_copy_global_density_auxiliar = (PetscVector<Number> *) aux_system.solution.get();


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

		// We need the local dof first
		numeric_index_type local_element_dof = local_copy_global_density_auxiliar->map_global_to_local_index(density_current[0]);

		++n_nz[local_element_dof];
		// Mark it as visited
		marked_elements[density_current[0]] = true;
		elements_array[local_element_dof].push_back(elem);
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
								// We need to know whether this non-zero is in the processor or not,
								// it depends on where the element is located in the solution vector
								// of the densities system
								if (density_neighbor[0] <= last_local_index && density_neighbor[0] >=first_local_index)
									++n_nz[local_element_dof];
								else
									++n_oz[local_element_dof];
								// Mark it as visited
								elements_array[local_element_dof].push_back(f);
							}
						}
					}
            	}
            }
		}
	}

	unsigned int local_size_matrix = aux_system.n_local_dofs();

	// Clear initialized matrices
	if (kernel_filter_parallel.initialized())
		kernel_filter_parallel.clear();

	// Allocate memory in matrix
	kernel_filter_parallel.init(n_elements_kernel,n_elements_kernel,local_size_matrix,local_size_matrix,n_nz,n_oz);
	std::cout<<"success"<<std::endl;
	// Second pass
	el     =  mesh_parallel.active_local_elements_begin();

	for ( ; el != end_el ; ++el){
		Elem * elem = *el;
		// Get index, it will be the row number of our matrix
		aux_system.get_dof_map().dof_indices(elem, density_current, 0);
		// Row where our DenseMatrix row_vector will be in the filter kernel

		// We need the local dof first
		numeric_index_type local_element_dof = local_copy_global_density_auxiliar->map_global_to_local_index(density_current[0]);

		std::vector<dof_id_type> rows, cols;
		std::vector<Number> values_row;
		rows.push_back(density_current[0]);



		// Iterators
		std::vector<const Elem *>::iterator el_kernel 		= elements_array[local_element_dof].begin();
		std::vector<const Elem *>::iterator el_kernel_end	= elements_array[local_element_dof].end();

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

		kernel_filter_parallel.add_matrix(row_vector,rows,cols);
	}
	kernel_filter_parallel.close();

	mesh_parallel.delete_remote_elements();

	kernel_filter_parallel.attach_dof_map(aux_system.get_dof_map());


	std::cout<<"Kernel finalized and built"<<std::endl;

}
