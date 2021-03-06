

#ifndef TOPOPT_H
#define TOPOPT_H


#include "libmesh/enum_fe_family.h"
#include "libmesh/fem_system.h"

#include "libmesh/system.h"


#include "libmesh/sensitivity_data.h"
#include "libmesh/parameter_vector.h"
#include "libmesh/qoi_set.h"
#include "libmesh/petsc_matrix.h"
#include "libmesh/explicit_system.h"
#include "libmesh/petsc_vector.h"


#include <iostream>
#include <utility>

#include "resder.h"

// Bring in everything from the libMesh namespace
using namespace libMesh;


// FEMSystem, TimeSolver and  NewtonSolver will handle most tasks,
// but we must specify element residuals
class TopOptSystem : public FEMSystem
{
public:
  // Constructor
	TopOptSystem(EquationSystems& es,
                const std::string& name_in,
                const unsigned int number_in)
    :   FEMSystem(es, name_in, number_in),
    	kernel_filter_parallel(this->comm()),
      _fe_family("LAGRANGE_VEC"), _fe_order(1),
      _analytic_jacobians(true){
	}

  std::string & fe_family() { return _fe_family;  }
  unsigned int & fe_order() { return _fe_order;  }
  bool & analytic_jacobians() { return _analytic_jacobians; }

  // Context initialization
  virtual void init_context (DiffContext &context);



  /*
   * Evaluate the elasticity tensor or the tangent stiffness tensor
   */
  void EvalElasticity(DenseMatrix<Real> & CMat);

  /**
   * Register a user function to use in computing the flux BCs.
   * The return value is std::pair<bool, Real>
   */
  void attach_flux_bc_function (std::pair<bool,Gradient> fptr(const TopOptSystem& system,
                                                          const Point& p,
                                                          const std::string& var_name));

  /**
   * Register a user function to use in computing the body force.
   * The return value is a Gradient
   */
  void attach_body_force (Gradient fptr( const Point& p,const std::string& var_name));


  /*
   * Overload this function for implicit_system because we want an analytical way to calculate
   * the partial derivatives of the residual and the QoI
   */
  virtual void adjoint_qoi_parameter_sensitivity(const QoISet&          qoi_indices,
																	  const ParameterVector& parameters,
																	  SensitivityData&       sensitivities);

  /*
   * Assemble the partial derivative of the residual with respect to the parameter
   * This calculates only the derivative with respect to one parameter. It returns
   * a vector of length equal to the number of degrees of freedom
   */
  void assemble_res_derivative (const QoISet& qoi_indices, const dof_id_type & elem_id);

  /*
   * We have several vectors that have the same structure than the densities field, but we don't have
   * to be projected to the new refined mesh, just to be resized. This has to be called after the
   * equations.reinit() in the AMR
   */
  void init_opt_vectors();

  NumericVector<Number> & add_resder_rhs (unsigned int i);

  NumericVector<Number> & get_resder_rhs (unsigned int i);

  /*
   * RAMP functions. Necessary to obtain a 0-1 solution.
   * Ideally, write here all the possible functions (SIMP and others)
   * Set up from the file general.in which one will be called when we calculate
   * the stiffness matrix, the residual and its derivative
   */
  void ramp(Number & phi){

	  Number density = phi/(1.0+_ramp_parameter*(1.0-phi));

	  //Number density = pow(phi,_ramp_parameter);

	  phi = density;



  }


  void ramp_deriv(Number & phi){

	  Number density = (1.0+_ramp_parameter)/((1.0+_ramp_parameter*(1.0-phi))*(1.0+_ramp_parameter*(1.0-phi)));


	  //Number density = _ramp_parameter*pow(phi,_ramp_parameter-1);

	  phi = density;
  }

  void SIMP_function(Number & density){
	  Number phi = density/(0.3*(1 - density) + density);

	  density = phi;
  }

  void SIMP_function_derivative(Number & density){
	  Number phi = 0.3/((0.3*(1 - density) + density)*(0.3*(1 - density) + density));

	  density = phi;
  }


  /*
   * Setters
   */
  void set_ramp_parameter(Number & ramp_parameter)
  	  {_ramp_parameter = ramp_parameter;}

  void set_elasticity_modules(Number & lambda, Number & mu)
  	  {_lambda = lambda; _mu = mu;}

  /*
   * Getters
   */

  Number &get_parameter_value(unsigned int parameter_index)
  {
    return parameters[parameter_index];
  }

  ParameterVector &get_parameter_vector()
  {
    parameter_vector.resize(this->get_mesh().n_active_elem());
    return parameter_vector;
  }



  /*
   * Calculate filter matrix
   */
  void update_kernel();


  /*
   * Calculate QoI's
   */
  void calculate_qoi(const QoISet &qoi_indices);

  Number &get_QoI_value(unsigned int QoI_index)
  {
	  return computed_QoI[QoI_index];
  }


	/*
	* Finite Difference for the global sensitivities.
	*/
  void finite_differences_check(ExplicitSystem * densities,
								const QoISet & qois,
								const SensitivityData&       sensitivities,
								const ParameterVector& parameters);

  void finite_differences_partial_derivatives_check(const QoISet&          qoi_indices,
													   const ParameterVector& parameters,
													   SensitivityData&       sensitivities,
													   bool & p_norm_objectivefunction);

  void attach_advanced_qoi( DifferentiableQoI* qoi_in )
  { advanced_qoi.push_back((qoi_in->clone()).release());
    // User needs to resize qoi system qoi accordingly
  advanced_qoi.back()->init_qoi( this->qoi );}

  // Some parameters that are public
  // Parameter for the filter
  Number epsilon;

  // Parameter for the p-norm of the objective function
  // This should be inside the objective function I think
  // Same with the other two.
  unsigned int pnorm_parameter;
  Number von_mises_qoi;
  bool von_mises_qoi_bool;


  void filter_gradient();

  void transfer_densities(ExplicitSystem & densities, const std::vector<double> & x, const bool & filter);

  /*
   * The constraint is implemented as follows
   * 	V  	: total volume of the structure, without considering the densities
   * 	Vd 	: total volume of the structure considering the densities, already filtered
   * 	Vf	: volume fraction that is the upper boundary for the volume fraction of our structure
   * 	g(x) = Vd / V - Vf <= 0
   *
   * 	The gradient is just an array with the volume of the element for that gradient component
   */
  Number calculate_filtered_volume_constraint(ExplicitSystem & densities);

  // Matrix that contains the filter to be applied to the densities
  PetscMatrix<Number> kernel_filter_parallel;


  // Counter for the number of iterations for the optimization
  unsigned int contador = 0;

  // Volume constraint. Relative volume. Over 1
  Number volume_fraction_constraint;

  // Is our QoI a p-norm?
  bool p_norm_objectivefunction = false;

  bool read_solution_from_file = false;

  bool output_solution_to_file = false;


  Real opt_scaling, traction_force;

  Vec * dgdx;


protected:

  /*
   * Array that contains the advaced diff qoi
   */
  std::vector<DifferentiableQoI *> advanced_qoi;

  // System initialization
  virtual void init_data ();

  // Boundary conditions
  void init_bcs();

  // Element residual and jacobian calculations
  // Time dependent parts
  virtual bool element_time_derivative (bool request_jacobian,
                                        DiffContext &context);

  // Constraint parts
  virtual bool side_constraint (bool request_jacobian,
                                DiffContext &context);


  // Parameter for the RAMP function
  Real _ramp_parameter;



  // Parameters associated with the system
  std::vector<Number> parameters;

  // The ParameterVector object that will contain pointers to
  // the system parameters
  ParameterVector parameter_vector;

  // The FE type to use
  std::string _fe_family;
  unsigned int _fe_order;

  // Indices for each variable;
  unsigned int u_var;

  // Calculate Jacobians analytically or not?
  bool _analytic_jacobians;

  // Matrices used in element_time_derivative
  // Create CMat Matrices
  DenseMatrix<Real> CMat, DNhat, Nhat;
  DenseVector<Real> stress_flux_vect;

  // Body force gradient
  Gradient f;

  /**
   * Pointer to function that returns BC information.
   */
  std::pair<bool,Gradient> (* _bc_function) (const TopOptSystem& system,
                                         const Point& p,
                                         const std::string& var_name);

  /**
   * Pointer to function that holds the body force information
   */
  Gradient (* _body_force) (const Point& p, const std::string& var_name);

  // ResidualDerivative is going to access protected members of this class.
  // That's why we friend it
  friend class ResidualDerivative;

  // Pointer to the ResidualDerivative class
  ResidualDerivative _res_der;

  // Array to hold the objective functions
  Number computed_QoI[1];

  // Elasticity constants
  Number _lambda, _mu;



  // bool necessary for the finite difference check
  bool sensitivities_calculated = false;

  // Contributions from tractions?
  bool integrate_boundary_sides = false;
  // Contributions from body force?
  bool integrate_body_force = false;

  /*
   *
   * Stiffness matrix recycling
   *
   *
   */

  /*
   * Initial stiffness matrix we will reuse
   */
  DenseMatrix<Number> K_initial;
  Number detJ_initial, rho_initial;
  bool _is_K_initial_calculated;

  PerfLog perf_log;
};

#endif
