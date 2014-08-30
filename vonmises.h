
#ifndef VONMISES_H
#define VONMISES_H_H

#include "libmesh/libmesh_common.h"
#include "libmesh/elem.h"
#include "libmesh/fe_base.h"
#include "libmesh/fem_context.h"
#include "libmesh/point.h"
#include "libmesh/quadrature.h"
#include "libmesh/diff_qoi.h"
#include "libmesh/explicit_system.h"


// Bring in everything from the libMesh namespace
using namespace libMesh;

class TopOptSystem;

class VonMisesPnorm : public DifferentiableQoI
{
public:
	VonMisesPnorm(ExplicitSystem * densities, TopOptSystem * femsystem, QoISet * qois);
	virtual ~VonMisesPnorm(){}

	virtual void init_qoi( std::vector<Number>& sys_qoi );

	virtual void postprocess( ){}

	virtual void element_qoi_derivative(DiffContext &context, const QoISet & qois);

	virtual void element_qoi (DiffContext &context, const QoISet & qois);

	void element_qoi_derivative_parameter (DiffContext &context, Number & parameter_derivative, Number density_element);

	void SIMP_function(Number & density){
		Number phi = density/(0.3*(1 - density) + density);

		density = phi;
	}

	void SIMP_function_derivative(Number & density){
		Number phi = 0.3/((0.3*(1 - density) + density)*(0.3*(1 - density) + density));

		density = phi;
	}



	virtual AutoPtr<DifferentiableQoI> clone( ) {
		return AutoPtr<DifferentiableQoI> ( new VonMisesPnorm(*this) );
	}

protected:

	ExplicitSystem * _densities;

	TopOptSystem * _femsystem;

	QoISet * _qois;

	// Matrix to keep the product of the Pdev and the Cmat
	DenseMatrix<Number> PdevCMat;

	unsigned int von_mises_index;



};
#endif // L_QOI_H
