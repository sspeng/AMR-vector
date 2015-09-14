/*
 * elasticity_exact_solution.h
 *
 *  Created on: Jun 30, 2014
 *      Author: miguel
 */

#ifndef ELASTICITY_EXACT_SOLUTION_H_
#define ELASTICITY_EXACT_SOLUTION_H_

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

#include "libmesh/libmesh_common.h"

using namespace libMesh;

#ifndef __Elasticity_exact_solution_h__
#define __Elasticity_exact_solution_h__

class ElasticityExactSolution
{
public:
  ElasticityExactSolution(){}

  ~ElasticityExactSolution(){}

  Real operator()( unsigned int component,
                   Real x, Real y, Real z = 0.0)
  {
    switch(component)
      {
      case 0:
        return x*(x-1)*y*(y-1);

      case 1:
        return x*(x-1)*y*(y-1);

      case 2:
        return 0;

      default:
        libmesh_error_msg("Invalid component = " << component);
      }
  }
};


class ElasticityExactGradient
{
public:
  ElasticityExactGradient(){}

  ~ElasticityExactGradient(){}

  RealGradient operator()( unsigned int component,
                           Real x, Real y, Real z = 0.0)
  {
    switch(component)
      {
      case 0:
        return RealGradient( y*(y-1)*(2*x-1),
        					x*(x-1)*(2*y-1),
                             0);

      case 1:
          return RealGradient( y*(y-1)*(2*x-1),
          					x*(x-1)*(2*y-1),
                               0);

      case 2:
          return RealGradient( 0, 0, 0);

      default:
        libmesh_error_msg("Invalid component = " << component);
      }
  }
};

#endif // __Elasticity_exact_solution_h__




#endif /* ELASTICITY_EXACT_SOLUTION_H_ */
