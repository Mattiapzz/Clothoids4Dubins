/****************************************************************************\
  Copyright (c) Enrico Bertolazzi 2014
  All Rights Reserved.

  This program is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation;

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
\****************************************************************************/

#include "Clothoid.hh"
#include "mex.h"

#include <sstream>
#include <stdexcept>

#define MEX_ERROR_MESSAGE \
"%=============================================================================%\n" \
"%  buildClothoid:  Compute parameters of the G1 Hermite clothoid fitting      %\n" \
"%                                                                             %\n" \
"%  USAGE: [S0,S1,flg] = buildClothoid2arcG2( x0, y0, th0, k0,                 %\n" \
"%                                            x1, y1, th1, k1 ) ;              %\n" \
"%                                                                             %\n" \
"%  On input:                                                                  %\n" \
"%                                                                             %\n" \
"%       x0, y0  = coodinate of initial point                                  %\n" \
"%       theta0  = orientation (angle) of the clothoid at initial point        %\n" \
"%       k0      = initial curvature                                           %\n" \
"%       x1, y1  = coodinate of final point                                    %\n" \
"%       theta1  = orientation (angle) of the clothoid at final point          %\n" \
"%                                                                             %\n" \
"%  On output:                                                                 %\n" \
"%       S0     = initial arc of clothoid                                      %\n" \
"%       S1     = final arc of clothoid                                        %\n" \
"%       flg    = true:  computation success                                   %\n" \
"%                false: computation failed                                    %\n" \
"%                                                                             %\n" \
"%=============================================================================%\n" \
"%                                                                             %\n" \
"%  Autors: Enrico Bertolazzi and Marco Frego                                  %\n" \
"%          Department of Industrial Engineering                               %\n" \
"%          University of Trento                                               %\n" \
"%          enrico.bertolazzi@unitn.it                                         %\n" \
"%          m.fregox@gmail.com                                                 %\n" \
"%                                                                             %\n" \
"%=============================================================================%\n"

#define ASSERT(COND,MSG)                          \
  if ( !(COND) ) {                                \
    std::ostringstream ost ;                      \
    ost << "buildClothoid2arc: " << MSG << '\n' ; \
    mexErrMsgTxt(ost.str().c_str()) ;             \
  }

#define arg_x0     prhs[0]
#define arg_y0     prhs[1]
#define arg_theta0 prhs[2]
#define arg_kappa0 prhs[3]
#define arg_x1     prhs[4]
#define arg_y1     prhs[5]
#define arg_theta1 prhs[6]
#define arg_kappa1 prhs[7]

#define arg_S0     plhs[0]
#define arg_S1     plhs[1]
#define arg_flg    plhs[2]

static
void
save_struct( Clothoid::ClothoidCurve const & curve, mxArray * & plhs ) {
  char const * fieldnames[] = { "x", "y", "theta", "k", "dk", "L" } ;
  plhs = mxCreateStructMatrix(1,1,6,fieldnames);
  mxSetFieldByNumber( plhs, 0, 0, mxCreateDoubleScalar(curve.getX0()) );
  mxSetFieldByNumber( plhs, 0, 1, mxCreateDoubleScalar(curve.getY0()) );
  mxSetFieldByNumber( plhs, 0, 2, mxCreateDoubleScalar(curve.getTheta0()) );
  mxSetFieldByNumber( plhs, 0, 3, mxCreateDoubleScalar(curve.getKappa()) );
  mxSetFieldByNumber( plhs, 0, 4, mxCreateDoubleScalar(curve.getKappa_D()) );
  mxSetFieldByNumber( plhs, 0, 5, mxCreateDoubleScalar(curve.getSmax()) );
}

extern "C"
void
mexFunction( int nlhs, mxArray       *plhs[],
             int nrhs, mxArray const *prhs[] ) {

  // Check for proper number of arguments, etc
  if ( nrhs != 8 ) {
	  mexErrMsgTxt(MEX_ERROR_MESSAGE) ;
    return ;
  } else if ( mxGetClassID(arg_x0)     != mxDOUBLE_CLASS ||
              mxGetClassID(arg_y0)     != mxDOUBLE_CLASS ||
              mxGetClassID(arg_theta0) != mxDOUBLE_CLASS ||
              mxGetClassID(arg_kappa0) != mxDOUBLE_CLASS ||
              mxGetClassID(arg_x1)     != mxDOUBLE_CLASS ||
              mxGetClassID(arg_y1)     != mxDOUBLE_CLASS ||
              mxGetClassID(arg_theta1) != mxDOUBLE_CLASS ||
              mxGetClassID(arg_kappa1) != mxDOUBLE_CLASS ) {
	  mexErrMsgTxt("Input arguments should be double");
  }

  for ( int kk = 0 ; kk < 8 ; ++kk )
    if ( mxGetM(prhs[kk]) != 1 || mxGetN(prhs[kk]) != 1 )
	    mexErrMsgTxt("Input arguments must be scalars");

  ASSERT( nlhs == 3,
          "wrong number of output arguments\n"
          "expected 3, found " << nlhs ) ;

  Clothoid::G2solve2arc g2solve2arc ;

  Clothoid::valueType x0  = mxGetScalar(arg_x0) ;
  Clothoid::valueType y0  = mxGetScalar(arg_y0) ;
  Clothoid::valueType th0 = mxGetScalar(arg_theta0) ;
  Clothoid::valueType k0  = mxGetScalar(arg_kappa0) ;
  Clothoid::valueType x1  = mxGetScalar(arg_x1) ;
  Clothoid::valueType y1  = mxGetScalar(arg_y1) ;
  Clothoid::valueType th1 = mxGetScalar(arg_theta1) ;
  Clothoid::valueType k1  = mxGetScalar(arg_kappa1) ;

  g2solve2arc.setup( x0, y0, th0, k0, x1, y1, th1, k1 ) ;
  bool ok = g2solve2arc.solve() ;

  if ( ok ) arg_flg = mxCreateLogicalScalar(1);
  else      arg_flg = mxCreateLogicalScalar(0);

  save_struct( g2solve2arc.getS0(), arg_S0 ) ;
  save_struct( g2solve2arc.getS1(), arg_S1 ) ;

}