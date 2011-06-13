/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010 by Joel Andersson, Moritz Diehl, K.U.Leuven. All rights reserved.
 *
 *    CasADi is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    CasADi is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with CasADi; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

#include <iostream>
#include <cstdlib>

#include <casadi/stl_vector_tools.hpp>
#include <casadi/sx/sx_tools.hpp>
#include <casadi/fx/sx_function.hpp>
//#include <casadi/fx/jacobian.hpp>


#include "Ode.hpp"
#include "Ocp.hpp"
#include "OcpMultipleShooting.hpp"

#include <SnoptInterface.hpp>

#include <string>
#include <map>

using namespace CasADi;
using namespace std;

void
dxdt(map<string,SX> &xDot, map<string,SX> state, map<string,SX> action, map<string,SX> param, SX t __attribute__((unused)))
{
	// constants
	double AR = 6;     // aspect ration
	double Cd0 = 0.03; // parasitic drag
	double m = 2.0;    // mass
	double rho = 1.22; // air density
	double A = 1.0;    // area
	double g = 9.8;    // acceleration due to gravity
	
	// eom
	SX alpha = action["alphaDeg"]*3.14159/180.0;
	SX CL = 2.0*3.14159*alpha;
	SX Cd = Cd0 + 1.1*CL*CL/(3.14159*AR);

	// airspee
	SX vx = state["vx"];
	SX vz = state["vz"];
	SX norm_v = sqrt( vx*vx + vz*vz );
        
//        Lift = 0.5*rho*A*norm_v*CL*[ vz, -vx];
//        Drag = 0.5*rho*A*norm_v*Cd*[-vx, -vz];
	SX cAero = 0.5*rho*A*norm_v;
	SX Fx = cAero*( CL*vz - Cd*vx);
	SX Fz = cAero*(-CL*vx - Cd*vz);
	SX ax = Fx/m;
	SX az = Fz/m + g;

	xDot["x"] = vx;
	xDot["z"] = vz;
	xDot["vx"] = ax;
	xDot["vz"] = az;
}

int
main()
{
	Ode ode("glider");
	ode.addState("x");
	ode.addState("z");
	ode.addState("vx");
	ode.addState("vz");
	ode.addAction("alphaDeg");
	ode.addParam("tEnd");

	ode.dxdt = &dxdt;

	OcpMultipleShooting ocp(&ode);

	ocp.discretize(200);

	SX tEnd = ocp.getParam("tEnd");
	ocp.setTimeInterval(1.0, tEnd);
	ocp.objFun = -tEnd; // maximum time
	//ocp.objFun = -ocp.getState("x", ocp.N-1); // maximum distance

	// Bounds/initial condition
	ocp.boundParam("tEnd", 2, 200);
	for (int k=0; k<ocp.N; k++){
		ocp.boundStateAction("x", 0, 1e3, k);
		ocp.boundStateAction("z", -100, 0, k);
		ocp.boundStateAction("vx", 0, 100, k);
		ocp.boundStateAction("vz", -100, 100, k);

		ocp.boundStateAction("alphaDeg", -15, 15, k);
	}

	#define INCLINATION0_DEG 0.0
	#define V0 20.0
	ocp.boundStateAction("x", 0, 0, 0);
	ocp.boundStateAction("z", 0, 0, 0);
	ocp.boundStateAction("vx",  V0*cos(INCLINATION0_DEG*M_PI/180.0),  V0*cos(INCLINATION0_DEG*M_PI/180.0),  0);
	ocp.boundStateAction("vz", -V0*sin(INCLINATION0_DEG*M_PI/180.0), -V0*sin(INCLINATION0_DEG*M_PI/180.0),  0);

	// initial guesses
	ocp.setParamGuess("tEnd", 80);
	for (int k=0; k<ocp.N; k++){
		ocp.setStateActionGuess("x", double(k)/double(ocp.N)*20, k);
		ocp.setStateActionGuess("z", double(k)/double(ocp.N)*0, k);
		ocp.setStateActionGuess("vx", 4, k);
	}

	// Create the NLP solver
	SnoptInterface si(ocp);

	si.run();

	// Print the optimal cost
	cout << "optimal time: " << -si.F[0] << endl;
	
	// Print the optimal solution
	// vector<double>xopt(ocp.getBigN());
	// solver.getOutput(xopt,NLP_X_OPT);
	//cout << "optimal solution: " << xopt << endl;

	ocp.writeMatlabOutput( si.x );

	return 0;
}
