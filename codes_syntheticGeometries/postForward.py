#-----------------------------# IMPORT LIBRARIES #-----------------------------#
#!/usr/bin/env python
from __future__ import generators
import sys
import time
from mshr import *
from collections import OrderedDict
from dolfin import *
from dolfin_adjoint import *
from problem import baseProblem
from solvers import Coupled, IPCS
import argparse
import numpy as np
from os.path import join
from utils import *

#-----------------------------# SETTINGS: how to load .json file #-----------------------------#
# The .json file contains the values of all the important variables of the project.
# It's enough to modify the.json file in order to run different kind of simulations.
parser = argparse.ArgumentParser()
parser.add_argument('--v', type=str, default='0', help='version') # N.B. the default version is "v0"
args = parser.parse_args()
cf = load_conf_file(f'v{int(args.v)}.json')

#-----------------------------# EXPERIMENT PARAMETERS #-----------------------------#
# These parameters define the directories used during the simulation process.
mesh_file = cf.exp_params.meshfile  # Directory where the mesh file is located.
forwardDir = cf.exp_params.forwardDir  # Directory where the forward simulation results are saved.
inletDir = cf.exp_params.inletDir  # Directory where the inlet boundary condition file is located.
obsDir = cf.exp_params.obsDir  # Directory where the observation data is stored.
logDir = cf.exp_params.logDir  # Directory for the .log file created during optimization (contains callbacks and progress logs).
patient = cf.exp_params.patient  # OBSOLETE: Previously used to point to a specific mesh file.

#-----------------------------# MODEL PARAMETERS #-----------------------------#
saveinterval = cf.exp_params.saveinterval  # Number of iterations after which results are saved during optimization.

# Dimensionality and physical properties.
meshD = cf.model_params.meshD  # Dimensionality of the problem (2D or 3D).
profile = cf.model_params.profile  # Velocity profile at the inlet (Parabolic or Plug flow).
inletID = cf.model_params.inletID  # Tag associated with the inlet boundary condition.
rho = cf.model_params.rho  # Blood density (kg/m³).
nu = cf.model_params.nu  # Kinematic viscosity of the blood (m²/s).
theta = cf.model_params.theta  # Theta parameter used in time-stepping schemes.
dt = cf.model_params.dt  # Time step for the solver loop.

obs_dt = cf.model_params.obs_dt  # Time step used for observations.
frames = cf.model_params.frames  # Number of observation frames (obs_dt * frames = total simulation time T).

# Boundary condition settings.
ObsUmax = cf.model_params.ObsUmax  # Maximum velocity value for the observation profile.
InletUmax = cf.model_params.InletUmax  # Maximum velocity value for the inlet boundary condition.

# Solver parameters.
generate = cf.model_params.generate  # OBSOLETE: Previously used to switch between boundary generation and simulation.
radius = cf.model_params.radius  # OBSOLETE: Radius of the 3D mesh (if applicable).
stationary = cf.model_params.stationary  # Simulation type: Stationary or TimeVariant.
solver = cf.model_params.solver  # Solver type (Coupled or IPCS).
control = cf.model_params.control  # Defines the control parameter (e.g., inlet).
weakControl = cf.model_params.weakControl  # Boolean for enabling weak control (using the Neitzche method).
tractionFree = cf.model_params.tractionFree  # Boolean for enabling the traction-free condition.
linear = cf.model_params.linear  # Boolean to toggle between linear and nonlinear simulations.
BDF2 = cf.model_params.BDF2  # Boolean for using Backward Differentiation Formulas (BDF2) for time-stepping.
minMethod = cf.model_params.minMethod  # Minimization method (e.g., BFGS, L-BFGS, or TNC).
minIt = cf.model_params.minIt  # Number of iterations for the minimization process.

#-----------------------------# READ STABILIZATION PARAMS #-----------------------------#
beta = cf.stab_params.beta
convective = cf.stab_params.convective
backflow = cf.stab_params.backflow
simvascularstab = cf.stab_params.simvascularstab
nitsche_gamma = cf.stab_params.nitsche_gamma

#-----------------------------# READ DA PARAMS #-----------------------------#
alpha = cf.da_params.alpha
alpha = Constant(alpha, name="alpha")
beta_J = cf.da_params.beta
beta_J = Constant(beta_J, name="beta_J")
regularize = cf.da_params.regularize
optimization = cf.da_params.optimization

#---> INITIAL TIME STEP
t = 0.0

#--->Get working tape
tape = get_working_tape()

#-----------------------------# CONFIGURE PROBLEM #-----------------------------#
problem = baseProblem(mesh_file, forwardDir, inletDir, obsDir, meshD, profile, dt, obs_dt, frames, control, inletID, solver, stationary, generate, ObsUmax, InletUmax, t, radius)

g = OrderedDict()
Hdf = HDF5File(MPI.comm_world, join('/home/biomech/sara/paratico-oasis/simple_forward_sara/tests/results/cluster/2D_Umax600/{}/{}_Inlet_Parabolic_H5_optimized_Reg{}.h5'.format(patient, patient, regularize)), 'r')
for i, t in enumerate(problem.t_range[1:]):
    t = np.round(t, 8)
    g[t] = Function(problem.V, name="g{}".format(t), annotate=False)

    Hdf.read(g[t], "g/vector_{}".format(i+1))

Hdf.close()

#-----------------------------# RUN FORWARD MODEL #-----------------------------#
if solver == "Coupled":
    ns_solver = Coupled(problem)
    solver_params = {"nu": nu, "rho": rho, "BDF2": BDF2, "stationary": stationary, "dt": dt,
                     "weakControl": weakControl, "meshD": meshD}
    if linear:
        solver_run = ns_solver.linear_run(problem.t_range, g, optimization, **solver_params)
    else:
        solver_run = ns_solver.nonlinear_run(dt, problem.V, problem.u_h, g, **solver_params)

elif solver == "IPCS":
    ns_solver = IPCS(problem)
    solver_params = {"nu": nu, "rho": rho, "BDF2": BDF2, "stationary": stationary, "dt": dt,
                     "weakControl": weakControl, "meshD": meshD}
    solver_run = ns_solver.runIPCS(problem.t_range, g, **solver_params)


if stationary:
    u_obs = problem.read_stationary_observation()
else:
    u_obs = problem.read_observation()

dx_J = ns_solver.dx
optimization = True
J = assemble(Constant(0.0)*dx_J)

i = 0
c = 0


tic = time.time()
#-----------------------------# NEW RUN WITH OPTIMIZllED RESULTS & FINAL SAVING #-----------------------------#
solver_run = ns_solver.runIPCS(problem.t_range, g, **solver_params)
with XDMFFile(MPI.comm_world, join('/home/biomech/sara/paratico-oasis/simple_forward_sara/tests/results/cluster/2D_Umax600/{}/{}_AfterOptimization_{}_{}_2It_InletUmax{}_ObsUmax{}_REG{}_postF.xdmf'.format(patient, patient, solver, minMethod, InletUmax, ObsUmax, regularize))) as file:
    file.parameters.update(
        {
            "functions_share_mesh": True,
            "rewrite_function_mesh": False,
            "flush_output": True
        }
    )
    for t, u_h, p_h, g_theta in solver_run:
        t = np.round(t, 8)
        if c % 1 == 0 and c != 0:
            file.write(u_h, t)
            file.write(p_h, t)
            #file.write(g_theta, t)

        # -----------------------------# FINAL CHECK: NEW FUNCTIONAL CALCULATION #-----------------------------#
        if not stationary:
            if i != problem.obs_t_range.shape[0]:
                if t == np.round(problem.obs_t_range[i], 8): #enable if observation interpolation is False
                    J += assemble(Constant(0.5)*inner((u_h - u_obs[t]), (u_h - u_obs[t])) * dx_J)

                    i += 1

        c += 1

toc = time.time()

if stationary:
    J += assemble(Constant(0.5)*inner((u_h - u_obs), (u_h - u_obs)) * dx_J)

print("After optimization the functional is:", J)

def t_grad(w, V):
    # Returns the Tangential part of the gradient on cell facets
    # S = w.function_space()
    S = V
    n = FacetNormal(S.mesh())
    if len(w.ufl_shape) == 0:
        return grad(w) - n * inner(n, w)
    elif len(w.ufl_shape) == 1:
        return grad(w) - outer(grad(w) * n, n)

dsu_ = ns_solver.dsu_

if regularize:
    if stationary:
        J += assemble(Constant(0.5 * alpha * dt) * g ** 2 * t_grad(g, problem.V) ** 2 * dsu_)
        print("After optimization the functional is:", J)
