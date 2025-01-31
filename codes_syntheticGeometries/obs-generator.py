################################################################################
############################## - obs-generator.py - ############################
# This script generates synthetic observational velocity fields, serving as a
# reference for the optimization process.
# It must be executed before main.py to produce the required input data for
# optimization.
# The resulting velocity field from the preliminary forward, with specific inlet
# conditions imposed, is actually used as reference data to guide
# the optimization process relarted the forward simulation Tape.
################################################################################
################################################################################


#-----------------------------# IMPORT LIBRARIES #-----------------------------#
#!/usr/bin/env python
from __future__ import generators
import sys
import time
from mshr import *
from collections import OrderedDict
from dolfin import *
from dolfin_adjoint import *
import argparse
import numpy as np
from os.path import join

# Import from codes contained in the folder of the project
from problem import baseProblem
from solvers import Coupled, IPCS
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
# Note: The presence of `obs_dt` and `frames` reflects and anticipats the logic behind generating observations
# in real-world scenarios, where observations are actual measurements with a specific resolution
# that differs from the time step `dt`. Based on the number of measurements (`frames`),
# we compute the entire period.

# Boundary condition settings.
ObsUmax = cf.model_params.ObsUmax  # Maximum velocity value for the observation profile.
InletUmax = cf.model_params.InletUmax  # Maximum velocity value for the inlet boundary condition.

# Solver parameters.
generate = cf.model_params.generate # Boolean controls which velocity profile is generated.
# If True, it creates the inlet velocity for the preliminary simulation,
# used as a synthetic observation.
# If False, it generates the inlet velocity for the main simulation,
# whose results will be compared with the observations.
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

#-----------------------------# STABILIZATION PARAMETERS #-----------------------------#
beta = cf.stab_params.beta  # OBSOLETE: Previously used for P1-P1 elements with Brezzi-Pitkaranta stabilization.
convective = cf.stab_params.convective  # OBSOLETE: Boolean previously used for stabilization in P1-P1 elements with SUPG/PSPG (from Fumagalli).
backflow = cf.stab_params.backflow  # OBSOLETE: In recent versions, backflow stabilization is always included by default.
simvascularstab = cf.stab_params.simvascularstab  # Boolean: Enables backflow stabilization in nonlinear solver.
nitsche_gamma = cf.stab_params.nitsche_gamma  # OBSOLETE: Previously used for Nitsche stabilization terms.

#-----------------------------# DATA ASSIMILATION PARAMETERS #-----------------------------#.
alpha = cf.da_params.alpha  # Coefficient for spatial regularization term.
alpha = Constant(alpha, name="alpha")  # Define alpha as a constant for use in the solver.
beta_J = cf.da_params.beta  # Coefficient for temporal regularization term.
beta_J = Constant(beta_J, name="beta_J")  # Define beta_J as a constant for use in the solver.
regularize = cf.da_params.regularize # Boolean: Decides whether to include regularization in the optimization process.
optimization = cf.da_params.optimization # OBSOLETE: Previously used before the existence of control_generator.py and obs_generator.py; It was required to generate results without performing optimization.

#---> Get the active working tape for recording operations
tape = get_working_tape()

#-----------------------------# CONFIGURE PROBLEM #-----------------------------#
t = 0.0
problem = baseProblem(mesh_file, forwardDir, inletDir, obsDir, meshD, profile, dt, obs_dt, frames, control, inletID, solver, stationary, generate, ObsUmax, InletUmax, t, radius)

#-----------------------------# SET UP CONTROLS: function used to create INLET g #-----------------------------#
#---> SET UP CONTROLS: function used to create g
if not generate:
    sys.exit()

# 1st VERSION: Stationary case.
# In this case, 'g' is defined from an expression defining the velocity profile in space,
# which remains constant over all time instances.
if stationary:
    # The inlet is read from pre-generated data.
    # The values are loaded into a function object.
    g = problem.construct_stationary_inlet()
# 2nd VERSION: Time-variant case.
# Here, 'g' is a dictionary of Function(V), with one entry for each time step.
# The inlet values are read from pre-generated data and stored in a dictionary of function objects.
else:
    g = problem.construct_control()

# OBSOLETE: Previously, the inlet was created by defining
# an expression and projecting it onto a function object.
#uinlet = problem.construct_expression_control()
#g = project(uinlet, problem.V)

#-----------------------------# RUN FORWARD MODEL #-----------------------------#
#### OPTION 1: Coupled solver
if solver == "Coupled":
    ns_solver = Coupled(problem)

    # Define solver parameters for the Coupled solver:
    # - "rho": blood density
    # - "nu": kinematic viscosity
    # - "meshD": problem dimensionality (2D/3D)
    # - "nitsche_gamma": OBSOLETE parameter for Nitsche stabilization
    # - "beta": OBSOLETE parameter for stabilization (Brezzi-Pitkaranta)
    # - "theta": parameter for time discretization (e.g., backward Euler)
    # - "weakControl": Boolean for weak control using Nitsche's method
    # - "tractionFree": Boolean to activate traction-free boundary conditions
    # - "backflow": Boolean for backflow stabilization (enabled by default)
    # - "simvascularstab": Boolean for specific non-linear stabilization (not IPCS)
    # - "stationary": Boolean to define if the simulation is time-independent
    solver_params = {"rho": rho, "nu": nu, "meshD": meshD,
                     "nitsche_gamma": nitsche_gamma, "beta": beta, "theta": theta, "weakControl": weakControl,
                     "tractionFree": tractionFree, "backflow": backflow,
                     "simvascularstab": simvascularstab, "stationary": stationary}
    # Run the solver based on the linearity of the problem:
    if linear:
        # Solver for linear systems
        solver_run = ns_solver.linear_run(problem.T, dt, problem.V, problem.u_h, g, **solver_params)
    else:
        # Solver for nonlinear systems
        solver_run = ns_solver.nonlinear_run(dt, problem.V, problem.u_h, g, **solver_params)

#### OPTION 2: IPCS solver
elif solver == "IPCS":
    ns_solver = IPCS(problem)

    # Define solver parameters for the IPCS solver:
    # - "nu": kinematic viscosity
    # - "rho": blood density
    # - "BDF2": Boolean to enable second-order Backward Differentiation Formula
    # - "stationary": Boolean to define if the simulation is time-independent
    # - "dt": time step
    # - "weakControl": Boolean for weak control using Nitsche's method
    # - "meshD": problem dimensionality (2D/3D)
    solver_params = {"nu": nu, "rho": rho, "BDF2": BDF2, "stationary": stationary, "dt": dt,
                     "weakControl": weakControl, "meshD": meshD}
    solver_run = ns_solver.runIPCS(problem.t_range, g, **solver_params)

# i: counter used to track the index for storing the velocity field (u_h) in the HDF5 file.
# It ensures that the velocity field is saved at the correct timestep in non-stationary simulations.
i = 0
# c: counter used to determine how many steps occur before saving result values.
c = 0
# tic: Stores the starting time for computing the total duration of the forward simulation to generate observation.
tic = time.time()

# Open an XDMF file to save simulation velocities used to generate observations.
with XDMFFile(MPI.comm_world, join(obsDir, '{}/Obs{}Umax{}_{}_mpirun.xdmf'.format(meshD, profile, ObsUmax, solver))) as file:
    # Configure XDMF file parameters:
    # "functions_share_mesh": Assumes all functions share the same computational mesh.
    # "rewrite_function_mesh": Avoids rewriting the mesh definition every time a function is written.
    # "flush_output": Ensures data is written to the file immediately for every call to
    file.parameters.update(
        {

            "functions_share_mesh": True,
            "rewrite_function_mesh": False,
            "flush_output": True
        }
    )
    Hdf = HDF5File(MPI.comm_world, join(obsDir, '{}/Obs_{}Umax{}_{}_mpirun.h5'.format(meshD, profile, ObsUmax, solver)), 'w')

    # Iterate through the results of the solver run, which provides:
    # `t` - the current time step,
    # `u_h` - velocity solution at time `t`,
    # --> u_h will be saved to generate the oservation !!!
    # `p_h` - pressure solution at time `t`,
    # `g_theta` - inlet velocity profile at time `t`.
    for t, u_h, p_h, g_theta in solver_run:

        t = np.round(t, 8)

        # The result is savd and then will be used as observation, to be
        # compared to the Tape's result.
        # Since observations is only described by velocities, I only save them.
        # Save velocity fields to the XDMF file every step (controlled by `c`).
        if c % 1 == 0 and c != 0:
            file.write(u_h, t)
        c += 1
        print("--> at time t =", t, ": \n u max is:", u_h.vector().max())


        if not stationary:
            Hdf.write(u_h, "u", i)

        i+=1

    Hdf.close()

if stationary:
    with HDF5File(MPI.comm_world, join(obsDir, '{}/Obs_{}Umax{}_{}_mpirun.h5'.format(meshD, profile, ObsUmax, solver)), 'w') as file5:
        file5.write(u_h, "/data")

# toc: Stores the ending time for computing the total duration of the forward simulatio to generate observation.
toc = time.time()
timer = toc-tic
print("Obs generator time:", timer)
