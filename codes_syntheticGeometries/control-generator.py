################################################################################
############################# - control-generator.py - #########################
# This script generates the synthetic inlet velocity profiles required for
# the simulations.
# It must be executed before obs-generator.py and main.py to provide the
# necessary input data.
# Two inlet velocity profiles are required:
# - One for preliminary forward simulation, which generates the observation data.
# - One for the official forward simulation (Tape), which produces velocity
#   results to be compared with the observation data.
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

# Boolean controls which velocity profile is generated.
# If True, it creates the inlet velocity for the preliminary simulation,
# used as a synthetic observation.
# If False, it generates the inlet velocity for the main simulation,
# whose results will be compared with the observations.
generate = cf.model_params.generate

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

# --------------------------- SETUP CONTROLS: Function for Inlet g --------------------------- #
# This section defines the function used to generate the inlet velocity profile g.

# Define U_inlet as an expression describing the velocity profile at the inlet of the geometry.
uinlet = problem.construct_expression_control()

# i: Counter tracking the index for storing the inlet velocity field (g_theta) in the HDF5 file.
# Ensures that velocity fields are saved at the correct timestep in transient simulations.
i = 0

# c: Counter determining how many steps occur before saving the velocity field values.
c = 0

# tic: Stores the starting time for generating the inlet velocity.
tic = time.time()

# If `generate = True`, we are creating the inlet velocity profile (g_theta)
# for the preliminary forward simulation. The resulting velocity field will be used
# as a synthetic observation in the optimization process.
if generate:
    # Set the Umax value for g_theta to ensure the final velocity profile
    # has the desired maximum velocity corresponding to the observation.
    Umax = ObsUmax

# If `generate = False`, we are generating the inlet velocity profile (g_theta)
# for the main forward simulation (Tape). This simulation precedes the optimization
# and its velocity results will be compared against the observation velocities.
else:
    # Set the Umax value for g_theta to define the final velocity profile.
    # This will influence the results of the Tape simulation.
    # The g_theta obtained from this Umax will serve as the control variable in the optimization process.
    Umax = InletUmax

# Open an XDMF file to store the inlet velocity fields, which will be used
# as boundary conditions in the forward simulation.
# These velocity fields are obtained by projecting the expression values onto a function object.
with XDMFFile(MPI.comm_world, join(inletDir, '{}/Inlet{}Umax{}_{}_mpirun.xdmf'.format(meshD, profile, Umax, solver))) as file:
    # Configure XDMF file parameters:
    file.parameters.update({
        "functions_share_mesh": True,   # Assumes all functions share the same computational mesh.
        "rewrite_function_mesh": False, # Avoids rewriting the mesh definition for every function write.
        "flush_output": True            # Ensures data is written to the file immediately.
    })

    # Stationary option: the expression is projected onto a single function object,
    # as it remains constant over time.
    if stationary:
        g_theta = project(uinlet, problem.V)
        g_theta.rename("g", "g_theta")

        print("--> g max is", g_theta.vector().max())

        # Save the inlet velocity profile to an HDF5 file.
        with HDF5File(MPI.comm_world, join(inletDir, '{}/Inlet_{}Umax{}_{}_mpirun.h5'.format(meshD, profile, Umax, solver)), 'w') as file5:
            file5.write(g_theta, "g")

        # Save to XDMF
        file.write(g_theta)

    # Time-variant option: the expression is projected onto multiple function objects,
    # one for each time step. The expression changes dynamically at each step.
    else:
        Hdf = HDF5File(MPI.comm_world, join(inletDir, '{}/Inlet_{}Umax{}_{}_mpirun.h5'.format(meshD, profile, Umax, solver)), 'w')

        for i, t in enumerate(problem.t_range):
            t = np.round(t, 8)  # Ensure numerical precision in time values
            uinlet.t = t  # Update the expression with the current time step
            g_theta = project(uinlet, problem.V)
            g_theta.rename("g", "g_theta")

            # Save the inlet velocity field to the XDMF file every step (controlled by `c`).
            if c % 1 == 0 and c != 0:
                file.write(g_theta, t)
            c += 1

            print("--> At time t =", t, ": \ng max is", g_theta.vector().max())

            # Save to HDF5 only if the simulation is time-dependent
            if not stationary:
                Hdf.write(g_theta, "g", i)

        Hdf.close()


# Store the ending time for generating the inlet velocity.
toc = time.time()
timer = toc - tic
print("Control generator time:", timer)
