################################################################################
################################### - main.py - ################################
# This is the core file meant to be run.
# It contains the structure of the optimization and simulation process,
# involving the minimization of a functional using a control-based approach.
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
# we compute the entire observation period.

# Boundary condition settings.
ObsUmax = cf.model_params.ObsUmax  # Maximum velocity value for the observation profile.
InletUmax = cf.model_params.InletUmax  # Maximum velocity value for the inlet boundary condition.

# Solver parameters.
generate = cf.model_params.generate  # Boolean controls which velocity profile is generated.
# If True, it creates the inlet velocity for the preliminary simulation,
# used as a synthetic observation.
# If False, it generates the inlet velocity for the main simulation,
# whose results will be compared with the observations.
radius = cf.model_params.radius  # OBSOLETE: Radius of the 3D mesh (if applicable).
stationary = cf.model_params.stationary  # Simulation type: Stationary or TimeVariant.
solver = cf.model_params.solver  # Solver type (Coupled or IPCS).
control = cf.model_params.control  # Defines the control parameter (in this problem: inlet).
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

#---> INITIAL TIME STEP
t = 0.0

#---> Get the active working tape for recording operations
tape = get_working_tape()

#-----------------------------# CONFIGURE PROBLEM #-----------------------------#
# The "problem" class handles:
# - Reading and defining the mesh
# - Generating function spaces
# - Defining the inlet boundary condition (control)
# - Handling observations
problem = baseProblem(
    mesh_file, forwardDir, inletDir, obsDir, meshD, profile, dt, obs_dt,
    frames, control, inletID, solver, stationary, generate, ObsUmax,
    InletUmax, t, radius
)

#-----------------------------# SET UP CONTROLS: function used to create INLET 'g' #-----------------------------#
#---> Setting up controls: defines the function 'g' for the inlet boundary condition
# 1st VERSION: Stationary case.
# In this case, 'g' is defined from an expression defining the velocity profile in space,
# which remains constant over all time instances.
if stationary:
    # The inlet is read from pre-generated data using ad hoc code.
    # The values are loaded into a function object.
    g = problem.construct_stationary_inlet()

    # OBSOLETE: Previously, the inlet was created within this code by defining
    # an expression and projecting it onto afunction object.
    # uinlet = problem.construct_expression_control()
    # g = project(uinlet, problem.V)

# 2nd VERSION: Time-variant case.
# Here, 'g' is a dictionary of Function(V), with one entry for each time step.
# The inlet values are read from pre-generated data and stored in a dictionary of function objects.
else:
    g = problem.construct_control()


#---> Option to impose an initial velocity (at t0) different from zero
# The variable 'u0' can be used to read and impose a predefined initial velocity profile.
# NOTE: Currently commented out as it is not required in the current configuration.
# u0 = problem.read_u0()

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
    solver_params = {
        "rho": rho, "nu": nu, "meshD": meshD, "nitsche_gamma": nitsche_gamma,
        "beta": beta, "theta": theta, "weakControl": weakControl,
        "tractionFree": tractionFree, "backflow": backflow,
        "simvascularstab": simvascularstab, "stationary": stationary
    }

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
    solver_params = {
        "nu": nu, "rho": rho, "BDF2": BDF2, "stationary": stationary,
        "dt": dt, "weakControl": weakControl, "meshD": meshD
    }

    solver_run = ns_solver.runIPCS(problem.t_range, g, **solver_params)


#-----------------------------# SET UP OBSERVATION: function used to create oservation 'u_obs' #-----------------------------#
# 1st VERSION: Stationary case.
# In this case, 'u_obs' is defined from an expression defining the velocity profile in space,
# which remains constant over all time instances.
if stationary:
    # The observation is read from pre-generated data using ad hoc code.
    # The values are loaded into a function object.
    u_obs = problem.read_stationary_observation()
# 2nd VERSION: Time-variant case.
# Here, 'u_obs' is a dictionary of Function(V), with one entry for each time step.
# The values are read from pre-generated data and stored in a dictionary of function objects.
else:
    u_obs = problem.read_observation()

#-----------------------------# INITIALIZATION OF FUNCTIONAL #-----------------------------#
dx_J = ns_solver.dx
# Functional initially set to zero.
J = assemble(Constant(0.0) * dx_J)
# i: counter used to determine how many time steps occur before triggering the optimization process.
i = 0
# c: counter used to determine how many steps occur before saving result values.
c = 0
# tic: Stores the starting time for computing the total duration of the forward simulation.
tic = time.time()

# Initialization of the logger for callback information.
# The logger is only created by the root process (rank 0) to avoid redundancy in parallel simulations.
if MPI.comm_world.rank == 0:
    logger = get_logger(patient, log_dir=logDir, version=f'{args.v}')

#-----------------------------# RESULTS SAVING #-----------------------------#
# g_theta: stores the inlet velocity profiles used as input for the simulation.
# In the time-variant case, where the inlet velocity changes over time,
# g_theta is a dictionary containing a function object for each time instant.
if not stationary:
    g_theta = OrderedDict()

# u_values and p_values: dictionaries used to save forward simulation results.
# These contain velocity ('u_values') and pressure ('p_values') fields for each time step.
# Since simulations run at each time step (dt) over the total simulation time (T),
# the results are saved as function objects for every time instance.
u_values = OrderedDict()
p_values = OrderedDict()

# Open an XDMF file to save simulation results (velocities and pressures) during execution.
with XDMFFile(MPI.comm_world, join(forwardDir,
    '{}/Umax{}/{}/BeforeOptimization_{}_{}_{}It_InletUmax{}_REG{}.xdmf'.format(
        meshD, ObsUmax, patient, solver, minMethod, minIt, InletUmax, regularize))) as file:

    # Configure XDMF file parameters:
    # "functions_share_mesh": Assumes all functions share the same computational mesh.
    # "rewrite_function_mesh": Avoids rewriting the mesh definition every time a function is written.
    # "flush_output": Ensures data is written to the file immediately for every call to `file.write`.
    file.parameters.update(
        {
            "functions_share_mesh": True,
            "rewrite_function_mesh": False,
            "flush_output": True
        }
    )

    # Iterate through the results of the solver run, which provides:
    # `t` - the current time step,
    # `u_h` - velocity solution at time `t`,
    # `p_h` - pressure solution at time `t`,
    # `g_theta` - inlet velocity profile at time `t`.
    for t, u_h, p_h, g_theta in solver_run:
        # Store velocity and pressure solutions for the current time step in dictionaries.
        u_values[t] = Function(problem.V)
        u_values[t].assign(u_h)
        p_values[t] = Function(problem.V)
        p_values[t].assign(p_h)

        # Round the time step to 8 decimal places for consistency in comparisons and output.
        t = np.round(t, 8)

        # Save velocity and pressure fields to the XDMF file every 10 steps (controlled by `c`).
        if c % 10 == 0 and c != 0:
            file.write(u_h, t)  # Write velocity field at time `t`.
            file.write(p_h, t)  # Write pressure field at time `t`.

        # Time-variant option:
        if not stationary:
            tic2 = time.time()  # Start timer for tracking optimization in time-variant case.

            # Perform optimization only if within the range of observation times.
            if i != problem.obs_t_range.shape[0]:
                # Check if the current time step matches the observation time (rounded for accuracy).
                # Note: Observation interpolation must be disabled for this condition to apply.
                if t == np.round(problem.obs_t_range[i], 8):
                    # Compute and accumulate the functional `J`, which represents the error between
                    # simulated velocity `u_h` and observed velocity `u_obs` at the current time step.
                    J += assemble(Constant(0.5) * inner((u_h - u_obs[t]), (u_h - u_obs[t])) * dx_J)
                    i += 1  # Move to the next observation time step.

            toc2 = time.time()  # Stop timer for tracking optimization in time-variant case.

        c += 1  # Increment the counter used for triggering the optimization process.

# Stationary option:
if stationary:
    tic2 = time.time()   # Start timer for tracking optimization in stationary case.

    # Compute the functional `J` for stationary cases by comparing the final velocity `u_h`
    # with the observed velocity `u_obs` across the entire domain.
    J += assemble(Constant(0.5) * inner((u_h - u_obs), (u_h - u_obs)) * dx_J)
    # N.B. Differently from in time-variant case, here J is computed only once, at the end of the forward process.

    toc2 = time.time()  # Stop timer for tracking optimization in stationary case.

# Compute the elapsed time for functional computation.
timer2 = toc2 - tic2
print("J at time t:", t, "is computed in:", timer2)

# Capture the total simulation time by recording the final timestamp.
toc = time.time()
timer = toc - tic

# If the current process is the master process (rank 0), log the total duration
# of the forward simulation using the logger.
if MPI.comm_world.rank == 0:
    logger.info('Forward pass ended in: {}s'.format(timer))

# Start timer for regularization computation
tic3 = time.time()

#-----------------------------# SPACE & TIME REGULARIZATION #-----------------------------#

def t_grad(w, V):
    """
    Computes the tangential component of the gradient on cell facets.

    Parameters:
    - w: Function whose gradient needs to be computed.
    - V: Function space of the problem.

    Returns:
    - The tangential part of the gradient, ensuring it is projected correctly
      onto the facet normals.
    """
    # Get the function space (mesh information)
    S = V
    n = FacetNormal(S.mesh())  # Extract the facet normal

    # If w is a scalar function, compute the tangential gradient accordingly
    if len(w.ufl_shape) == 0:
        return grad(w) - n * inner(n, grad(w))

    # If w is a vector field, compute the tangential gradient projection
    elif len(w.ufl_shape) == 1:
        return grad(w) - outer(grad(w) * n, n)

dsu_ = ns_solver.dsu_

# Compute the time step size
dt = problem.t_range[1] - problem.t_range[0]

# If True: compute and add space and time regularization to functional
if regularize:
    # Stationary option: only spatial regularization.
    if stationary:
        # Penalizes high gradients in the control function g to ensure smoothness
        J += assemble(Constant(0.5 * alpha * dt) * g ** 2 * t_grad(g, problem.V) ** 2 * dsu_)

    # Time-variant option: both spatial and time regularization.
    else:
        # H1 (spatial) regularization for time-variant control:
        # Penalizes large values and large gradients of g across the domain
        h1_regularisation = sum([g0 ** 2 + t_grad(g0, problem.V) ** 2 for g0 in g.values()])
        J += assemble(Constant(0.5 * alpha * dt) * h1_regularisation * dsu_)

        # Time regularization:
        # Penalizes large changes in control function g over consecutive time steps
        time_regularisation = sum([(g1 - g0) ** 2 + t_grad(g1 - g0, problem.V) ** 2 for g1, g0 in zip(list(g.values())[:-1], list(g.values())[1:])])
        J += assemble(Constant(0.5 * beta_J / dt) * time_regularisation * dsu_)

# Stop timer for regularization computation
toc3 = time.time()
timer3 = toc3 - tic3

# Print the execution time for regularization computation
print("Regularization is computed in:", timer3)

# Print the value of the functional J before optimization begins
print("Before optimization the functional is:", J)

#-----------------------------# CONVERGENCE TAYLOR TEST #-----------------------------#

# Control variables definition!
## Stationary option:
if stationary:
    m = Control(g)   # Control variable is the velocity in inlet
    u_ass = Control(u_h)
    p_ass = Control(p_h)
# Time-variant option:
else:
    m = [Control(i) for i in g.values()]  # Control variables are velocities in inlet at each time step
    u_ass = [Control(uu) for uu in u_values.values()]
    p_ass = [Control(pp) for pp in p_values.values()]

# Log initial functional value at iteration 0
if MPI.comm_world.rank == 0:
    logger.info('It = 0, J = {}'.format(J))

# Reduced functional for optimization
Jhat = ReducedFunctional(J, m)

# Optimize the tape (used in adjoint-based optimization to store computational graphs)
Jhat.optimize_tape()

# Uncomment the following lines if Taylor test or gradient computation is needed:
# taylor_test(Jhat, g, h)  # Perform a Taylor test for checking adjoint consistency
# dJ = compute_gradient(J, m)  # Compute the gradient of J with respect to control m

ctrl_J = Control(J)

# Initialization of optimization iteration counter
optimization_iterations = 0

def cb(*args, **kwargs):
    """
    Callback function executed at each optimization iteration.
    It logs the optimization progress, saves the control variables, and writes results to file.
    """
    global optimization_iterations
    optimization_iterations += 1

    # Save control values at every iteration
    if optimization_iterations % 1 == 0:  # Currently saving at every iteration
        with XDMFFile(MPI.comm_world, join(logDir, 'control/{}/Umax{}/{}/{:02d}/result.xdmf'
                                           .format(meshD, ObsUmax, patient, optimization_iterations))) as file:
            file.parameters.update(
                {
                    "functions_share_mesh": True,  # Ensures that all functions share the same mesh
                    "rewrite_function_mesh": False,  # Prevents rewriting the mesh at every save
                    "flush_output": True  # Ensures immediate writing of data to disk
                }
            )
            c = 1  # Counter for saving control variables

            # Stationary case: save control at the end of procedure
            if stationary:
                # Retrieve the current control value from the tape
                current_control = m.tape_value()
                current_control.rename("control", "")

                # Uncomment if you want to retrieve and save velocity and pressure too
                # current_velocity = u_ass.tape_value()
                # current_velocity.rename("Velocity", "")
                # current_pressure = p_ass.tape_value()
                # current_pressure.rename("Pressure", "")

                # Save control function at specified intervals
                if c % saveinterval == 0:
                    file.write(current_control)
                     # file.write(current_velocity)
                     # file.write(current_pressure)

                # Save the control in an HDF5 restart file
                Hdf = HDF5File(MPI.comm_world,
                               join(logDir, 'restart/{}/Umax{}/{}/CBcontrol{:02d}.h5'
                                    .format(meshD, ObsUmax, patient, optimization_iterations)), 'w')
                Hdf.write(current_control, "control")
                Hdf.close()

                c += 1  # Increment the counter

            # Time-dependent case: save control, velocity, and pressure for each time step
            else:

                for i in range(0, len(m)):
                    current_control = m[i].tape_value()
                    current_control.rename("control", "")
                    current_velocity = u_ass[i].tape_value()
                    current_velocity.rename("Velocity", "")
                    current_pressure = p_ass[i].tape_value()
                    current_pressure.rename("Pressure", "")

                    # Save control, velocity, and pressure at specified intervals
                    if c % saveinterval == 0:
                        file.write(current_control, (i + 1) * problem.dt)
                        file.write(current_velocity, (i + 1) * problem.dt)
                        file.write(current_pressure, (i + 1) * problem.dt)

                    # Save control in an HDF5 restart file for each time step
                    Hdf = HDF5File(MPI.comm_world,
                                   join(logDir, 'restart/{}/Umax{}/{}/CBcontrol{:02d}_{:.8f}.h5'
                                        .format(meshD, ObsUmax, patient, optimization_iterations, (i + 1) * problem.dt)),
                                   'w')
                    Hdf.write(current_control, "control")
                    Hdf.close()

                    c += 1  # Increment the counter

    # Log the current functional value J after each optimization iteration
    if MPI.comm_world.rank == 0:
        logger.info('It = {}, J = {}'.format(optimization_iterations, ctrl_J.tape_value()))

# Start timer for minimization process
tic = time.time()

#-----------------------------# MINIMIZATION #-----------------------------#
# Optimization process to minimize the reduced functional Jhat.
# The callback function 'cb' is used to monitor the progress of the minimization,
# and the optimization is configured using 'minMethod', with a specified tolerance and gradient tolerance.
m_opt = minimize(Jhat, callback=cb, method=minMethod, tol=1.0e-12, options={'gtol': 1.0e-12, 'iprint': 101, "maxiter": minIt})


with stop_annotating():
    # Time-variant option:
    if not stationary:
        # Solution is stored as a time-dependent sequence in 'g_opt'.
        g_opt = OrderedDict(zip(np.round(problem.t_range[1:], 8), m_opt))
    # Stationary option:
    else:
        g_opt = Function(problem.V)
        # Solution is assigned directly to 'g_opt' as a function object.
        g_opt.assign(m_opt)

# Record the time taken for the minimization process and log it.
toc = time.time()
if MPI.comm_world.rank == 0:
    logger.info('Assimilation pass ended in: {}s'.format(toc - tic))

#-----------------------------# PREPARATION FOR NEW SIMULATION RUN #-----------------------------#
optimization = True # OBSOLETE
J = assemble(Constant(0.0) * dx_J)  # Initialize the functional to be recalculated after optimization.

i = 0  # Counter for matching observed time steps (for non-stationary problems).
c = 0  # Counter for output saving frequency.

# Start timer for the new simulation run with the optimized control.
tic = time.time()

#-----------------------------# NEW RUN WITH OPTIMIZED RESULTS & FINAL SAVING #-----------------------------#
# Based on the solver type, execute a new simulation with optimized inlet g_opt!
if solver == "Coupled":
    solver_run = ns_solver.linear_run(problem.T, dt, problem.V, problem.u_h, g_opt, **solver_params)
elif solver == "IPCS":
    solver_run = ns_solver.runIPCS(problem.t_range, g_opt, **solver_params)


with XDMFFile(MPI.comm_world, join(forwardDir, '{}/Umax{}/{}/AfterOptimization_{}_{}_{}It_InletUmax{}_REG{}.xdmf'.format(meshD, ObsUmax, patient, solver, minMethod, minIt, InletUmax, regularize))) as file:
    file.parameters.update(
        {
            "functions_share_mesh": True,
            "rewrite_function_mesh": False,
            "flush_output": True
        }
    )
    for t, u_h, p_h, g_theta in solver_run:
        t = np.round(t, 8)  # Ensure time is rounded for precision consistency.

        # Save results every 10 steps
        if c % 10 == 0 and c != 0:
            file.write(u_h, t)
            file.write(p_h, t)
            file.write(g_theta, t)

        #-----------------------------# FINAL FUNCTIONAL CALCULATION #-----------------------------#
        # Time-variant option: compute the functional J at observation time steps.
        if not stationary:
            if i != problem.obs_t_range.shape[0]:
                if t == np.round(problem.obs_t_range[i], 8):
                    J += assemble(Constant(0.5) * inner((u_h - u_obs[t]), (u_h - u_obs[t])) * dx_J)
                    i += 1  # Move to the next observation step.

        c += 1  # Increment the saving counter.

# Stop timer for the new simulation run with the optimized control.
toc = time.time()
if MPI.comm_world.rank == 0:
    logger.info('Post assimilation pass ended in: {}s'.format(toc - tic))

# Stationary option: calculate the functional J only once over the entire domain.
if stationary:
    J += assemble(Constant(0.5) * inner((u_h - u_obs), (u_h - u_obs)) * dx_J)

# Final functional value after optimization for validation purposes.
print("After optimization the functional is:", J)
