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
mesh_file = cf.exp_params.meshfile # Name of the mesh file.
forwardDir = cf.exp_params.forwardDir  # Directory where the forward simulation results are saved.
inletDir = cf.exp_params.inletDir  # Directory where the inlet boundary condition file is located.
obsDir = cf.exp_params.obsDir  # Directory where the observation data is stored.
meshDir = cf.exp_params.meshDir # Directory where the mesh file is located.
logDir = cf.exp_params.logDir  # Directory for the .log file created during optimization (contains callbacks and progress logs).
patient = cf.exp_params.patient  # OBSOLETE: Previously used to point to a specific mesh file.

#-----------------------------# MODEL PARAMETERS #-----------------------------#
saveinterval = cf.exp_params.saveinterval # Number of iterations after which results are saved during optimization.

# Dimensionality and physical properties.
profile = cf.model_params.profile  # OBSOLETE: Used only in synthetic data codes when BCs and observations were artificially generated.
inletID = cf.model_params.inletID  # Tag associated with the inlet boundary condition.
wallID = cf.model_params.wallID  # Tag associated with the wall boundary condition.
outletIDs = cf.model_params.outletIDs  # Tags associated with the outlet boundary conditions (two outlets in real AAA geometry).
rho = cf.model_params.rho  # Blood density (kg/m³).
nu = cf.model_params.nu  # Kinematic viscosity of the blood (m²/s).
theta = cf.model_params.theta  # Theta parameter used in time-stepping schemes.
dt = cf.model_params.dt  # Time step for the solver loop.

obs_dt = cf.model_params.obs_dt # Time step used for observations.
frames = cf.model_params.frames  # Number of observation frames (obs_dt * frames = total simulation time T).
# Note: The presence of `obs_dt` and `frames` reflects the logic behind generating observations
# As shown in "inletW.py" and "obsW.py", observations are actual measurements with a specific resolution
# that differs from the time step `dt`. Based on the number of measurements (`frames`),
# we compute the entire period.

# Boundary condition settings.
ObsUmax = cf.model_params.ObsUmax # OBSOLETE: Used only in synthetic data codes when observations were artificially generated.
InletUmax = cf.model_params.InletUmax # OBSOLETE: Used only in synthetic data codes when BCs were artificially generated.

# Solver parameters.
generate = cf.model_params.generate  # OBSOLETE: Used only in synthetic data codes when BCs and observations were artificially generated.

radius = cf.model_params.radius # OBSOLETE: Used only in synthetic 3D mesh generation
stationary = cf.model_params.stationary # OBSOLETE: Used only in synthetic data codes when BCs and observations were artificially generated.
solver = cf.model_params.solver  # Solver type (Coupled or IPCS).
control = cf.model_params.control  # Defines the control parameter (in this problem: inlet).
weakControl = cf.model_params.weakControl  # Boolean for enabling weak control (using the Neitzche method).
tractionFree = cf.model_params.tractionFree  # Boolean for enabling the traction-free condition.
linear = cf.model_params.linear  # Boolean to toggle between linear and nonlinear simulations.
BDF2 = cf.model_params.BDF2  # Boolean for using Backward Differentiation Formulas (BDF2) for time-stepping.
minMethod = cf.model_params.minMethod  # Minimization method (e.g., BFGS, L-BFGS, or TNC).

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
# - Reading and defining inlet boundary condition (control)
# - Reading and defining observations
problem = baseProblem(mesh_file, meshDir, forwardDir, inletDir, obsDir, profile, dt, obs_dt, frames, control, inletID, wallID, outletIDs, solver, stationary, generate, ObsUmax, InletUmax, t, radius)

#-----------------------------# SET UP CONTROLS: function used to read INLET g #-----------------------------#
#---> SET UP CONTROLS: function used to read g
# STATIONARY CASE:
if stationary:
    g = problem.construct_stationary_control()

# TIME-VARIANT CASE
else:
    g = problem.construct_control()

#---> We tried to use it in order to impose an initial velocity (in t0) differenc from zero
# u0 = problem.read_u0()

#-----------------------------# RUN FORWARD MODEL #-----------------------------#
if solver == "Coupled":
    ns_solver = Coupled(problem)
    solver_params = {"nu": nu, "rho": rho, "BDF2": BDF2, "stationary": stationary, "dt": dt,
                     "weakControl": weakControl}
    if linear:
        solver_run = ns_solver.linear_run(problem.t_range, g, optimization, **solver_params)
    else:
        solver_run = ns_solver.nonlinear_run(dt, problem.V, problem.u_h, g, **solver_params)

elif solver == "IPCS":
    ns_solver = IPCS(problem)
    solver_params = {"nu": nu, "rho": rho, "BDF2": BDF2, "stationary": stationary, "dt": dt,
                     "weakControl": weakControl}
    solver_run = ns_solver.runIPCS(problem.t_range, g, **solver_params)

#-----------------------------# SET UP OBSERVATIONS: function used to read OBSERVATION u_obs #-----------------------------#
# STATIONARY CASE:
if stationary:
    u_obs = problem.read_stationary_observation()
# TIME-VARIANT CASE:
else:
    u_obs = problem.read_observation()

dx_J = ns_solver.dx
J = assemble(Constant(0.0)*dx_J)
i = 0
c = 0
tic = time.time()

if MPI.comm_world.rank == 0:
    logger = get_logger(patient, log_dir=logDir, version=f'{args.v}')

#-----------------------------# RESULTS SAVING #-----------------------------#
if not stationary:
    g_theta = OrderedDict()
u_values = OrderedDict()
p_values = OrderedDict()

with XDMFFile(MPI.comm_world, join(forwardDir, '{}_BeforeOptimization_{}_RealInlet_0.5_10It_{}_obsNoisy.xdmf'.format(patient, solver, minMethod))) as file:
    file.parameters.update(
        {
            "functions_share_mesh": True,
            "rewrite_function_mesh": False,
            "flush_output": True
        }
    )

    for t, u_h, p_h, g_theta in solver_run:
        u_values[t] = Function(problem.V)
        u_values[t].assign(u_h)
        p_values[t] = Function(problem.V)
        p_values[t].assign(p_h)

        t = np.round(t, 8)
        # c di solito viene messo "c%10" ma qua ho messo "%1" perché voglio vedere ogni singolo salvataggio
        if c % 1 == 0 and c != 0:
            file.write(u_h, t)
            file.write(p_h, t)

        if not stationary:
            tic2 = time.time()
        # -----------------------------# OPTIMIZATION #-----------------------------#
            if i != problem.obs_t_range.shape[0]:
                if t == np.round(problem.obs_t_range[i], 8):  # enable if observation interpolation is False
                    J += assemble(Constant(0.5)*inner((u_h - u_obs[t]), (u_h - u_obs[t])) * dx_J)

                    i += 1
            toc2 = time.time()

        c += 1
#sys.exit()

if stationary:
    tic2 = time.time()
    J += assemble(Constant(0.5)*inner((u_h - u_obs), (u_h - u_obs)) * dx_J)
    toc2 = time.time()

timer2 = toc2-tic2
print("J at time t:", t, "is computed in:", timer2)

toc = time.time()
timer = toc-tic
if MPI.comm_world.rank == 0:
    logger.info('Forward pass ended in: {}s'.format(timer))

tic3 = time.time()
#-----------------------------# SPACE & TIME REGULARIZATION #-----------------------------#
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
dt = problem.t_range[1] - problem.t_range[0]

if regularize:
    if stationary:
        J += assemble(Constant(0.5 * alpha * dt) * g ** 2 * t_grad(g, problem.V) ** 2 * dsu_)
    else:
        h1_regularisation = sum([g0 ** 2 + t_grad(g0, problem.V) ** 2 for g0 in g.values()])
        J += assemble(Constant(0.5 * alpha * dt) * h1_regularisation * dsu_)

        time_regularisation = sum([(g1 - g0) ** 2 + t_grad(g1 - g0, problem.V) ** 2 for g1, g0 in
                                   zip(list(g.values())[:-1], list(g.values())[1:])])
        J += assemble(Constant(0.5 * beta_J / dt) * time_regularisation * dsu_)

toc3 = time.time()
timer3 = toc3-tic3
print("Regularization is computed in:", timer3)
print("Before optimization the functional is:", J)

#-----------------------------# CONVERGENCE TAYLOR TEST #-----------------------------#
if stationary:
    m = Control(g)
    u_ass = Control(u_h)
    p_ass = Control(p_h)
else:
    m = [Control(i) for i in g.values()]
    u_ass = [Control(uu) for uu in u_values.values()]
    p_ass = [Control(pp) for pp in p_values.values()]

if MPI.comm_world.rank == 0:
    logger.info('It = 0, J = {}'.format(J))

Jhat = ReducedFunctional(J, m)
Jhat.optimize_tape()
#taylor_test(Jhat, g, h)
#dJ = compute_gradient(J, m)
ctrl_J = Control(J)


optimization_iterations = 0
def cb(*args, **kwargs):
    global optimization_iterations
    optimization_iterations += 1

    #if optimization_iterations % 1 == 0:
    if optimization_iterations % 1 == 0:
        with XDMFFile(MPI.comm_world, join(logDir, 'control/{}/{:02d}/result.xdmf'.format(patient,optimization_iterations))) as file:
            file.parameters.update(
                {
                    "functions_share_mesh": True,
                    "rewrite_function_mesh": False,
                    "flush_output": True
                }
            )
            c = 1

            if stationary:
                current_control = m.tape_value()
                current_control.rename("control", "")
                #current_velocity = u_ass.tape_value()
                #current_velocity.rename("Velocity", "")
                # current_pressure = p_ass.tape_value()
                # current_pressure.rename("Pressure", "")
                if c % saveinterval == 0:
                    file.write(current_control)
                    #file.write(current_velocity)
                    # file.write(current_pressure)

                Hdf = HDF5File(MPI.comm_world,
                               join(logDir, 'restart/{}/CBcontrol{:02d}.h5'.format(patient, optimization_iterations)),
                               'w')
                Hdf.write(current_control, "control")
                Hdf.close()

                c += 1
            else:
                for i in range(0, len(m)):
                    current_control = m[i].tape_value()
                    current_control.rename("control", "")
                    current_velocity = u_ass[i].tape_value()
                    current_velocity.rename("Velocity", "")
                    current_pressure = p_ass[i].tape_value()
                    current_pressure.rename("Pressure", "")
                    if c % saveinterval == 0:
                        file.write(current_control, (i + 1) * problem.dt)
                        file.write(current_velocity, (i + 1) * problem.dt)
                        file.write(current_pressure, (i + 1) * problem.dt)

                    Hdf = HDF5File(MPI.comm_world, join(logDir, 'restart/{}/CBcontrol{:02d}_{:.8f}.h5'.format(patient, optimization_iterations, (i + 1) * problem.dt)),'w')
                    Hdf.write(current_control, "control")
                    Hdf.close()
                    c += 1

    if MPI.comm_world.rank == 0:
        logger.info('It = {}, J = {}'.format(optimization_iterations, ctrl_J.tape_value()))

tic = time.time()

#-----------------------------# MINIMIZATION #-----------------------------#
m_opt = minimize(Jhat, callback = cb, method=minMethod, tol=1.0e-12, options={'gtol': 1.0e-12, 'iprint': 101, "maxiter": 10})
with stop_annotating():
    # Split optimal solution in its components
    if not stationary:
        g_opt = OrderedDict(zip(np.round(problem.t_range[1:], 8), m_opt))
    else:
        g_opt = Function(problem.V)
        g_opt.assign(m_opt)



toc = time.time()

if MPI.comm_world.rank == 0:
    logger.info('Assimilation pass ended in: {}s'.format(toc-tic))

optimization = True
J = assemble(Constant(0.0)*dx_J)

i = 0
c = 0

tic = time.time()
#-----------------------------# NEW RUN WITH OPTIMIZllED RESULTS & FINAL SAVING #-----------------------------#
solver_run = ns_solver.runIPCS(problem.t_range, g_opt, **solver_params)
with XDMFFile(MPI.comm_world, join(forwardDir, '{}_AfterOptimization_{}_RealInlet_0.5_10It_{}_obsNoisy.xdmf'.format(patient, solver, minMethod))) as file:
    file.parameters.update(
        {
            "functions_share_mesh": True,
            "rewrite_function_mesh": False,
            "flush_output": True
        }
    )
    for t, u_h, p_h, g_theta in solver_run:
        t = np.round(t, 8)
        if c % 10 == 0 and c != 0:
            file.write(u_h, t)
            file.write(p_h, t)
            file.write(g_theta, t)

        # -----------------------------# FINAL CHECK: NEW FUNCTIONAL CALCULATION #-----------------------------#
        if not stationary:
            if i != problem.obs_t_range.shape[0]:
                if t == np.round(problem.obs_t_range[i], 8): #enable if observation interpolation is False
                    J += assemble(Constant(0.5)*inner((u_h - u_obs[t]), (u_h - u_obs[t])) * dx_J)

                    i += 1

        c += 1

toc = time.time()

if MPI.comm_world.rank == 0:
    logger.info('Post assimilation pass ended in: {}s'.format(toc-tic))

if stationary:
    J += assemble(Constant(0.5)*inner((u_h - u_obs), (u_h - u_obs)) * dx_J)

print("After optimization the functional is:", J)

if regularize:
    if stationary:
        J += assemble(Constant(0.5 * alpha * dt) * g_opt ** 2 * t_grad(g_opt, problem.V) ** 2 * dsu_)
        print("After optimization the functional is:", J)
