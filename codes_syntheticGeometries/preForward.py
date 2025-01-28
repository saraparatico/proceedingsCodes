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

#-----------------------------# SET UP CONTROLS: function used to create INLET g #-----------------------------#
#---> SET UP CONTROLS: function used to create g
# 1st VERSION: U_inlet is an expression
if stationary:
    ########## COMMENTO PER FARE TEST
    g = problem.construct_stationary_inlet()
    ######### VERSIONE TEST
    #uinlet = problem.construct_expression_control()
    #g = project(uinlet, problem.V)
# 2nd VERSION: g is a dictionary of Function(V)
else:
    g = problem.construct_control()

#---> We tried to use it in order to impose an initial velocity (in t0) differenc from zero
# u0 = problem.read_u0()

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

with XDMFFile(MPI.comm_world, join(forwardDir, '{}_BeforeOptimization_{}_{}_8It_InletUmax{}_ObsUmax{}_REG{}_preForward.xdmf'.format(patient, solver, minMethod, InletUmax, ObsUmax, regularize))) as file:
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

        time_regularisation = sum([(g1 - g0) ** 2 + t_grad(g1 - g0, problem.V) ** 2 for g1, g0 in zip(list(g.values())[:-1], list(g.values())[1:])])
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
                current_velocity = u_ass.tape_value()
                current_velocity.rename("Velocity", "")
                current_pressure = p_ass.tape_value()
                current_pressure.rename("Pressure", "")
                if c % saveinterval == 0:
                    file.write(current_control)
                    file.write(current_velocity)
                    file.write(current_pressure)

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
if minMethod == "L-BFGS-B":
    m_opt = minimize(Jhat, callback = cb, method=minMethod, tol=1.0e-12, options={'gtol': 1.0e-12, 'iprint': 101, "maxiter": 8})
elif minMethod == "TNC":
    m_opt = minimize(Jhat, callback = cb, method=minMethod, options={'gtol': 1.0e-12, "maxiter": 4})
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

g_save = OrderedDict()
c = 0

with XDMFFile(MPI.comm_world, join(inletDir, '{}_Inlet_{}_optimized_Reg{}.xdmf'.format(patient, profile, regularize))) as file:
    file.parameters.update(
        {
            "functions_share_mesh": True,
            "rewrite_function_mesh": False,
            "flush_output": True
        }
    )

    Hdf = HDF5File(MPI.comm_world, join(inletDir, '{}_Inlet_{}_H5_optimized_Reg{}.h5'.format(patient, profile, regularize)), 'w')
    print("H5 file initialized.")
    g_s = Function(problem.V)
    g_s.rename("g", "g_theta")
    Hdf.write(g_s, 'g', 0)

    for i, t in enumerate(problem.t_range[1:]):
        print("Inside for loop.")
        t = np.round(t, 8)

        g_s.assign(g_opt[t])
        g_s.rename("g", "g_theta")

        file.write(g_s, t)

        Hdf.write(g_s, 'g', i+1)

    Hdf.close()

print("End PreForward")
