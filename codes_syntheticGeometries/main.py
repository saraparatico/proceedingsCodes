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
parser = argparse.ArgumentParser()
parser.add_argument('--v', type=str, default='0', help='version') # N.B. the default version is "v0"
args = parser.parse_args()
cf = load_conf_file(f'v{int(args.v)}.json')

#-----------------------------# READ EXP PARAMS #-----------------------------#
mesh_file = cf.exp_params.meshfile
forwardDir = cf.exp_params.forwardDir
inletDir = cf.exp_params.inletDir
obsDir = cf.exp_params.obsDir
logDir = cf.exp_params.logDir
patient = cf.exp_params.patient
saveinterval = cf.exp_params.saveinterval

#-----------------------------# READ MODEL PARAMS #-----------------------------#
meshD = cf.model_params.meshD # <--> dimensionality of my problem: default 2D
profile = cf.model_params.profile
inletID = cf.model_params.inletID
rho = cf.model_params.rho
nu = cf.model_params.nu
theta = cf.model_params.theta
dt = cf.model_params.dt
obs_dt = cf.model_params.obs_dt
frames = cf.model_params.frames
generate = cf.model_params.generate
ObsUmax = cf.model_params.ObsUmax
InletUmax = cf.model_params.InletUmax
radius = cf.model_params.radius
stationary = cf.model_params.stationary
solver = cf.model_params.solver
control = cf.model_params.control
weakControl = cf.model_params.weakControl
tractionFree = cf.model_params.tractionFree
linear = cf.model_params.linear
BDF2 = cf.model_params.BDF2
minMethod = cf.model_params.minMethod
minIt = cf.model_params.minIt

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
    solver_params = {"rho": rho, "nu": nu, "meshD": meshD,
                     "nitsche_gamma": nitsche_gamma, "beta": beta, "theta": theta, "weakControl": weakControl,
                     "tractionFree": tractionFree, "backflow": backflow,
                     "simvascularstab": simvascularstab, "stationary": stationary}
    if linear:
        solver_run = ns_solver.linear_run(problem.T, dt, problem.V, problem.u_h, g, **solver_params)
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

with XDMFFile(MPI.comm_world, join(forwardDir, '{}/Umax{}/{}/BeforeOptimization_{}_{}_{}It_InletUmax{}_REG{}.xdmf'.format(meshD, ObsUmax, patient, solver, minMethod, minIt, InletUmax, regularize))) as file:
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
        if c % 10 == 0 and c != 0:
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
        with XDMFFile(MPI.comm_world, join(logDir, 'control/{}/Umax{}/{}/{:02d}/result.xdmf'.format(meshD, ObsUmax, patient, optimization_iterations))) as file:
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
                               join(logDir, 'restart/{}/Umax{}/{}/CBcontrol{:02d}.h5'.format(meshD, ObsUmax, patient, optimization_iterations)),
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

                    Hdf = HDF5File(MPI.comm_world, join(logDir, 'restart/{}/Umax{}/{}/CBcontrol{:02d}_{:.8f}.h5'.format(meshD, ObsUmax, patient, optimization_iterations, (i + 1) * problem.dt)),'w')
                    Hdf.write(current_control, "control")
                    Hdf.close()
                    c += 1

    if MPI.comm_world.rank == 0:
        logger.info('It = {}, J = {}'.format(optimization_iterations, ctrl_J.tape_value()))

tic = time.time()

#-----------------------------# MINIMIZATION #-----------------------------#
m_opt = minimize(Jhat, callback = cb, method=minMethod, tol=1.0e-12, options={'gtol': 1.0e-12, 'iprint': 101, "maxiter": minIt})
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