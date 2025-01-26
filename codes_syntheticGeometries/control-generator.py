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

#--->Get working tape
tape = get_working_tape()

#-----------------------------# CONFIGURE PROBLEM #-----------------------------#
t = 0.0
problem = baseProblem(mesh_file, forwardDir, inletDir, obsDir, meshD, profile, dt, obs_dt, frames, control, inletID, solver, stationary, generate, ObsUmax, InletUmax, t, radius)

#-----------------------------# SET UP CONTROLS: function used to create INLET g #-----------------------------#
#---> SET UP CONTROLS: function used to create g
# U_inlet is an expression
uinlet = problem.construct_expression_control()
i = 0
c = 0
tic = time.time()

if generate:
    Umax = ObsUmax
else:
    Umax = InletUmax


with XDMFFile(MPI.comm_world, join(inletDir, '{}/Inlet{}Umax{}_{}_mpirun.xdmf'.format(meshD, profile, Umax, solver))) as file:
    file.parameters.update(
        {
            "functions_share_mesh": True,
            "rewrite_function_mesh": False,
            "flush_output": True
        }
    )

    if stationary:
        g_theta = project(uinlet, problem.V)
        g_theta.rename("g", "g_theta")

        print("--> g max is ", g_theta.vector().max())
        with HDF5File(MPI.comm_world, join(inletDir, '{}/Inlet_{}Umax{}_{}_mpirun.h5'.format(meshD, profile, Umax, solver)), 'w') as file5:
            file5.write(g_theta, "g")

        file.write(g_theta)

    else:
        Hdf = HDF5File(MPI.comm_world, join(inletDir, '{}/Inlet_{}Umax{}_{}_mpirun.h5'.format(meshD, profile, Umax, solver)), 'w')

        for i, t in enumerate(problem.t_range):
            t = np.round(t, 8)
            uinlet.t = t
            g_theta = project(uinlet, problem.V)
            g_theta.rename("g", "g_theta")


            if c % 1 == 0 and c != 0:
                file.write(g_theta, t)
            c += 1

            print("--> at time t =", t, ": \ng max is ", g_theta.vector().max())

            if not stationary:

                Hdf.write(g_theta, "g", i)

        Hdf.close()

toc = time.time()
timer = toc-tic
print("Control generator time:", timer)