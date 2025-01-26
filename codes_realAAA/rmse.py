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
meshDir = cf.exp_params.meshDir
logDir = cf.exp_params.logDir
patient = cf.exp_params.patient
saveinterval = cf.exp_params.saveinterval

#-----------------------------# READ MODEL PARAMS #-----------------------------#
profile = cf.model_params.profile
inletID = cf.model_params.inletID
wallID = cf.model_params.wallID
outletIDs = cf.model_params.outletIDs
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
problem = baseProblem(mesh_file, meshDir, forwardDir, inletDir, obsDir, profile, dt, obs_dt, frames, control, inletID, wallID, outletIDs, solver, stationary, generate, ObsUmax, InletUmax, t, radius)

u_h = OrderedDict()

f = XDMFFile(MPI.comm_world, join('/home/biomech/sara/paratico-oasis/simple_forward_sara/tesi-Sara-Paratico/results/AAA03_AfterOptimization_IPCS_RealInlet_0.5_10It_L-BFGS-B_obs4.xdmf'))
for i, t in enumerate(problem.t_range[1:]):
    t = np.round(t, 8)
    u_h[t] = Function(problem.V, name="control{}".format(t), annotate=False)
    f.read(u_h, 'velocity', t)