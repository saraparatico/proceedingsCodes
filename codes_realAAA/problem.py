import time
from collections import OrderedDict
import os
from os.path import join
import numpy as np
from mshr import *
from dolfin import *
from dolfin_adjoint import *
from utils import Inlet2D, Outlet2D, Walls2D, BoundaryFunction2D, StationaryBoundaryFunction2D, StationaryBoundaryFunction3D, BoundaryFunction3D

class baseProblem(object):

    def __init__(self, mesh_file, meshDir, forwardDir, inletDir, obsDir, profile, dt, obs_dt, frames, control, inletID, wallID, outletIDS, solver, stationary, generate, ObsUmax, InletUmax, t, radius):
        # -----------------------------# READING INPUT PARAMETERS #-----------------------------#
        self.forwardDir = forwardDir
        self.inletDir = inletDir
        self.obsDir = obsDir
        self.meshDir = meshDir
        self.profile = profile
        self.dt = dt
        self.obs_dt = obs_dt
        self.frames = frames
        self.control = control
        self.solver = solver
        self.stationary = stationary
        self.generate = generate
        self.ObsUmax = ObsUmax
        self.InletUmax = InletUmax
        self.t = t
        self.radius = radius

        # -----------------------------# DEFINING TOTAL TIME & TIME STEPS #-----------------------------#
        self.T = frames * obs_dt + DOLFIN_EPS  # frames = 20, obs_dt = 0.042 --> 0.8437500000000003
        self.obs_t_range = np.arange(obs_dt, self.T, obs_dt)
        self.t_range = np.arange(0, self.T, dt)

        # ids boundaries
        self.wallID = wallID
        self.inletID = inletID
        self.outletIDS = outletIDS

        # ids bulk
        self.bulk_id = 1

        # -----------------------------# READING MESH #-----------------------------#

        f = HDF5File(MPI.comm_world, meshDir, 'r')
        # Load mesh
        mesh = Mesh()
        f.read(mesh, "mesh", False)
        self.mesh = mesh
        facet_ids = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        f.read(facet_ids, "/boundaries")
        cell_ids = MeshFunction("size_t", mesh, mesh.topology().dim())
        f.read(cell_ids, "/bulk")
        self.facet_marker = facet_ids
        self.cell_ids = cell_ids
        f.close()

        # --> SURFACE UNIT FOR SURFACE INTEGRALS
        #ds = self.ds(subdomain_data=facet_marker)

        # -----------------------------# DEFINE FUNCTION SPACE #-----------------------------#
        if solver == "Coupled":
            self.V_ele = VectorElement("CG", mesh.ufl_cell(), 1)
            self.Q_ele = FiniteElement("CG", mesh.ufl_cell(), 1)
            self.W = FunctionSpace(mesh, MixedElement([self.V_ele, self.Q_ele]))
            self.V = self.W.sub(0).collapse()
            self.Q = self.W.sub(1).collapse()
            self.u_h = Function(self.V, name="Velocity")

        elif solver == "IPCS":
            self.V = VectorFunctionSpace(mesh, "CG", 2)
            self.Q = FunctionSpace(mesh, "CG", 1)
            self.u_h = Function(self.V, name="Velocity")

    def construct_control(self):

        control_space = self.V

        g = OrderedDict()

        f = HDF5File(MPI.comm_world, join(self.inletDir, 'Inlet_H5_TimeVariant_Real.h5'), 'r')

        for i, t in enumerate(self.t_range[1:]):
            t = np.round(t, 8)

            g[t] = Function(control_space, name="control{}".format(t), annotate=False)
            f.read(g[t], "g/vector_{}".format(i + 1))

        return g

    def construct_check_control(self):
        forwardDir = self.forwardDir

        g = Function(self.V, annotate=False)

        with HDF5File(MPI.comm_world, join(forwardDir, 'CBcontrol10.h5'), 'r') as file5:
            file5.read(g, "control")

        return g

    def construct_stationary_control(self):

        #t = 21
        t = 147
        # Create u_obs vector
        g = Function(self.V, name="control", annotate=False)
        #f = HDF5File(MPI.comm_world, join(self.inletDir + '/AAA03/mesh/all_results_True.h5'), 'r')
        #f.read(g, 'u_0.147')
        f = HDF5File(MPI.comm_world, join(self.inletDir + '/AAA03/mesh_2.4/int/ps/uin_0.15.h5'), 'r')
        f.read(g, 'u')
        XDMFFile("./g_check.xdmf").write(g)

        return g


    # -----------------------------# READING OBSERVATION VELOCITY #-----------------------------#
    def read_observation(self):

        # Create u_obs vector
        obs = OrderedDict()
        obsDir = self.obsDir
        profile = self.profile
        ObsUmax = self.ObsUmax

        f = HDF5File(MPI.comm_world, join(self.inletDir + 'Inlet_H5_TimeVariant_Real.h5'), 'r')

        #f = HDF5File(MPI.comm_world, join(self.dataDir + '/obs', self.patient, 'uobs.h5'), 'r')
        for i, t in enumerate(self.obs_t_range):
            t = np.round(t, 8)

            obs[t] = Function(self.V, name="observation{}".format(t), annotate=False)
            f.read(obs[t], "g/vector_{}".format(i + 1))

        return obs

    def read_stationary_observation(self):

        #t = 1
        t = 7
        # Create u_obs vector
        obs = Function(self.V, name="observation", annotate=False)
        f = HDF5File(MPI.comm_world, join(self.obsDir + '/AAA03/mesh_2.4/uobs_velocities_noisy.h5'), 'r')
        f.read(obs, 'u')

        return obs
