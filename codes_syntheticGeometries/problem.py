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

    def __init__(self, mesh_file, forwardDir, inletDir, obsDir, meshD, profile, dt, obs_dt, frames, control, inletID, solver, stationary, generate, ObsUmax, InletUmax, t, radius):
        # -----------------------------# READING INPUT PARAMETERS #-----------------------------#
        self.forwardDir = forwardDir
        self.inletDir = inletDir
        self.obsDir = obsDir
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
        self.inflow_id = inletID
        # ids bulk
        self.bulk_id = 1

        # -----------------------------# READING MESH #-----------------------------#

        # Check dimensionality: 2D VS 3D
        self.meshD = meshD

        if meshD == '2D':
           meshDir = os.path.join(mesh_file, 'mesh2D.xdmf')
           # Load mesh
           mesh = Mesh()
           with XDMFFile(meshDir) as infile:
               infile.read(mesh)
           facet_marker = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
           facet_marker.set_all(10)
           inlet_subdomain = Inlet2D()
           inlet_subdomain.mark(facet_marker, 1)
           outlet_subdomain = Outlet2D()
           outlet_subdomain.mark(facet_marker, 3)
           wall_subdomain = Walls2D()
           wall_subdomain.mark(facet_marker, 2)
           cell_ids = MeshFunction("size_t", mesh, mesh.topology().dim())

        elif meshD == '3D':
           meshDir = os.path.join(mesh_file, 'cylindermesh3D_8.xdmf')
           # Load mesh
           mesh = Mesh()
           with XDMFFile(meshDir) as infile:
               infile.read(mesh)
           mvc = MeshValueCollection("size_t", mesh, 2)
           with XDMFFile(mesh_file+"cylindermesh3D_8_exterior.xdmf") as infile:
               infile.read(mvc, "exterior")
           facet_marker = cpp.mesh.MeshFunctionSizet(mesh, mvc)
           cell_ids = MeshFunction("size_t", mesh, mesh.topology().dim())

        self.mesh = mesh
        # -----------------------------# SET BOUNDARY IDS #-----------------------------#
        self.facet_marker = facet_marker
        self.cell_ids = cell_ids

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
        inletDir = self.inletDir
        generate = self.generate
        profile = self.profile

        if generate:
            Umax = self.ObsUmax
        else:
            Umax = self.InletUmax

        g = OrderedDict()

        Hdf = HDF5File(MPI.comm_world, join(inletDir, 'Inlet_{}Umax{}_cluster_TimeVariant.h5'.format(profile, Umax)), 'r')
        for i, t in enumerate(self.t_range[1:]):
            t = np.round(t, 8)

            g[t] = Function(self.V, name="g{}".format(t), annotate=False)

            Hdf.read(g[t], "g/vector_{}".format(i + 1))

        Hdf.close()

        return g

    def construct_check_control(self):
        forwardDir = self.forwardDir

        g = Function(self.V, annotate=False)

        with HDF5File(MPI.comm_world, join(forwardDir, 'cb/restart/opt1/CBcontrol03.h5'), 'r') as file5:
            file5.read(g, "control")

        return g

    # -----------------------------# READING INITIAL VELOCITY #-----------------------------#
    def read_u0(self):
        # Create u_obs vector
        u0 = Function(self.V)

        with HDF5File(MPI.comm_world, join("./results", "intialization_u0.h5"), 'r') as file5:
            file5.write(u0, "/data")

        return u0


    # -----------------------------# READING OBSERVATION VELOCITY #-----------------------------#
    def read_observation(self):

        # Create u_obs vector
        obs = OrderedDict()
        obsDir = self.obsDir
        profile = self.profile
        ObsUmax = self.ObsUmax

        Hdf = HDF5File(MPI.comm_world, join(obsDir, 'Obs_TimeVariant{}Umax{}_generate_cluster.h5'.format(profile, ObsUmax)), 'r')
        for i, t in enumerate(self.obs_t_range[:]):
            t = np.round(t, 8)

            obs[t] = Function(self.V, name="observation{}".format(t), annotate=False)
            Hdf.read(obs[t], "u/vector_{}".format(i + 1))

        Hdf.close()

        return obs

    def construct_stationary_inlet(self):

        # Create u_obs vector
        meshD = self.meshD
        inletDir = self.inletDir
        profile = self.profile
        generate = self.generate
        solver = self.solver

        if generate:
            Umax = self.ObsUmax
        else:
            Umax = self.InletUmax

        g = Function(self.V)

        with HDF5File(MPI.comm_world, join(inletDir, '{}/Inlet_{}Umax{}_{}_mpirun.h5'.format(meshD, profile, Umax, solver)), 'r') as file5:
            file5.read(g, "g")

        return g

    def read_stationary_observation(self):

        # Create u_obs vector
        meshD = self.meshD
        obs = Function(self.V)
        obsDir = self.obsDir
        profile = self.profile
        ObsUmax = self.ObsUmax
        solver = self.solver

        with HDF5File(MPI.comm_world, join(obsDir, '{}/Obs_{}Umax{}_{}_mpirun.h5'.format(meshD, profile, ObsUmax, solver)), 'r') as file5:
            file5.read(obs, "/data")

        return obs

    # -----------------------------# BUILDING INLET VELOCITY FUNCTION #-----------------------------#

    def construct_expression_control(self):
        meshD = self.meshD
        stationary = self.stationary
        generate = self.generate
        t = self.t
        radius = self.radius
        profile = self.profile

        if generate:
            Umax = self.ObsUmax
        else:
            Umax = self.InletUmax

        if meshD == '2D':
            if stationary:
                U_inlet = StationaryBoundaryFunction2D(Umax, profile)
            else:
                U_inlet = BoundaryFunction2D(t, Umax, profile)
        elif meshD == '3D':
            if stationary:
                U_inlet = StationaryBoundaryFunction3D(t, radius, Umax, profile)
            else:
                U_inlet = BoundaryFunction3D(t, radius, Umax, profile)


        return U_inlet





