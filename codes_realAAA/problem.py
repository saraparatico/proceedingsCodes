################################################################################
################################# - problem.py - ###############################
# This file is not intended to be run directly, as it is a helper for the main
# script execution. It is responsible for setting up the mesh, function spaces,
# boundary conditions and observations required for the simulation.
################################################################################
################################################################################

#-----------------------------# IMPORT LIBRARIES #-----------------------------#
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

        mesh = Mesh()

        f.read(mesh, "mesh", False)
        # Read the mesh data from the HDF5 file into the 'mesh' object.
        # The third argument 'False' indicates that no ghost data is being read.

        self.mesh = mesh
        # Store the mesh in the instance variable 'self.mesh' for later use.

        facet_ids = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        # Create a MeshFunction to store facet IDs, which represent the boundary of the mesh.
        # The 'size_t' type is used to store the facet IDs.

        f.read(facet_ids, "/boundaries")
        # Read the facet IDs from the HDF5 file, which are stored under the "/boundaries" group.

        cell_ids = MeshFunction("size_t", mesh, mesh.topology().dim())
        # Create a MeshFunction to store cell IDs for the bulk of the mesh (non-boundary cells).

        f.read(cell_ids, "/bulk")
        # Read the cell IDs from the HDF5 file, which are stored under the "/bulk" group.

        self.facet_marker = facet_ids
        # Store the facet IDs in 'self.facet_marker' for use in later computations.

        self.cell_ids = cell_ids
        # Store the cell IDs in 'self.cell_ids' for use in later computations.

        f.close()


        # --> SURFACE UNIT FOR SURFACE INTEGRALS
        #ds = self.ds(subdomain_data=facet_marker)

        # -----------------------------# DEFINE FUNCTION SPACE #-----------------------------#

        if solver == "Coupled":

            self.V_ele = VectorElement("CG", mesh.ufl_cell(), 1)
            # Define the vector function space for the velocity using the continuous Galerkin (CG)
            # element of degree 1 (first-order) on the mesh's finite element cell.

            self.Q_ele = FiniteElement("CG", mesh.ufl_cell(), 1)
            # Define the scalar function space for the pressure using the continuous Galerkin (CG)
            # element of degree 1 (first-order) on the mesh's finite element cell.

            self.W = FunctionSpace(mesh, MixedElement([self.V_ele, self.Q_ele]))
            # Define a mixed function space 'W' combining the velocity and pressure function spaces.

            self.V = self.W.sub(0).collapse()
            # Extract the velocity subspace (first component of the mixed space) and collapse it
            # into a standard function space.

            self.Q = self.W.sub(1).collapse()
            # Extract the pressure subspace (second component of the mixed space) and collapse it
            # into a standard function space.

            self.u_h = Function(self.V, name="Velocity")
            # Create a function 'u_h' to store the velocity solution in the velocity function space.

        elif solver == "IPCS":

            self.V = VectorFunctionSpace(mesh, "CG", 2)
            # Define the vector function space for the velocity using the continuous Galerkin (CG)
            # element of degree 2 (second-order) on the mesh.

            self.Q = FunctionSpace(mesh, "CG", 1)
            # Define the scalar function space for the pressure using the continuous Galerkin (CG)
            # element of degree 1 (first-order) on the mesh.

            self.u_h = Function(self.V, name="Velocity")
            # Create a function 'u_h' to store the velocity solution in the velocity function space.



    # -----------------------------# READING INLET VELOCITIES FOR TIME-VARIANT CASES # -----------------------------#
    # This function read time-variant velocity data to put in input to my simulation.
    # This data was prviously created throught the proper inlet generator code.
    def construct_control(self):

        control_space = self.V

        g = OrderedDict()

        f = HDF5File(MPI.comm_world, join(self.inletDir, 'Inlet_H5_TimeVariant_Real.h5'), 'r')

        for i, t in enumerate(self.t_range[1:]):
            t = np.round(t, 8)

            g[t] = Function(control_space, name="control{}".format(t), annotate=False)
            f.read(g[t], "g/vector_{}".format(i + 1))

        return g


    # -----------------------------# READ INLET VELOCITIES FOR STATIONARY CASES # -----------------------------#
    # This function read stationary velocity data to put in input to my simulation.
    # This data was prviously created throught the proper inlet generator code.
    def construct_stationary_control(self):

        # !!! YOU CAN MODIFY THIS t VALUE !!!
        #t = 21
        t = 147
        # !!! REMEMBER: dt = 0.001 !!!
        # The variable 't' represents the specific time instant at which
        # we want to acquire observations.
        # In other words, the corresponding relative time is given by
        # T = 0.001 * 147 = 0.147.

        # Create u_obs vector
        g = Function(self.V, name="control", annotate=False)
        #f = HDF5File(MPI.comm_world, join(self.inletDir + '/AAA03/mesh/all_results_True.h5'), 'r')
        #f.read(g, 'u_0.147')
        f = HDF5File(MPI.comm_world, join(self.inletDir + '/AAA03/mesh_2.4/int/ps/uin_0.15.h5'), 'r')
        f.read(g, 'u')
        XDMFFile("./g_check.xdmf").write(g)

        return g


    # -----------------------------# READING OBERVATIONS FOR TIME-VARIANT CASES # -----------------------------#
    # This function read time-variant velocity data to be used as observations
    # and to be compred to my simulation result.
    # This data was prviously created throught the proper inlet generator code.
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


    # -----------------------------# READ INLET VELOCITIES FOR STATIONARY CASES # -----------------------------#
    # This function read stationary velocity data to be used as observations
    # and to be compred to my simulation result.
    # This data was prviously created throught the proper inlet generator code.
    def read_stationary_observation(self):

        # !!! YOU CAN MODIFY THIS t VALUE !!!
        #t = 1
        t = 7
        # !!! REMEMBER: obs_dt = 0.021 !!!
        # The variable 't' represents the specific time instant at which
        # we want to acquire observations.
        # In other words, the corresponding relative time is given by
        # T = 0.021 * 7 = 0.147.
        # Note: The observation should be compared with the simulation result
        # evaluated at the same time step.
        # t = 7 is selected to align with the inlet time step,
        # despite the different resolution in terms of obs_dt and dt.


        # Create u_obs vector
        obs = Function(self.V, name="observation", annotate=False)
        f = HDF5File(MPI.comm_world, join(self.obsDir + '/AAA03/mesh_2.4/uobs_velocities_noisy.h5'), 'r')
        f.read(obs, 'u')

        return obs
