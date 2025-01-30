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

    # -----------------------------# INITIALIZATION OF INPUT PARAMETERS #-----------------------------#
    def __init__(self, mesh_file, forwardDir, inletDir, obsDir, meshD, profile, dt, obs_dt, frames, control, inletID, solver, stationary, generate, ObsUmax, InletUmax, t, radius):

        self.forwardDir = forwardDir  # Directory where the forward simulation results are saved.
        self.inletDir = inletDir      # Directory where the inlet boundary condition file is located.
        self.obsDir = obsDir          # Directory where the observation data is stored.
        self.profile = profile        # Velocity profile at the inlet (Parabolic or Plug flow).
        self.dt = dt                  # Time step for the solver loop.
        self.obs_dt = obs_dt          # Time step for observations.
        self.frames = frames          # Number of observation frames (obs_dt * frames = total simulation time T).
        self.control = control        # Defines the control parameter (in this problem: inlet).
        self.solver = solver          # Solver type (Coupled or IPCS).
        self.stationary = stationary  # Simulation type: Stationary or TimeVariant.
        self.generate = generate      # Boolean controls which velocity profile is generated.
        # If True, it creates the inlet velocity for the preliminary simulation,
        # used as a synthetic observation.
        # If False, it generates the inlet velocity for the main simulation,
        # whose results will be compared with the observations.
        self.ObsUmax = ObsUmax        # Maximum velocity for observations
        self.InletUmax = InletUmax    # Maximum velocity for the inlet
        self.t = t                    # Current time in the simulation
        self.radius = radius           # OBSOLETE: Radius of the 3D mesh (if applicable).

        # -----------------------------# DEFINING TOTAL TIME & TIME STEPS #-----------------------------#
        self.T = frames * obs_dt + DOLFIN_EPS
        self.obs_t_range = np.arange(obs_dt, self.T, obs_dt)  # Time range for observations
        self.t_range = np.arange(0, self.T, dt)  # Time range for simulation with step size dt
        # Note: The presence of `obs_dt` and `frames` reflects and anticipats the logic behind generating observations
        # in real-world scenarios, where observations are actual measurements with a specific resolution
        # that differs from the time step `dt`. Based on the number of measurements (`frames`),
        # we compute the entire observation period.

        # -----------------------------# DEFINING BOUNDARY IDS #-----------------------------#
        self.inflow_id = inletID  # Inlet boundary identifier
        self.bulk_id = 1          # Bulk domain identifier (central region)

        # -----------------------------# MESH READING #-----------------------------#
        # Check mesh dimensionality: 2D vs 3D
        self.meshD = meshD

        # If the mesh is 2D
        if meshD == '2D':
            meshDir = os.path.join(mesh_file, 'mesh2D.xdmf')  # Path to 2D mesh
            mesh = Mesh()  # Create a new mesh
            with XDMFFile(meshDir) as infile:  # Open the XDMF file for the mesh
                infile.read(mesh)  # Read the mesh from the file
            facet_marker = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)  # Create a marker for the facets
            facet_marker.set_all(10)  # Set a default value (10) for all faces
            # Assign labels to subdomains
            inlet_subdomain = Inlet2D()  # Inlet subdomain
            inlet_subdomain.mark(facet_marker, 1)  # Mark the inlet with value 1
            outlet_subdomain = Outlet2D()  # Outlet subdomain
            outlet_subdomain.mark(facet_marker, 3)  # Mark the outlet with value 3
            wall_subdomain = Walls2D()  # Wall subdomain
            wall_subdomain.mark(facet_marker, 2)  # Mark the wall with value 2
            cell_ids = MeshFunction("size_t", mesh, mesh.topology().dim())  # Mesh function for cell IDs

        # If the mesh is 3D
        elif meshD == '3D':
            meshDir = os.path.join(mesh_file, 'cylindermesh3D_8.xdmf')  # Path to 3D mesh
            mesh = Mesh()  # Create a new mesh
            with XDMFFile(meshDir) as infile:  # Open the XDMF file for the mesh
                infile.read(mesh)  # Read the mesh from the file
            mvc = MeshValueCollection("size_t", mesh, 2)  # Create a mesh value collection
            with XDMFFile(mesh_file+"cylindermesh3D_8_exterior.xdmf") as infile:  # Open the file for the exterior boundaries
                infile.read(mvc, "exterior")  # Read the exterior boundaries
            facet_marker = cpp.mesh.MeshFunctionSizet(mesh, mvc)  # Assign the facet markers
            cell_ids = MeshFunction("size_t", mesh, mesh.topology().dim())  # Mesh function for cell IDs

        self.mesh = mesh

        # -----------------------------# DEFINING BOUNDARY IDS #-----------------------------#
        self.facet_marker = facet_marker  # Assign the facet marker
        self.cell_ids = cell_ids  # Assign the cell IDs

        # --> SURFACE UNIT FOR SURFACE INTEGRALS (ds)
        # ds = self.ds(subdomain_data=facet_marker)  # Defines surface integrals using the subdomain markers


        # -----------------------------# DEFINE FUNCTION SPACE #-----------------------------#
       # If the solver is "Coupled":
       if solver == "Coupled":
            self.V_ele = VectorElement("CG", mesh.ufl_cell(), 1)  # CG vector element for velocity (order 1)
            self.Q_ele = FiniteElement("CG", mesh.ufl_cell(), 1)  # CG scalar element for pressure (order 1)
            self.W = FunctionSpace(mesh, MixedElement([self.V_ele, self.Q_ele]))  # Mixed function space for velocity and pressure
            self.V = self.W.sub(0).collapse()  # Velocity subspace (collapsed for independent access)
            self.Q = self.W.sub(1).collapse()  # Pressure subspace (collapsed for independent access)
            self.u_h = Function(self.V, name="Velocity")  # Function for velocity field

       # If the solver is "IPCS":
       elif solver == "IPCS":
            self.V = VectorFunctionSpace(mesh, "CG", 2)  # CG vector space for velocity (order 2)
            self.Q = FunctionSpace(mesh, "CG", 1)  # CG scalar space for pressure (order 1)
            self.u_h = Function(self.V, name="Velocity")  # Function for velocity field


    # -----------------------------# READING INLET VELOCITIES FOR TIME-VARIANT CASES # -----------------------------#
    # This function reads pre-generated velocity data for the inlet at different time steps
    # and stores the values in a dictionary, `g`, where each entry corresponds to a specific time step.
    # The dictionary `g` holds `Function` objects that represent the velocity field at each time step.
    # Depending on whether the `generate` flag is set to True or False, the function uses either the
    # observed maximum inlet velocity (`ObsUmax`) or the prescribed inlet velocity (`InletUmax`).
    def construct_control(self):
        inletDir = self.inletDir
        generate = self.generate
        profile = self.profile

        # If `generate = True`, we are creating the inlet velocity profile (g)
        # for the preliminary forward simulation. The resulting velocity field will be used
        # as a synthetic observation in the optimization process.
        if generate:
            Umax = self.ObsUmax  # Use observed maximum velocity if generating inlet data
        # If `generate = False`, we are generating the inlet velocity profile (g)
        # for the main forward simulation (Tape). This simulation precedes the optimization
        # and its velocity results will be compared against the observation velocities.
        else:
            Umax = self.InletUmax  # Use prescribed inlet velocity if not generating data

        g = OrderedDict()

        Hdf = HDF5File(MPI.comm_world, join(inletDir, 'Inlet_{}Umax{}_cluster_TimeVariant.h5'.format(profile, Umax)), 'r')

        # Loop through each time step in the simulation and read the corresponding velocity data
        for i, t in enumerate(self.t_range[1:]):
            t = np.round(t, 8)  # Round time step value to 8 decimal places for consistency

            # Create a new Function object for the velocity at this time step and store it in the dictionary
            g[t] = Function(self.V, name="g{}".format(t), annotate=False)

            # Read the velocity data from the file and assign it to the Function object at the corresponding time step
            Hdf.read(g[t], "g/vector_{}".format(i + 1))

        Hdf.close()

        return g



    # -----------------------------# READING INITIAL VELOCITY #-----------------------------#
    # This function reads the initial velocity profile (u0) from a pre-generated HDF5 file.
    # The HDF5 file contains the velocity data that will be applied as the initial condition in the simulation.
    def read_u0(self):
        # Create a Function object to store the initial velocity
        u0 = Function(self.V)

        # Open the HDF5 file containing the initial velocity data and read it into `u0`
        with HDF5File(MPI.comm_world, join("./results", "intialization_u0.h5"), 'r') as file5:

            file5.read(u0, "/data")

        return u0



    # -----------------------------# READING OBERVATIONS FOR TIME-VARIANT CASES # -----------------------------#
    # This function reads the observation velocity data from a pre-generated HDF5 file.
    # These observations will be used as reference data in the
    # optimization process, typically for comparing with the simulated velocity field.
    def read_observation(self):
        # Create an empty OrderedDict to store the observation velocities for each time step
        obs = OrderedDict()
        obsDir = self.obsDir
        profile = self.profile
        ObsUmax = self.ObsUmax

        # Open the HDF5 file containing the observation data for the time-variant velocity field
        Hdf = HDF5File(MPI.comm_world, join(obsDir, 'Obs_TimeVariant{}Umax{}_generate_cluster.h5'.format(profile, ObsUmax)), 'r')

        # Iterate over the range of observation times and read the corresponding velocity data
        for i, t in enumerate(self.obs_t_range[:] ):
            t = np.round(t, 8)

            # Create a Function object for each time step to store the observation velocity
            obs[t] = Function(self.V, name="observation{}".format(t), annotate=False)

            # Read the velocity data for the current time step from the HDF5 file and store it in `obs`
            Hdf.read(obs[t], "u/vector_{}".format(i + 1))

        Hdf.close()

        return obs



    # -----------------------------# READ INLET VELOCITIES FOR STATIONARY CASES # -----------------------------#
    # This function is designed to read inlet velocity data for stationary cases, where the velocity profile
    # does not vary over time. The inlet velocity is read from a pre-generated data file and stored in
    # a Function object.
    def construct_stationary_inlet(self):

        meshD = self.meshD
        inletDir = self.inletDir
        profile = self.profile
        generate = self.generate
        solver = self.solver

        # generat = Boolean controls which velocity profile is generated.
        # If True, it creates the inlet velocity for the preliminary simulation,
        # used as a synthetic observation.
        # If False, it generates the inlet velocity for the main simulation,
        # whose results will be compared with the observations.
        if generate:
            Umax = self.ObsUmax
        else:
            Umax = self.InletUmax

        # Create a Function object to store the inlet velocity
        g = Function(self.V)

        # Read the pre-generated inlet velocity data from the corresponding HDF5 file
        with HDF5File(MPI.comm_world, join(inletDir, '{}/Inlet_{}Umax{}_{}_mpirun.h5'.format(meshD, profile, Umax, solver)), 'r') as file5:

            file5.read(g, "g")

        return g



    # -----------------------------# READING OBSERVATIONS FOR STATIONARY CASES # -----------------------------#
    # This function is designed to read observation data for stationary cases, where the observations
    # do not change over time. The observation data is read from a pre-generated HDF5 file and stored
    # in a Function object.
    # The function returns the observation velocity as a Function object, which will be used for
    # comparison against the simulation results.
    def read_stationary_observation(self):

        meshD = self.meshD
        obs = Function(self.V)
        obsDir = self.obsDir
        profile = self.profile
        ObsUmax = self.ObsUmax
        solver = self.solver

        # Read the observation data from the pre-generated HDF5 file
        with HDF5File(MPI.comm_world, join(obsDir, '{}/Obs_{}Umax{}_{}_mpirun.h5'.format(meshD, profile, ObsUmax, solver)), 'r') as file5:
            file5.read(obs, "/data")

        return obs



    # -----------------------------# GENERATING INLET VELOCITIES for control-generator.py # -----------------------------#
    # This function generates inlet velocity expressions based on the configuration, which can be either stationary or time-variant and either 2D or 3D.
    # Afterwards, these expressions will b usd to be projected onto appropriat function objects.
    def construct_expression_control(self):
        meshD = self.meshD
        stationary = self.stationary
        generate = self.generate
        t = self.t
        radius = self.radius
        profile = self.profile

        # generate = Boolean controls which velocity profile is generated.
        # If True, it creates the inlet velocity for the preliminary simulation,
        # used as a synthetic observation.
        # If False, it generates the inlet velocity for the main simulation,
        # whose results will be compared with the observations.
        if generate:
            Umax = self.ObsUmax
        else:
            Umax = self.InletUmax

        if meshD == '2D':
            if stationary:
                # Generate stationary inlet velocity profile for 2D case
                U_inlet = StationaryBoundaryFunction2D(Umax, profile)
            else:
                # Generate time-variant inlet velocity profile for 2D case
                U_inlet = BoundaryFunction2D(t, Umax, profile)
        elif meshD == '3D':
            if stationary:
                # Generate stationary inlet velocity profile for 3D case
                U_inlet = StationaryBoundaryFunction3D(t, radius, Umax, profile)
            else:
                # Generate time-variant inlet velocity profile for 3D case
                U_inlet = BoundaryFunction3D(t, radius, Umax, profile)

        return U_inlet
