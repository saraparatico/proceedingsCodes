################################################################################
############################## - project_inlet.py - ############################
# It's a code needed to prepare real inlet data to its use in IPCS scheme.
# This file is necessary because "inletW.py" generates velocity input data
# using first-order Lagrange elements, whereas the IPCS scheme
# requires velocities in a second-order vector function space.
# It must be run after the "init-data" codes but before running the simulation.
################################################################################
################################################################################
# !!! IMPORTANT: MODIFY LINE RELATED TO DIRS TO SET YOUR OWN ROOT DIRECTORY !!!
#-----------------------------# IMPORT LIBRARIES #-----------------------------#
from dolfin import *
from dolfin_adjoint import *

import numpy as np
from os.path import join

# ---> Define patient ID (this should be a user-defined parameter)
patient = 'AAA03'  #!TODO: make it a user parameter

# ---> Define whether to use a pulsatile (ps) profile
psProfile = True

# ---> Set flag to enable data interpolation over time
InterpolateDataOverTime = True

# ---> Define whether the simulation is stationary or time-variant
stationary = True

# ---> Define the directory where the mesh files are stored
meshDir = join('./init-data_2.4/mesh/mesh_2.4')
# ---> Construct the full path to the mesh file
mesh_file = join(meshDir, f'{patient}.h5')
f = HDF5File(MPI.comm_world, mesh_file, 'r')
mesh = Mesh()
# ---> Read the mesh data from the file into the mesh object
f.read(mesh, "mesh", False)

# ---> Create a function to store boundary facet IDs
facet_ids = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
# ---> Read boundary facet information from the file
f.read(facet_ids, "/boundaries")
f.close()

# ---> Define the ID of the inflow boundary
inflow_id = 2

# ---> Initialize a counter variable for time steps
c = 0


frames = 40  # Frames in the dataset
obs_dt = 0.021 # Observation time step (time between frames)
dt = 0.001 # Simulation time step
# Note: The presence of `obs_dt` and `frames` reflects the logic behind generating observations
# As shown in "inletW.py" and "obsW.py", observations are actual measurements with a specific resolution
# that differs from the time step `dt`. Based on the number of measurements (`frames`),
# we compute the entire period.
T = frames * obs_dt + DOLFIN_EPS
obs_t_range = np.arange(obs_dt, T, obs_dt)
t_range = np.arange(0, T, dt)

# ---> Define a vector function space on the mesh using first-order Lagrange elements
V = VectorFunctionSpace(mesh, 'CG', 1)

# ---> Define the directory where inlet velocity data is stored
inletDir = "./init-data_2.4/inlet"
f = HDF5File(MPI.comm_world, join(inletDir + '/AAA03/mesh_2.4/int/ps/uin.h5'), 'r')

# ---> Define a higher-order vector function space (for projection)
Q = VectorFunctionSpace(mesh, 'CG', 2)

# ---> Check if the simulation is stationary
if stationary:
    # ---> Define the specific time step to extract.
    # In stationary casem I want to focus on only one instant.
    # !!! YOU CAN MODIFY THIS t VALUE !!!
    t = 63
    # !!! REMEMBER: dt = 0.001 !!!
    # The variable 't' represents the specific time instant at which
    # we want to acquire observations.
    # In other words, the corresponding relative time is given by
    # T = 0.001 * 63 = 0.063.

    # ---> Create an empty function in the velocity space
    g = Function(V)
    # ---> Read the velocity field corresponding to time step t
    f.read(g, 'u/vector_{}'.format(t))

    # KEY PASSAGE FROM FIRST ORDER FUNCTION SPACE TO SCOND ORDER FUNCTION SPACE
    # ---> Project the velocity field onto the higher-order function space Q
    j = project(g, Q)

    # ---> Create another function for storing the final velocity field
    r = Function(Q)

    # ---> Assign the projected velocity field to r
    # r.vector()[:] = 0.15 * j.vector()[:]  # Uncomment if scaling is needed
    r.vector()[:] = j.vector()[:]
    r.rename("g", "g_theta")
    XDMFFile("./g_63.xdmf").write(r)
    HDF5File(MPI.comm_world, join(inletDir + '/AAA03/mesh_2.4/int/ps/uin_prova.h5'), 'w').write(r, "u")

else:
    # ---> Time-dependent case: Prepare files for storing velocity fields over time
    with XDMFFile(MPI.comm_world, join(inletDir, 'Inlet_TimeVariant_Real.xdmf')) as file:

        # ---> Set file parameters for writing multiple time steps
        file.parameters.update(
            {
                "functions_share_mesh": True,  # All functions use the same mesh
                "rewrite_function_mesh": False,  # Do not rewrite mesh at each time step
                "flush_output": True  # Ensure data is written immediately
            }
        )

        # ---> Open an HDF5 file to store the time-variant inlet velocity field
        Hdf = HDF5File(MPI.comm_world, join(inletDir, 'Inlet_H5_TimeVariant_Real.h5'), 'w')

        # ---> Loop over ALL TIME STEPS (!!!every 0.001!!!) steps except the first one
        for i, t in enumerate(t_range[1:]):
            # ---> Round the time value to avoid numerical errors
            t = np.round(t, 8)

            # ---> Create an empty function for the velocity field
            g_tmp = Function(V)

            # ---> Read the velocity field corresponding to the current time step
            f.read(g_tmp, 'u/vector_{}'.format(i + 1))

            # KEY PASSAGE FROM FIRST ORDER FUNCTION SPACE TO SCOND ORDER FUNCTION SPACE
            # ---> Project the velocity field onto the higher-order function space
            j = project(g_tmp, Q)

            # ---> Rename the function for output files
            j.rename("g", "g_theta")

            # ---> Write the velocity field to the XDMF file at certain time steps
            if c % 1 == 0 and c != 0:  # Modify the condition to control writing frequency
                file.write(j, t)

            c += 1

            Hdf.write(j, 'g', i)

        Hdf.close()
