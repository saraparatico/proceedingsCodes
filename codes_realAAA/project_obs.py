################################################################################
############################### - project_obs.py - #############################
# It's a code needed to prepare real observation to its use in IPCS scheme.
# This file is necessary because "obsW.py" generates observations
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
from collections import OrderedDict

# ---> Define patient ID (this should be a user-defined parameter)
patient = 'AAA03'  #!TODO: make it a user parameter

# ---> Define whether to use a pulsatile (ps) profile
psProfile = True


frames = 3  # Frames in the dataset
obs_dt = 0.021 # Observation time step (time between frames)
dt = 0.001 # Simulation time step
# Note: The presence of `obs_dt` and `frames` reflects the logic behind generating observations
# As shown in "inletW.py" and "obsW.py", observations are actual measurements with a specific resolution
# that differs from the time step `dt`. Based on the number of measurements (`frames`),
# we compute the entire period.
T = frames * obs_dt + DOLFIN_EPS
obs_t_range = np.arange(obs_dt, T, obs_dt)
t_range = np.arange(0, T, dt)

# ---> Set flag to enable data interpolation over time
InterpolateDataOverTime = True

# ---> Root directory for potential data storage (commented out, as it's not currently used)
# root = '../femda/data'
# root = '/home/biomech/sara/paratico-oasis/simple_forward_sara/tesi-Sara-Paratico'

# ---> Define directories
# processed4DFlowDataDir = join('./4DFlow/velocities/{patient}/cut')  # Commented out, but might be used for processed data

# ---> Directory containing mesh files
meshDir = join('./init-data_2.4/mesh/mesh_2.4')
# ---> Output directory for simulation results
outputDir = join('./init-data_prova/inlet/')
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

# ---> Define a vector function space on the mesh using first-order Lagrange elements
V = VectorFunctionSpace(mesh, 'CG', 1)
# ---> Define a higher-order vector function space (for projection)
Q = VectorFunctionSpace(mesh, 'CG', 2)
# ---> Define the directory containing observational data
obsDir = "./init-data_2.4/obs"
f = HDF5File(MPI.comm_world, join(obsDir + '/AAA03/mesh_2.4/uobs.h5'), 'r')

# ---> Initialize a counter variable for time steps
c = 0

# ---> Define whether the simulation is stationary or time-variant
stationary = False

# ---> Check if the simulation is stationary
if stationary:
    # ---> Define the specific instant to extract.
    # In stationary casem I want to focus on only one instant.
    # !!! YOU CAN MODIFY THIS t VALUE !!!
    t = 3
    # !!! REMEMBER: obs_dt = 0.021 !!!
    # The variable 't' represents the specific time instant at which
    # we want to acquire observations.
    # In other words, the corresponding relative time is given by
    # T = 0.021 * 3 = 0.063.
    # Note: The observation should be compared with the simulation result
    # evaluated at the same time step.
    # t = 3 is selected to align with the inlet time step,
    # despite the different resolution in terms of obs_dt and dt.


    # ---> Create an empty function in the velocity space
    g = Function(V)

    # ---> Read the velocity field corresponding to time step t
    f.read(g, 'u/vector_{}'.format(t))

    # KEY PASSAGE FROM FIRST ORDER FUNCTION SPACE TO SCOND ORDER FUNCTION SPACE
    # ---> Project the velocity field onto the higher-order function space Q
    j = project(g, Q)
    j.rename("obs", "u_obs")

    XDMFFile("./obs_3.xdmf").write(g)
    HDF5File(MPI.comm_world, join(obsDir + '/AAA03/mesh_2.4/uobs_velocity3.h5'), 'w').write(j, "u")

else:
    # ---> Time-dependent case: Prepare files for storing velocity fields over time
    with XDMFFile(MPI.comm_world, join(obsDir, 'Obs_TimeVariant_RealData_provaReading.xdmf')) as file:

        # ---> Set file parameters for writing multiple time steps
        file.parameters.update(
            {
                "functions_share_mesh": True,  # All functions use the same mesh
                "rewrite_function_mesh": False,  # Do not rewrite mesh at each time step
                "flush_output": True  # Ensure data is written immediately
            }
        )

        # ---> Open an HDF5 file to store the time-variant observation velocity field
        Hdf = HDF5File(MPI.comm_world, join(obsDir, 'Obs_H5_TimeVariant_RealData_provaReading.h5'), 'w')

        # ---> Loop over all OBSERVATION time steps !! (every 0.021) !!
        for i, t in enumerate(obs_t_range[:]):
            # ---> Round the time value to avoid numerical errors
            t = np.round(t, 8)

            # ---> Create an empty function for the observation velocity field
            obs_tmp = Function(V)

            # ---> Read the velocity field corresponding to the current time step
            f.read(obs_tmp, 'u/vector_{}'.format(i + 1))

            # KEY PASSAGE FROM FIRST ORDER FUNCTION SPACE TO SCOND ORDER FUNCTION SPACE
            # ---> Project the velocity field onto the higher-order function space
            j = project(obs_tmp, Q)


            j.rename("obs", "u_obs")
            file.write(j, t)
            # ---> Print the maximum value of the velocity field for debugging
            print("at", t, ":", j.vector().max())
            Hdf.write(j, 'obs', i)

        Hdf.close()

print("the end")
