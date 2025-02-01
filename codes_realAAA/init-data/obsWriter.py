################################################################################
################################ - obsW.py - ###################################
# This script reads the velocity files to be used as inlet observation and
# compared to the forward simulation result.
# It should be run after the execution of convertMesh.py,
# but before running the simulation.
# !!! IMPORTANT: MODIFY LINE 36 TO SET YOUR OWN ROOT DIRECTORY !!!
################################################################################
################################################################################

from dolfin import *
from dolfin_adjoint import *
from scipy import interpolate
from os.path import join
import vtk
import pandas as pd
from tqdm import tqdm
import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
from glob import glob
import pyvista as pv

# ---> Define patient ID (TODO: this should be a user parameter)
patient = 'AAA03'

# ---> Define time parameters (TODO: obs_dt  and frames should be user parameters)
obs_dt = 0.021  # Time step between frames
frames = 40  # Number of frames in the dataset
# Note: The presence of `obs_dt` and `frames` reflects the logic behind generating observations
# As shown in "inletW.py" and "obsW.py", observations are actual measurements with a specific resolution
# that differs from the time step `dt`. Based on the number of measurements (`frames`),
# we compute the entire period.
T = frames * obs_dt + DOLFIN_EPS  # Total simulation time
obs_t_range = np.arange(0, T, obs_dt)  # Time steps for velocity field (t=0 -> zero velocity)
t_range = np.arange(0, T, obs_dt / 21)  # Finer time range for interpolation

#---> Flag for interpolation over time
InterpolateDataOverTime = False  # If True, will interpolate velocity data over time

# ---> Define root directory (!!! Modify this path according to your setup !!!)
root = '/home/biomech/sara/paratico-oasis/simple_forward_sara/tesi-Sara-Paratico'  # Set root directory for project data

#---> Define directories for input and output data
processed4DFlowDataDir = join(root, f'4DFlow/velocities/{patient}/cut')  # Directory with pre-processed 4D flow data
meshDir = join(root, f'init-data/mesh')  # Directory with mesh data
outputDir = join(root, 'init-data/obs/')  # Directory where output data will be stored

#---> Read mesh data using FEniCS
mesh_file = join(meshDir, f'{patient}.h5')
f = HDF5File(MPI.comm_world, mesh_file, 'r')
mesh = Mesh()
f.read(mesh, "mesh", False)
facet_ids = MeshFunction("size_t", mesh, mesh.topology().dim()-1)  # Define facets (boundary identifiers)
f.read(facet_ids, "/boundaries")  # Read boundary data
f.close()

#---> Read processed 4D flow data (velocity field)
allData = []
processed4DFlowFiles = sorted(glob(join(processed4DFlowDataDir, '*.vtu')))  # Get list of all processed 4D flow files

for f in tqdm(range(len(processed4DFlowFiles)), desc='Reading processed vtk frames'):

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(processed4DFlowFiles[f])
    reader.Update()
    grid = reader.GetOutput()

    # ---> Clean up data from 4D flow .vtu files (remove unwanted data)
    velpv = pv.wrap(grid)  # Convert the grid to a pyvista object
    velpvvel = velpv['velocity']  # Extract the velocity data
    velpv.clear_data()  # Clear any unnecessary data
    velpv['velocity'] = velpvvel  # Re-add the cleaned velocity data

    allData.append(velpv)  # Append the cleaned data to the allData list

#---> Read the registered mesh (target mesh for probing velocity data)
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(join(meshDir, f'{patient}.vtu'))
reader.Update()
geo = reader.GetOutput()

#---> Define field names for storing velocity data
fieldnames = ['x', 'y', 'z', 'u', 'v', 'w']
df_arr = np.empty((geo.GetNumberOfPoints(), len(fieldnames)))  # Create an empty array for storing the data

nVertices = mesh.num_vertices()  # Get the number of vertices in the mesh

#---> Initialize numpy arrays to store velocity field data over time
vel_np = np.zeros((nVertices, 3, obs_t_range.shape[0]))  # Velocity field for all time frames
u_i = np.zeros((nVertices, obs_t_range.shape[0]))  # U-component of velocity over time
v_i = np.zeros((nVertices, obs_t_range.shape[0]))  # V-component of velocity over time
w_i = np.zeros((nVertices, obs_t_range.shape[0]))  # W-component of velocity over time

#---> Loop through each frame of the processed 4D flow data
for f in tqdm(range(len(allData)), desc='Writing point data for frame'):
    vel_grid = allData[f]  # Get the current velocity grid

    # Probe the target mesh with the velocity field
    probe = vtk.vtkProbeFilter()
    probe.SetInputData(geo)  # Set the target mesh for probing
    probe.SetSourceData(vel_grid)  # Set the velocity data as the source
    probe.Update()
    geoWithVars = probe.GetOutput()  # Get the result with velocity data

    # Extract points and velocity data from the probe result
    vtk_pts = geoWithVars.GetPoints()
    ptsArr = vtk_to_numpy(vtk_pts.GetData())  # Convert points to a numpy array
    velArr = vtk_to_numpy(geoWithVars.GetPointData().GetArray('velocity'))  # Convert velocity to numpy

    # Store the velocity and point data in a dataframe
    df_arr[:, 0] = ptsArr[:, 0]
    df_arr[:, 1] = ptsArr[:, 1]
    df_arr[:, 2] = ptsArr[:, 2]
    df_arr[:, 3] = velArr[:, 0]
    df_arr[:, 4] = velArr[:, 1]
    df_arr[:, 5] = velArr[:, 2]

    # Store the velocity field over time
    vel_np[:, 0, f + 1] = velArr[:, 0]
    vel_np[:, 1, f + 1] = velArr[:, 1]
    vel_np[:, 2, f + 1] = velArr[:, 2]

    # Store individual velocity components for interpolation purposes
    u_i[:, f + 1] = velArr[:, 0]
    v_i[:, f + 1] = velArr[:, 1]
    w_i[:, f + 1] = velArr[:, 2]

    # Save the velocity and point data as CSV files for later use
    df = pd.DataFrame(df_arr, columns=fieldnames)
    df.to_csv(join(outputDir, '{}/mesh/probedData/point_data_{:02d}.csv'.format(patient, f)), index=False)

    # Save the velocity data as VTK files for visualization purposes
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(join(outputDir, '{}/mesh/probedData/probedData_{:02d}.vtp'.format(patient, f)))
    writer.SetInputData(geoWithVars)
    writer.Update()

#---> Define function space for velocity field (used in finite element analysis)
V = VectorFunctionSpace(mesh, "CG", 1)  # Create a vector function space for the velocity field (P1 elements)

#---> Create output file path for storing all observations
output_file_path = join(outputDir, '{}/mesh/all_OBS.h5'.format(patient))
Hdf = HDF5File(MPI.comm_world, output_file_path, 'w')  # Open the file for writing HDF5 data


if not InterpolateDataOverTime:
    for i, t in enumerate(obs_t_range):

        u_obs = Function(V, name='obs')  # Create a function object to store the velocity data at each observation point
        u_obs.vector()[vertex_to_dof_map(V)] = vel_np[:, :, i].flatten()  # Map the velocity data to the function's vector

        # xf_u_obs = XDMFFile(join(root, 'u_obs_{:02d}.xdmf'.format(f)))
        # xf_u_obs.write(project(u_obs, V))
        # ---> Save: NO! I do not want a t number of savings
        # Hdf = HDF5File(MPI.comm_world, join(outputDir, '{}/mesh/uin_{:02d}.h5'.format(patient, t)), 'w')
        # Hdf.write(u_obs, "u")
        # Hdf.close()

        dataset_name = 'u_{:02}'.format(t)  # Format the name of the dataset for the current time step
        Hdf.write(u_obs, dataset_name)

    Hdf.close()

if InterpolateDataOverTime:
    # ---> Interpolate in time velocity data for smoother transition between time steps
    vel_npinterp = np.zeros((nVertices, 3, t_range.shape[0]))  # Array to store interpolated velocity data

    # Loop over each vertex to perform interpolation on the velocity components
    for idx in tqdm(range(nVertices)):
        # Interpolate for the 'u' component (velocity in x-direction)
        f = interpolate.interp1d(list(obs_t_range), list(u_i[idx, :]), kind='quadratic')  # Quadratic interpolation
        vel_npinterp[idx, 0, :] = f(t_range)  # Store the interpolated 'u' values

        # Interpolate for the 'v' component (velocity in y-direction)
        f = interpolate.interp1d(list(obs_t_range), list(v_i[idx, :]), kind='quadratic')  # Quadratic interpolation
        vel_npinterp[idx, 1, :] = f(t_range)  # Store the interpolated 'v' values

        # Interpolate for the 'w' component (velocity in z-direction)
        f = interpolate.interp1d(list(obs_t_range), list(w_i[idx, :]), kind='quadratic')  # Quadratic interpolation
        vel_npinterp[idx, 2, :] = f(t_range)  # Store the interpolated 'w' values

    # ---> Initialize observation vector for time interpolation
    t = 0
    u_obs = Function(V, name='obs')  # Observation vector to store interpolated data

    # ---> Check
    # xf_u_obs = XDMFFile(join(outputDir, '{}/mesh_{}/check/uobs_{:.8f}.xdmf'.format(patient, dx, np.round(t, 8))))
    # xf_u_obs.write(u_obs)

    # Save the initial observation data (t = 0)
    Hdf = HDF5File(MPI.comm_world, join(outputDir, '{}/mesh/uobs_{:.8f}.h5'.format(patient, np.round(t, 8))), 'w')
    Hdf.write(u_obs, "u")
    Hdf.close()

    # Loop over the time steps (excluding the first step) to save interpolated data
    for i, t in enumerate(t_range[1:]):
        u_obs.vector()[vertex_to_dof_map(V)] = vel_npinterp[:, :, i].flatten()  # Initialize the observation vector with interpolated data

        # ---> Check
        # if i % 42 == 0:
        #     xf_u_obs = XDMFFile(join(outputDir, '{}/mesh_{}/check/uobs_{:.8f}.xdmf'.format(patient, dx, np.round(t, 8))))
        #     xf_u_obs.write(u_obs)

        Hdf = HDF5File(MPI.comm_world, join(outputDir, '{}/mesh/uobs_{:.8f}.h5'.format(patient, np.round(t, 8))), 'w')
        Hdf.write(u_obs, "u")
        Hdf.close()
