################################################################################
############################## - components.py - ##############################
# This script prepares noisy observations for use in the IPCS scheme.
# It can be considered as an aternative to obsW.py for noisy data.
# It processes real data modified with added noise, following the approach from:
#
# Saitta, S., Carioni, M., Mukherjee, S., Schönlieb, C. B., & Redaelli, A. (2024).
# "Implicit neural representations for unsupervised super-resolution
# and denoising of 4D flow MRI."
# Computer Methods and Programs in Biomedicine, 246, 108057.
# https://doi.org/10.1016/j.cmpb.2024.108057
#
# This script must be executed after the "init-data" preprocessing codes
# but before running the simulation.
# !!! IMPORTANT: MODIFY LINE RELATED TO DIRS TO SET YOUR OWN ROOT DIRECTORY !!!
################################################################################
################################################################################


import numpy as np
import pyvista as pv
from dolfin import *
import vtk
from os.path import join
import sys
from dolfin import *
from dolfin_adjoint import *
from scipy import interpolate
from os.path import join
import vtk
from tqdm import tqdm
import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
from glob import glob
import pyvista as pv
import math
import pyvista as pv
import numpy as np
import vtk
from dolfin import *
from mpi4py import MPI
from tqdm import tqdm
from vtk.util.numpy_support import vtk_to_numpy

### MESH LOADING
# Load VTU mesh file using PyVista
mesh_vtu = "./init-data_2.4/mesh/mesh_2.4/AAA03.vtu"
target_mesh = pv.read(mesh_vtu)

# Load HDF5 mesh file using FEniCS
mesh_h5 = "./init-data_2.4/mesh/mesh_2.4/AAA03.h5"
f = HDF5File(MPI.comm_world, mesh_h5, 'r')
mesh = Mesh()
f.read(mesh, "mesh", False)  # Read the mesh structure
facet_ids = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
f.read(facet_ids, "/boundaries")  # Read boundary facet IDs
f.close()

#### NOISY DATA LOADING
# Load velocity field data from a .vti file
vel_file = "/home/biomech/sara/paratico-oasis/simple_forward_sara/tesi-Sara-Paratico/init-data_2.4/aaa03_ivar0.0005_center5_snr10_uniformGridNoisy.vti"
grid = pv.read(vel_file)

### VEL COMPONENTS EXTRACTION
# Extract velocity components from the PyVista grid
gu = grid["u"]
u_array = np.array(gu)
v = grid["v"]
v_array = np.array(v)
w = grid["w"]
w_array = np.array(w)

### COMBINATION OF COMPONENTS 1)
# Stack velocity components into a single array
velocity = np.column_stack((u, v, w))
print(velocity.shape)

### COMBINATION OF COMPONENTS 2)
# Compute the magnitude of the velocity field
def compute_resultant_velocity(vx, vy, vz):
    return np.sqrt(vx**2 + vy**2 + vz**2)

resulting_v = compute_resultant_velocity(u_array, v_array, w_array)
print("Final v vector:", resulting_v)
print(len(resulting_v))

# Add 'velocity' vector to the PyVista grid
grid["velocity"] = velocity

# Interpolate the 'velocity' field onto the target mesh
interpolated = target_mesh.interpolate(grid, radius=2.0)
# Note: This interpolates onto a 22000-element mesh but not onto all 60000 DoFs

# Save the interpolated mesh with velocity values into a new .vtu file
output_file = "./init-data_2.4/obs/AAA03/obs_noisy.vtu"
interpolated.save(output_file)

###### FROM .VTU TO .H5
# Define time parameters
frames = 40
T = 0.840
obs_dt = T/frames  # TODO: Make this a user parameter
T = frames * obs_dt + DOLFIN_EPS
obs_t_range = np.arange(0, T, obs_dt)  # Time range for observations
dt = 0.001  # TODO: Make this a user parameter
t_range = np.arange(0, T, dt)

# Read interpolated .vtu file using VTK
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(output_file)
reader.Update()
grid = reader.GetOutput()

# Remove unnecessary data from 4D flow VTU file
velpv = pv.wrap(grid)
velpvvel = velpv['velocity']
velpv.clear_data()
velpv['velocity'] = velpvvel

# Read the original mesh .vtu file
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(mesh_vtu)
reader.Update()
geo = reader.GetOutput()

# Define field names for storing velocity data
fieldnames = ['x', 'y', 'z', 'u', 'v', 'w']
df_arr = np.empty((geo.GetNumberOfPoints(), len(fieldnames)))

# Get the number of vertices in the FEniCS mesh
nVertices = mesh.num_entities_global(0)

# Define numpy arrays to store velocity field over time
vel_np = np.zeros((nVertices, 3, obs_t_range.shape[0]))

# Define numpy arrays to store individual velocity components over time
u_i = np.zeros((nVertices, obs_t_range.shape[0]))
v_i = np.zeros((nVertices, obs_t_range.shape[0]))
w_i = np.zeros((nVertices, obs_t_range.shape[0]))

# Use a VTK probe filter to interpolate velocity onto the geometry
probe = vtk.vtkProbeFilter()
probe.SetInputData(geo)
probe.SetSourceData(velpv)
probe.Update()
geoWithVars = probe.GetOutput()

# Extract point data from VTK objects
vtk_pts = geoWithVars.GetPoints()
ptsArr = vtk_to_numpy(vtk_pts.GetData())
print("ptsArr", ptsArr.shape)

velArr = vtk_to_numpy(geoWithVars.GetPointData().GetArray('velocity'))
print("velArr", velArr.shape)

# Iterate over time steps and store velocity data
for i in tqdm(range(len(obs_t_range))):
    df_arr[:, 0] = ptsArr[:, 0]  # X coordinates
    df_arr[:, 1] = ptsArr[:, 1]  # Y coordinates
    df_arr[:, 2] = ptsArr[:, 2]  # Z coordinates
    df_arr[:, 3] = velArr[:, 0]  # U velocity component
    df_arr[:, 4] = velArr[:, 1]  # V velocity component
    df_arr[:, 5] = velArr[:, 2]  # W velocity component

    # Store velocity field over time
    vel_np[:, 0, i] = velArr[:, 0]
    vel_np[:, 1, i] = velArr[:, 1]
    vel_np[:, 2, i] = velArr[:, 2]

    # Store velocity components separately for interpolation purposes
    u_i[:, i] = velArr[:, 0]
    v_i[:, i] = velArr[:, 1]
    w_i[:, i] = velArr[:, 2]

# Define a function space for velocity in FEniCS
V = VectorFunctionSpace(mesh, "CG", 1)

# Save velocity data in HDF5 and XDMF formats
with XDMFFile(MPI.comm_world, join("./init-data_2.4/obs/AAA03/obs_noisy.xdmf")) as file:
    file.parameters.update({
        "functions_share_mesh": True,
        "rewrite_function_mesh": False,
        "flush_output": True
    })
    Hdf = HDF5File(MPI.comm_world, join('./init-data_2.4/obs/AAA03/obs_velocities_noisy.h5'), 'w')

    for i, t in enumerate(obs_t_range):
        # Define FEniCS function for observation
        u_obs = Function(V, name='obs')

        # Map velocity data onto function degrees of freedom
        u_obs.vector()[vertex_to_dof_map(V)] = vel_np[:, :, i].flatten()

        # Write velocity field to HDF5 and XDMF
        Hdf.write(u_obs, "u", i)
        file.write(u_obs, t)

    Hdf.close()
