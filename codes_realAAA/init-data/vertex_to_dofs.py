################################################################################
############################ - vertex_to_dofs.py - #############################
# This script generates a CSV file containing the Vertex IDs, DoFs IDs and
# their corresponding coordinates.
# It's not fundamental for running the simulations, it's more like an
# additional script for analysis.
# !!! IMPORTANT: MODIFY LINE 45 TO SET YOUR OWN ROOT DIRECTORY !!!
################################################################################
################################################################################

import csv
from dolfin import *

# Definisci la mesh e lo spazio delle funzioni
import time

from dolfin import *
from dolfin_adjoint import *
from scipy import interpolate
from os.path import join
import vtk
from tqdm import tqdm
import numpy as np
import pandas as pd
from vtkmodules.util.numpy_support import vtk_to_numpy
from glob import glob
import pyvista as pv
from scipy.interpolate import splrep, splev

# ---> Define patient ID (TODO: this should be a user parameter)
patient = 'AAA03'

psProfile = True  # Set a flag indicating the use of pressure profile (True or False)

# ---> Define time parameters (TODO: obs_dt  and frames should be user parameters)
obs_dt = 0.021  # Time step between frames
frames = 40  # Number of frames in the dataset
T = frames * obs_dt + DOLFIN_EPS  # Total simulation time
obs_t_range = np.arange(0, T, obs_dt)  # Time steps for velocity field (t=0 -> zero velocity)
t_range = np.arange(0, T, obs_dt / 21)  # Finer time range for interpolation
#---> Set flag for interpolation over time
InterpolateDataOverTime = True  # Flag to enable interpolation of velocity data over time

# ---> Define root directory (!!! Modify this path according to your setup !!!)
root = '/home/biomech/sara/paratico-oasis/simple_forward_sara/tesi-Sara-Paratico'  # Actual root directory for the project

#---> Directories for specific data locations
processed4DFlowDataDir = join(root, f'4DFlow/velocities/{patient}/cut')  # Directory for processed 4D flow velocity data
meshDir = join(root, f'init-data_2.4/mesh/mesh_2.4')  # Directory for mesh data
outputDir = join(root, 'init-data_prova/inlet/')  # Directory for output (inlet data)

#---> Read mesh with FEniCS
mesh_file = join(meshDir, f'{patient}.h5')
f = HDF5File(MPI.comm_world, mesh_file, 'r')
mesh = Mesh()
f.read(mesh, "mesh", False)
facet_ids = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
f.read(facet_ids, "/boundaries")
f.close()

inflow_id = 2  # Inflow boundary marker ID (to identify the inlet boundary)

Q = FunctionSpace(mesh, "CG", 1)  # Define function space (Continuous Galerkin, first order)

#---> Get vertex-to-degree-of-freedom map
vtd = vertex_to_dof_map(Q)  # Mapping from vertices to degrees of freedom (DoF)

#---> Get vertex coordinates
vertex_coordinates = mesh.coordinates()  # Array of coordinates for each vertex in the mesh

#---> Get degree-of-freedom coordinates
dof_coordinates = Q.tabulate_dof_coordinates()  # Coordinates of each degree of freedom in the function space

#---> Write coordinates and IDs to CSV
with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write header row to the CSV
    writer.writerow(['Vertex ID', 'DoF ID', 'Coordinate'])  # Column names for vertex ID, DoF ID, and coordinates

    # Write data for each degree of freedom
    for vertex_id, dof_id in enumerate(vtd):  # Iterate over all vertices and their corresponding DoF
        print(vertex_id)  # Print the vertex ID (for progress tracking)
        coordinate = vertex_coordinates[vertex_id]  # Get the coordinates for the vertex
        writer.writerow([vertex_id, dof_id, coordinate])  # Write the data for the vertex and its DoF to the CSV
