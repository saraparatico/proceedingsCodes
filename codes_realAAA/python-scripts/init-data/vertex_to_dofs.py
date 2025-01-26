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

#---> Get patient
patient = 'AAA03' #!todo make it user parameter
## DO NOT NEED dx
#dx = '1.4'
psProfile = True

#---> Get time parameters
obs_dt = 0.021 #!todo make it user parameter
frames = 40
T = frames * obs_dt + DOLFIN_EPS
obs_t_range = np.arange(0, T, obs_dt)  #at t=0 velocity field is all zero
t_range = np.arange(0, T, obs_dt / 21)

#--->Set flag for interpolation over time
InterpolateDataOverTime = True

#---> Root
#root = '../femda/data'
root = '/home/biomech/sara/paratico-oasis/simple_forward_sara/tesi-Sara-Paratico'

#---> Directories
processed4DFlowDataDir = join(root, f'4DFlow/velocities/{patient}/cut')
meshDir = join(root, f'init-data_2.4/mesh/mesh_2.4')
outputDir = join(root, 'init-data_prova/inlet/')

#---> Read mesh with FEniCS
mesh_file = join(meshDir, f'{patient}.h5')
f = HDF5File(MPI.comm_world, mesh_file, 'r')
mesh = Mesh()
f.read(mesh, "mesh", False)
facet_ids = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
f.read(facet_ids, "/boundaries")
f.close()
inflow_id = 2

Q = FunctionSpace(mesh, "CG", 1)

# Ottieni la mappa vertice-doF
vtd = vertex_to_dof_map(Q)

# Ottieni le coordinate dei vertici
vertex_coordinates = mesh.coordinates()

# Ottieni le coordinate dei gradi di libertà
dof_coordinates = Q.tabulate_dof_coordinates()

# Apri un file CSV in modalità scrittura
with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Scrivi l'intestazione
    writer.writerow(['Vertex ID', 'DoF ID', 'Coordinate'])

    # Scrittura dei dati per ogni grado di libertà
    for vertex_id, dof_id in enumerate(vtd):
        print(vertex_id)
        coordinate = vertex_coordinates[vertex_id]
        writer.writerow([vertex_id, dof_id, coordinate])
