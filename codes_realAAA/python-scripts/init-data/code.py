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

V = VectorFunctionSpace(mesh, 'CG', degree=2)
# Extract dofs for 2nd degree elements
dofmap = V.dofmap()
dofs = dofmap.dofs(mesh, 0)
dofs=np.array(dofs)
dofs=dofs.reshape(mesh.num_vertices(),3)

dof_map = Function(V, name="dof")
num_dof=dof_map.vector()[:].size
dof_map.vector()[:] = [int(i) for i in np.linspace(0, num_dof-1, num_dof)]
with XDMFFile("dof.xdmf") as xdmf:
    xdmf.write(dof_map)