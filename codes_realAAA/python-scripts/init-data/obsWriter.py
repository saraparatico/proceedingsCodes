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

#---> Get patient
patient = 'AAA03' #!todo make it user parameter
## DO NOT NEED IT
#dx = '1.4'

#---> Get time parameters
obs_dt = 0.021 #!todo make it user parameter
frames = 40
T = frames * obs_dt + DOLFIN_EPS
obs_t_range = np.arange(0, T, obs_dt)  #at t=0 velocity field is all zero
t_range = np.arange(0, T, obs_dt / 21)

#--->Set flag for interpolation over time
InterpolateDataOverTime = False

#---> Root
#root = '../femda/data'
root = '/home/biomech/sara/paratico-oasis/simple_forward_sara/tesi-Sara-Paratico'

#---> Directories
processed4DFlowDataDir = join(root, f'4DFlow/velocities/{patient}/cut')
meshDir = join(root, f'init-data/mesh')
outputDir = join(root, 'init-data/obs/')

#---> Read mesh with FEniCS
mesh_file = join(meshDir, f'{patient}.h5')
f = HDF5File(MPI.comm_world, mesh_file, 'r')
mesh = Mesh()
f.read(mesh, "mesh", False)
facet_ids = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
f.read(facet_ids, "/boundaries")
f.close()

#---> Read 4D flow data
allData = []
processed4DFlowFiles = sorted(glob(join(processed4DFlowDataDir, '*.vtu')))

for f in tqdm(range(len(processed4DFlowFiles)), desc='Reading processed vtk frames'):

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(processed4DFlowFiles[f])
    reader.Update()
    grid = reader.GetOutput()

    # ---> Remove shit from vtu of 4D flow
    velpv = pv.wrap(grid)
    velpvvel = velpv['velocity']
    velpv.clear_data()
    velpv['velocity'] = velpvvel

    allData.append(velpv)

#---> Read registered mesh, tagret mesh for probing
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(join(meshDir, f'{patient}.vtu'))
reader.Update()
geo = reader.GetOutput()

#---> Define field names for store velocity data
fieldnames = ['x', 'y', 'z', 'u', 'v', 'w']
df_arr = np.empty((geo.GetNumberOfPoints(), len(fieldnames)))

nVertices = mesh.num_vertices()

#---> Define a numpy vector to store velocity field in time
vel_np = np.zeros((nVertices, 3, obs_t_range.shape[0]))

#---> Define numpy vectors to store velocity components in time
u_i = np.zeros((nVertices, obs_t_range.shape[0]))
v_i = np.zeros((nVertices, obs_t_range.shape[0]))
w_i = np.zeros((nVertices, obs_t_range.shape[0]))

for f in tqdm(range(len(allData)), desc='Writing point data for frame'):
    vel_grid = allData[f]

    probe = vtk.vtkProbeFilter()

    probe.SetInputData(geo)
    probe.SetSourceData(vel_grid)
    probe.Update()
    geoWithVars = probe.GetOutput()
    vtk_pts = geoWithVars.GetPoints()

    ptsArr = vtk_to_numpy(vtk_pts.GetData())
    velArr = vtk_to_numpy(geoWithVars.GetPointData().GetArray('velocity'))

    df_arr[:, 0] = ptsArr[:, 0]
    df_arr[:, 1] = ptsArr[:, 1]
    df_arr[:, 2] = ptsArr[:, 2]
    df_arr[:, 3] = velArr[:, 0]
    df_arr[:, 4] = velArr[:, 1]
    df_arr[:, 5] = velArr[:, 2]

    #---> Store velocity field in time
    vel_np[:, 0, f + 1] = velArr[:, 0]
    vel_np[:, 1, f + 1] = velArr[:, 1]
    vel_np[:, 2, f + 1] = velArr[:, 2]

    #---> Store velocity components in time for interpolation purpose
    u_i[:, f + 1] = velArr[:, 0]
    v_i[:, f + 1] = velArr[:, 1]
    w_i[:, f + 1] = velArr[:, 2]

    # ---> Save velocity and space data as csv file
    df = pd.DataFrame(df_arr, columns=fieldnames)
    df.to_csv(join(outputDir, '{}/mesh/probedData/point_data_{:02d}.csv'.format(patient, f)), index=False)

    # ---> Save velocity data as vtu file for visualization purpose
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(join(outputDir, '{}/mesh/probedData/probedData_{:02d}.vtp'.format(patient, f)))
    writer.SetInputData(geoWithVars)
    writer.Update()

#---> Define function space for velocity field
V = VectorFunctionSpace(mesh, "CG", 1)  # Velocity

output_file_path = join(outputDir, '{}/mesh/all_OBS.h5'.format(patient))
Hdf = HDF5File(MPI.comm_world, output_file_path, 'w')

if not InterpolateDataOverTime:
    for i, t in enumerate(obs_t_range):

        #---> Define function object for observation
        u_obs = Function(V, name='obs')
        u_obs.vector()[vertex_to_dof_map(V)] = vel_np[:, :, i].flatten()
        #xf_u_obs = XDMFFile(join(root, 'u_obs_{:02d}.xdmf'.format(f)))
        #xf_u_obs.write(project(u_obs, V))
        # ---> Save: NO! I do not want a t number of savings
        #Hdf = HDF5File(MPI.comm_world, join(outputDir, '{}/mesh/uin_{:02d}.h5'.format(patient, t)), 'w')
        #Hdf.write(u_obs, "u")
        #Hdf.close()
        dataset_name = 'u_{:02}'.format(t)  # Modificare il nome del dataset se necessario
        Hdf.write(u_obs, dataset_name)
Hdf.close()
if InterpolateDataOverTime:
    #---> Interpolate in time velocity data
    vel_npinterp = np.zeros((nVertices, 3, t_range.shape[0]))

    for idx in tqdm(range(nVertices)):
        #---> u
        f = interpolate.interp1d(list(obs_t_range), list(u_i[idx, :]), kind='quadratic')
        vel_npinterp[idx, 0, :] = f(t_range)
        #---> v
        f = interpolate.interp1d(list(obs_t_range), list(v_i[idx, :]), kind='quadratic')
        vel_npinterp[idx, 1, :] = f(t_range)
        #---> w
        f = interpolate.interp1d(list(obs_t_range), list(w_i[idx, :]), kind='quadratic')
        vel_npinterp[idx, 2, :] = f(t_range)

        #vel_npinterp[idx, 0, :] = np.interp(list(t_range), list(obs_t_range), list(u_i[idx, :]))
        #vel_npinterp[idx, 1, :] = np.interp(list(t_range), list(obs_t_range), list(v_i[idx, :]))
        #vel_npinterp[idx, 2, :] = np.interp(list(t_range), list(obs_t_range), list(w_i[idx, :]))

    #---> Initialize observation vector
    t = 0
    u_obs = Function(V, name='obs')  # Observation vector

    # ---> Check
    #xf_u_obs = XDMFFile(join(outputDir, '{}/mesh_{}/check/uobs_{:.8f}.xdmf'.format(patient, dx, np.round(t, 8))))
    #xf_u_obs.write(u_obs)

    Hdf = HDF5File(MPI.comm_world, join(outputDir, '{}/mesh/uobs_{:.8f}.h5'.format(patient, np.round(t, 8))), 'w')
    Hdf.write(u_obs, "u")
    Hdf.close()

    for i, t in enumerate(t_range[1:]):
        u_obs.vector()[vertex_to_dof_map(V)] = vel_npinterp[:, :, i].flatten()  # Observation vector inizialization

        #if i % 42 == 0:
            # ---> Check
            #xf_u_obs = XDMFFile(join(outputDir, '{}/mesh_{}/check/uobs_{:.8f}.xdmf'.format(patient, dx, np.round(t, 8))))
            #xf_u_obs.write(u_obs)

        Hdf = HDF5File(MPI.comm_world, join(outputDir, '{}/mesh/uobs_{:.8f}.h5'.format(patient, np.round(t, 8))), 'w')
        Hdf.write(u_obs, "u")
        Hdf.close()