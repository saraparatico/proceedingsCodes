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

#---> Read 4D flow data
allData = []
processed4DFlowFiles = sorted(glob(join(processed4DFlowDataDir, '*.vtu')))

for f in tqdm(range(len(processed4DFlowFiles)), desc='Reading processed vtu frames'):

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(processed4DFlowFiles[f])
    reader.Update()
    grid = reader.GetOutput()
    #---> Remove shit from vtu of 4D flow
    velpv = pv.wrap(grid)
    velpvvel = velpv['velocity']
    velpv.clear_data()
    velpv['velocity'] = velpvvel

    allData.append(velpv)

#---> Read registered mesh, tagret mesh for probing
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(join(meshDir, f'{patient}.vtu'))
reader.Update()
msh = reader.GetOutput()

#---> Read registered inlet mesh, target mesh for probing
reader = vtk.vtkSTLReader()
reader.SetFileName(join(meshDir, f'{patient}_inlet.stl'))
reader.Update()
geo = reader.GetOutput()

#---> Define field names for store velocity data
fieldnames = ['x', 'y', 'z', 'u', 'v', 'w']
#df_arr = np.empty((geo.GetNumberOfPoints(), len(fieldnames)))
df_arr = np.empty((msh.GetNumberOfPoints(), len(fieldnames))) #questa modifica mi permette di definire senza passare dal codice bctfilewriter.py

#---> Define Inlet Mesh object
bmsh = BoundaryMesh(mesh, 'exterior')

# copy the meshfunction, but defined on the boundarymesh this time
bdim = bmsh.topology().dim()
boundary_boundaries = MeshFunction('size_t', bmsh, bdim)
boundary_boundaries.set_all(0)
for i, facet in enumerate(entities(bmsh, bdim)):
    parent_meshentity = bmsh.entity_map(bdim)[i]
    parent_boundarynumber = facet_ids.array()[parent_meshentity]
    boundary_boundaries.array()[i] = parent_boundarynumber

inletMesh = SubMesh(bmsh, boundary_boundaries, 2)

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

    #---> Match inlet coordinate with mesh coordinate
    geo_pv = pv.wrap(geo)
    msh_pv = pv.wrap(msh)

    for p, inletpoint in enumerate(geo_pv.points):
        i = msh_pv.find_closest_point(inletpoint)
        df_arr[i, 0] = ptsArr[p, 0]
        df_arr[i, 1] = ptsArr[p, 1]
        df_arr[i, 2] = ptsArr[p, 2]
        # ---> Store velocity field in time and store velocity components in time for interpolation purpose
        u_i[i, f + 1] = vel_np[i, 0, f + 1] = df_arr[i, 3] = velArr[p, 0]
        v_i[i, f + 1] = vel_np[i, 1, f + 1] = df_arr[i, 4] = velArr[p, 1]
        w_i[i, f + 1] = vel_np[i, 2, f + 1] = df_arr[i, 5] = velArr[p, 2]

    #---> Save velocity and space data as csv file
    df = pd.DataFrame(df_arr, columns=fieldnames)
    df.to_csv(join(outputDir, '{}/mesh/probedData/point_data_{:02d}.csv'.format(patient, f)), index=False)

    #---> Save velocity data as vtu file for visualization purpose
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(join(outputDir, '{}/mesh/probedData/probedData_{:02d}.vtp'.format(patient, f)))
    writer.SetInputData(geoWithVars)
    writer.Update()

#---> Define function space for velocity field
V = VectorFunctionSpace(mesh, "CG", 1)  # Velocity

# CORRECT WAY
output_file_path = join(outputDir, '{}/mesh/all_results_{}.h5'.format(patient, psProfile))
Hdf_inlet = HDF5File(MPI.comm_world, output_file_path, 'w')

if not InterpolateDataOverTime:

    for i, t in enumerate(obs_t_range):

        #---> Define function object for observation
        u_obs = Function(V)
        u_obs.vector()[vertex_to_dof_map(V)] = vel_np[:, :, i].flatten()

        #xf_u_obs = XDMFFile(join(root, 'u_obs_{:02d}.xdmf'.format(f)))
        #xf_u_obs.write(project(u_obs, V))

        #---> Save: NO! Because it saves a number t of .h5 files, I don't want it!
        #Hdf = HDF5File(MPI.comm_world, join(outputDir, '{}/mesh_{}/uin_{:02d}.h5'.format(patient, dx, t)), 'w')
        #Hdf.write(u_obs, "u")
        #Hdf.close()
        # CORRECT WAY
        dataset_name = "u_{:02d}".format(t)  # Modificare il nome del dataset se necessario
        Hdf_inlet.write(u_obs, dataset_name)
else:
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

    #---> Interpolation Check
    #for i, t in enumerate(t_range[1:]):
    #    inlet_pv = pv.wrap(mesh.coordinates())
    #    inlet_pv.point_data['velocity'] = vel_npinterp[:, :, i]
    #    inlet_pv.save(join(outputDir, '{}/check/uin_{:.8f}.vtp'.format(patient, np.round(t, 8))))

    #---> Initialize observation vector
    t = 0
    u_in = Function(V, name='inlet')  # Observation vector

    if psProfile:
        #---> Check
        xf_u_in = XDMFFile(join(outputDir, '{}/mesh/check/uin_{:.8f}.xdmf'.format(patient, np.round(t, 8))))
        xf_u_in.write(u_in)

        Hdf = HDF5File(MPI.comm_world, join(outputDir, '{}/mesh/ps/uin_{:.8f}.h5'.format(patient, np.round(t, 8))), 'w')

        Hdf_inlet.write(u_in, 'u', 0)
        for i, t in enumerate(t_range[1:]):
            u_in.vector()[vertex_to_dof_map(V)] = vel_npinterp[:, :, i].flatten()  # Observation vector inizialization

            if i % 42 == 0:
                #---> Check
                xf_u_obs = XDMFFile(join(outputDir, '{}/mesh/check/uin_{:.8f}.xdmf'.format(patient, np.round(t, 8))))
                xf_u_obs.write(u_in)

            #Hdf = HDF5File(MPI.comm_world, join(outputDir, '{}/mesh/ps/uin_{:.8f}.h5'.format(patient, np.round(t, 8))), 'w')
            #Hdf.write(u_in, "u")
            #Hdf.close()
            #dataset_name = "u_{:.3f}".format(t)  # Modificare il nome del dataset se necessario
            Hdf_inlet.write(u_in, 'u', i + 1)

    else:
        # ---> Check
        xf_u_in = XDMFFile(join(outputDir, '{}/mesh/check/uin_{:.8f}.xdmf'.format(patient, np.round(t, 8))))
        xf_u_in.write(u_in)

        Hdf = HDF5File(MPI.comm_world, join(outputDir, '{}/mesh/plug/uin_{:.8f}.h5'.format(patient,
                                                                                              np.round(t, 8))), 'w')
        Hdf.write(u_in, "u")
        Hdf.close()

        # ---> Define Inlet Plug
        # Here I define inlet plug from the averaged velocity taken from 4D flow
        # the implementation was taken from https://fenicsproject.discourse.group/t/define-a-time-varying-parabolic-profile-normal-to-a-boundary/6041

        df = pd.read_csv(join(outputDir, '{}/mesh/uin.csv'.format(patient)), sep=',')
        averagedvel = np.array(df.iloc[:]['Velocity'])


        def makeIC():
            v = averagedvel

            return splrep(obs_t_range[1:], v)

        # Approximate facet normal in a suitable space using projection
        n = FacetNormal(mesh)
        # V = VectorFunctionSpace(mesh, "CG", 2)
        u_ = TrialFunction(V)
        v_ = TestFunction(V)
        a = inner(u_, v_) * ds
        l = inner(n, v_) * ds
        A = assemble(a, keep_diagonal=True)
        L = assemble(l)

        A.ident_zeros()
        nh = Function(V)

        solve(A, nh.vector(), L)

        class InflowBoundaryValue(UserExpression):
            def __init__(self, t=None, period=None, **kwargs):
                super().__init__(**kwargs)
                self.t = t
                self.t_p = period
                self.bc_func = makeIC()

            def eval(self, values, x):
                # n = data.cell().normal(data.facet())
                n_eval = nh(x)
                t = self.t
                val = splev(t - int(t / self.t_p) * self.t_p, self.bc_func)
                values[0] = -n_eval[0] * val
                values[1] = -n_eval[1] * val
                values[2] = -n_eval[2] * val

            def value_shape(self):
                return (3,)

        expr = InflowBoundaryValue(t=0, period=T)

        inlet = DirichletBC(V, expr, facet_ids, inflow_id)

        for c, t in enumerate(t_range[1:]):
            expr.t = t
            t = np.round(t, 8)
            u_in = Function(V, name='inlet')
            # u_in.interpolate(expr)
            t0 = time.time()
            inlet.apply(u_in.vector())
            t1 = time.time()
            print(t1 - t0)
            if c % 42 == 0:
                xf_u_obs = XDMFFile(
                    join(outputDir, '{}/mesh/check/uin_{:.8f}.xdmf'.format(patient, np.round(t, 8))))
                xf_u_obs.write(u_in)
            Hdf = HDF5File(MPI.comm_world, join(outputDir, '{}/mesh/plug/uin_{:.8f}.h5'.format(patient,
                                                                                                  np.round(t, 8))), 'w')
            Hdf.write(u_in, "u")
            Hdf.close()


Hdf_inlet.close()

#test outflow boundary

# Approximate facet normal in a suitable space using projection
n = FacetNormal(mesh)
# V = VectorFunctionSpace(mesh, "CG", 2)
u_ = TrialFunction(V)
v_ = TestFunction(V)
a = inner(u_, v_) * ds
l = inner(n, v_) * ds
A = assemble(a, keep_diagonal=True)
L = assemble(l)

A.ident_zeros()
nh = Function(V)

solve(A, nh.vector(), L)

ds = Measure('ds', domain=mesh, subdomain_data=facet_ids)
area = assemble(1 * ds(3))

class OutflowBoundaryValue(UserExpression):
    def __init__(self, val=None, **kwargs):
        super().__init__(**kwargs)
        self.val = val

    def eval(self, values, x):
        # n = data.cell().normal(data.facet())
        n_eval = nh(x)

        val = self.val

        values[0] = -n_eval[0] * val
        values[1] = -n_eval[1] * val
        values[2] = -n_eval[2] * val

    def value_shape(self):
        return (3,)

u_in = Function(V, name='inlet')

expr = OutflowBoundaryValue(val=0)

import time
outlet = DirichletBC(V, expr, facet_ids, 3)

if not psProfile:
    for c, t in enumerate(t_range[1:]):
        Hdf = HDF5File(MPI.comm_world, join(outputDir, '{}/mesh/plug/uin_{:.8f}.h5'.format(patient,
                                                                                          np.round(t, 8))), 'r')
        Hdf.read(u_in, "u")
        Hdf.close()

        flux = dot(u_in, - n) * ds(2)
        Qin = assemble(flux)
        Qout = Qin / 2
        val = Qout / area

        expr.val = val
        u_out = Function(V, name='outlet')
        t0 = time.time()
        outlet.apply(u_out.vector())
        t1 = time.time()
        print(t1 - t0)
        if c % 42 == 0:
            xf_u_obs = XDMFFile(
                join(outputDir, '{}/mesh/check/uout_{:.8f}.xdmf'.format(patient, np.round(t, 8))))
            xf_u_obs.write(u_out)