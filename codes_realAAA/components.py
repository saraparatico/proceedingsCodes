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

### MESH LOADING
mesh_vtu = "./init-data_2.4/mesh/mesh_2.4/AAA03.vtu"
target_mesh = pv.read(mesh_vtu)

mesh_h5 = "./init-data_2.4/mesh/mesh_2.4/AAA03.h5"
f = HDF5File(MPI.comm_world, mesh_h5, 'r')
mesh = Mesh()
f.read(mesh, "mesh", False)
facet_ids = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
f.read(facet_ids, "/boundaries")
f.close()

#### NOISY DATA LOADING
vel_file = "/home/biomech/sara/paratico-oasis/simple_forward_sara/tesi-Sara-Paratico/init-data_2.4/aaa03_ivar0.0005_center5_snr10_uniformGridNoisy.vti"
grid = pv.read(vel_file)

### VEL COMPONENTS EXTRACTION
u = grid["u"]
u_array = np.array(u)
v = grid["v"]
v_array = np.array(v)
w = grid["w"]
w_array = np.array(w)

### COMBINATION OF COMPONENTS 1)
velocity = np.column_stack((u, v, w))
print(velocity.shape)
### COMBINATION OF COMPONENTS 2)
def calcola_velocita_risultante(vx, vy, vz):
    return np.sqrt(vx**2 + vy**2 + vz**2)
resulting_v = calcola_velocita_risultante(u_array, v_array, w_array)
print("Final v vector:", resulting_v)
print(len(resulting_v))


# Aggiungi il vettore 'velocity' al griglia PyVista
grid["velocity"] = velocity

# Interpola il dato 'velocity' sulla mesh
interpolated = target_mesh.interpolate(grid, radius=2.0)
# <-- interpolated però interpola su una mesh di 22000 elementi e non sui 60000 dofs!

# Salva la nuova mesh con i valori di velocità combinati in un nuovo file .vtu
output_file = "./init-data_2.4/obs/AAA03/obs_noisy.vtu"
interpolated.save(output_file)


###### FROM .VTU TO .H5
frames = 40
T = 0.840
obs_dt = T/frames #!todo make it user parameter
T = frames * obs_dt + DOLFIN_EPS
obs_t_range = np.arange(0, T, obs_dt)  #at t=0 velocity field is all zero
dt = 0.001  #!todo make it user parameter
t_range = np.arange(0, T, dt)

reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(output_file)
reader.Update()
grid = reader.GetOutput()

# ---> Remove shit from vtu of 4D flow
velpv = pv.wrap(grid)
velpvvel = velpv['velocity']
velpv.clear_data()
velpv['velocity'] = velpvvel
#print(data.shape) --> (22536, 3)

reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(mesh_vtu)
reader.Update()
geo = reader.GetOutput()
#print("points:", geo.GetNumberOfPoints()) --> 22536

#---> Define field names for store velocity data
fieldnames = ['x', 'y', 'z', 'u', 'v', 'w']
df_arr = np.empty((geo.GetNumberOfPoints(), len(fieldnames)))

nVertices = mesh.num_entities_global(0)
# print("vertices:", nVertices) --> 22536

#---> Define a numpy vector to store velocity field in time
vel_np = np.zeros((nVertices, 3, obs_t_range.shape[0]))
# print("ve_np:", vel_np.shape) --> (22536, 3, 41)

#---> Define numpy vectors to store velocity components in time
u_i = np.zeros((nVertices, obs_t_range.shape[0]))
v_i = np.zeros((nVertices, obs_t_range.shape[0]))
w_i = np.zeros((nVertices, obs_t_range.shape[0]))

probe = vtk.vtkProbeFilter()
probe.SetInputData(geo)
probe.SetSourceData(velpv)
probe.Update()
geoWithVars = probe.GetOutput()
vtk_pts = geoWithVars.GetPoints()

ptsArr = vtk_to_numpy(vtk_pts.GetData())
print("ptsArr", ptsArr.shape)
velArr = vtk_to_numpy(geoWithVars.GetPointData().GetArray('velocity'))
print("vell_Arr", velArr.shape)

for i in tqdm(range(len(obs_t_range))):
    df_arr[:, 0] = ptsArr[:, 0]
    df_arr[:, 1] = ptsArr[:, 1]
    df_arr[:, 2] = ptsArr[:, 2]
    df_arr[:, 3] = velArr[:, 0]
    df_arr[:, 4] = velArr[:, 1]
    df_arr[:, 5] = velArr[:, 2]

    #---> Store velocity field in time
    vel_np[:, 0, i] = velArr[:, 0]
    vel_np[:, 1, i] = velArr[:, 1]
    vel_np[:, 2, i] = velArr[:, 2]

    #---> Store velocity components in time for interpolation purpose
    u_i[:, i] = velArr[:, 0]
    v_i[:, i] = velArr[:, 1]
    w_i[:, i] = velArr[:, 2]

V = VectorFunctionSpace(mesh, "CG", 1)
with XDMFFile(MPI.comm_world, join("./init-data_2.4/obs/AAA03/obs_noisy.xdmf")) as file:
    file.parameters.update(
        {
            "functions_share_mesh": True,
            "rewrite_function_mesh": False,
            "flush_output": True
        }
    )
    Hdf = HDF5File(MPI.comm_world, join('./init-data_2.4/obs/AAA03/obs_velocities_noisy.h5'), 'w')
    for i, t in enumerate(obs_t_range):
        #---> Define function object for observation
        u_obs = Function(V, name='obs')
        #print(u_obs.vector()[:].shape)
        u_obs.vector()[vertex_to_dof_map(V)] = vel_np[:, :, i].flatten()
        #u_obs.vector()[vertex_to_dof_map(V)] = velocita_risultante[i]

        Hdf.write(u_obs, "u", i)

        file.write(u_obs, t)

    Hdf.close()
