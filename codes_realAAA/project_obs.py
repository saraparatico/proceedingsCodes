from dolfin import *
from dolfin_adjoint import *
import numpy as np
from os.path import join
from collections import OrderedDict

#---> Get patient
patient = 'AAA03' #!todo make it user parameter
psProfile = True

#---> Get time parameters
frames = 3# 40
obs_dt = 0.021
dt = 0.001
T = frames * obs_dt + DOLFIN_EPS
obs_t_range = np.arange(obs_dt, T, obs_dt)
t_range = np.arange(0, T, dt)

#--->Set flag for interpolation over time
InterpolateDataOverTime = True

#---> Root
#root = '../femda/data'
#root = '/home/biomech/sara/paratico-oasis/simple_forward_sara/tesi-Sara-Paratico'

#---> Directories
#processed4DFlowDataDir = join('./4DFlow/velocities/{patient}/cut')
meshDir = join('./init-data_2.4/mesh/mesh_2.4')
outputDir = join('./init-data_prova/inlet/')

#---> Read mesh with FEniCS
mesh_file = join(meshDir, f'{patient}.h5')
f = HDF5File(MPI.comm_world, mesh_file, 'r')
mesh = Mesh()
f.read(mesh, "mesh", False)
facet_ids = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
f.read(facet_ids, "/boundaries")
f.close()
inflow_id = 2

V = VectorFunctionSpace(mesh, 'CG', 1)
Q = VectorFunctionSpace(mesh, 'CG', 2)

obsDir = "./init-data_2.4/obs"

f = HDF5File(MPI.comm_world, join(obsDir + '/AAA03/mesh_2.4/uobs.h5'), 'r')

c = 0
stationary = False

if stationary:
    t = 3

    g = Function(V)
    f.read(g, 'u/vector_{}'.format(t))
    j = project(g, Q)
    j.rename("obs", "u_obs")

    XDMFFile("./obs_3.xdmf").write(g)

    HDF5File(MPI.comm_world, join(obsDir + '/AAA03/mesh_2.4/uobs_velocity3.h5'), 'w').write(j, "u")

else:

    with XDMFFile(MPI.comm_world, join(obsDir, 'Obs_TimeVariant_RealData_provaReading.xdmf')) as file:
        file.parameters.update(
            {
                "functions_share_mesh": True,
                "rewrite_function_mesh": False,
                "flush_output": True
            }
        )

        Hdf = HDF5File(MPI.comm_world, join(obsDir, 'Obs_H5_TimeVariant_RealData_provaReading.h5'), 'w')

        for i, t in enumerate(obs_t_range[:]):
            t = np.round(t, 8)

            obs_tmp = Function(V)
            f.read(obs_tmp, 'u/vector_{}'.format(i + 1))
            j = project(obs_tmp, Q)
            j.rename("obs", "u_obs")

            file.write(j, t)
            print("at", t, ":", j.vector().max())

            Hdf.write(j, 'obs', i)

        Hdf.close()

print("the end")


