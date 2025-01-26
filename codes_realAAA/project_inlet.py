from dolfin import *
from dolfin_adjoint import *
import numpy as np
from os.path import join

#---> Get patient
patient = 'AAA03' #!todo make it user parameter
psProfile = True

#--->Set flag for interpolation over time
InterpolateDataOverTime = True
#---> Stationary/TimeVariant
stationary = True

#---> Directories
meshDir = join('./init-data_2.4/mesh/mesh_2.4')

#---> Read mesh with FEniCS
mesh_file = join(meshDir, f'{patient}.h5')
f = HDF5File(MPI.comm_world, mesh_file, 'r')
mesh = Mesh()
f.read(mesh, "mesh", False)
facet_ids = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
f.read(facet_ids, "/boundaries")
f.close()
inflow_id = 2

c = 0
frames = 40
obs_dt = 0.021
dt = 0.001
T = frames * obs_dt + DOLFIN_EPS
obs_t_range = np.arange(obs_dt, T, obs_dt)
t_range = np.arange(0, T, dt)

V = VectorFunctionSpace(mesh, 'CG', 1)
inletDir = "./init-data_2.4/inlet"

f = HDF5File(MPI.comm_world, join(inletDir + '/AAA03/mesh_2.4/int/ps/uin.h5'), 'r')
Q = VectorFunctionSpace(mesh, 'CG', 2)

if stationary:
    t = 63

    g = Function(V)
    f.read(g, 'u/vector_{}'.format(t))
    j = project(g, Q)

    r = Function(Q)
    #r.vector()[:]=0.15*j.vector()[:]
    r.vector()[:] = j.vector()[:]
    r.rename("g", "g_theta")

    XDMFFile("./g_63.xdmf").write(r)
    HDF5File(MPI.comm_world, join(inletDir + '/AAA03/mesh_2.4/int/ps/uin_prova.h5'), 'w').write(r, "u")

else:

    with XDMFFile(MPI.comm_world, join(inletDir, 'Inlet_TimeVariant_Real.xdmf')) as file:
        file.parameters.update(
            {
                "functions_share_mesh": True,
                "rewrite_function_mesh": False,
                "flush_output": True
            }
        )

        Hdf = HDF5File(MPI.comm_world, join(inletDir, 'Inlet_H5_TimeVariant_Real.h5'), 'w')

        for i, t in enumerate(t_range[1:]):
            t = np.round(t, 8)

            g_tmp = Function(V)
            f.read(g_tmp, 'u/vector_{}'.format(i + 1))
            j = project(g_tmp, Q)
            j.rename("g", "g_theta")

            if c % 1 == 0 and c != 0:
                file.write(j, t)
            c += 1

            Hdf.write(j, 'g', i)

        Hdf.close()
