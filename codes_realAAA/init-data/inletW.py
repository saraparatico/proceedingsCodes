################################################################################
############################### - inletW.py - #################################
# This script reads the velocity files to be used as inlet boundary
# conditions for the forward simulation. It should be run after the execution
# of convertMesh.py, but before running the simulation.
# !!! IMPORTANT: MODIFY LINE 39 TO SET YOUR OWN ROOT DIRECTORY !!!
################################################################################
################################################################################

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

# ---> Flag for interpolation over time
InterpolateDataOverTime = True

# ---> Define root directory (!!! Modify this path according to your setup !!!)
root = '/home/biomech/sara/paratico-oasis/simple_forward_sara/tesi-Sara-Paratico'

# ---> Define directories for input and output data
processed4DFlowDataDir = join(root, f'4DFlow/velocities/{patient}/cut')  # 4D Flow velocity data
meshDir = join(root, f'init-data_2.4/mesh/mesh_2.4')  # Mesh directory
outputDir = join(root, 'init-data_prova/inlet/')  # Output directory

# ---> Read mesh using FEniCS
mesh_file = join(meshDir, f'{patient}.h5')
f = HDF5File(MPI.comm_world, mesh_file, 'r')
mesh = Mesh()
f.read(mesh, "mesh", False)
facet_ids = MeshFunction("size_t", mesh, mesh.topology().dim()-1)  # Boundary facet markers
f.read(facet_ids, "/boundaries")  # Read boundary data
f.close()
inflow_id = 2  # ID of the inlet boundary

# ---> Read processed 4D Flow velocity data (VTU files)
allData = []
processed4DFlowFiles = sorted(glob(join(processed4DFlowDataDir, '*.vtu')))  # List VTU files in order

for f in tqdm(range(len(processed4DFlowFiles)), desc='Reading processed vtu frames'):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(processed4DFlowFiles[f])
    reader.Update()
    grid = reader.GetOutput()

    # ---> Clean up VTU data (remove unnecessary fields)
    velpv = pv.wrap(grid)  # Convert to PyVista object
    velpvvel = velpv['velocity']  # Extract velocity field
    velpv.clear_data()  # Remove all other data
    velpv['velocity'] = velpvvel  # Reassign velocity data

    allData.append(velpv)  # Store processed data

# ---> Read registered (deformed) mesh, target for probing
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(join(meshDir, f'{patient}.vtu'))
reader.Update()
msh = reader.GetOutput()

# ---> Read registered inlet mesh (target surface for velocity probing)
reader = vtk.vtkSTLReader()
reader.SetFileName(join(meshDir, f'{patient}_inlet.stl'))
reader.Update()
geo = reader.GetOutput()  # Load STL geometry

# ---> Define field names for velocity storage
fieldnames = ['x', 'y', 'z', 'u', 'v', 'w']
df_arr = np.empty((msh.GetNumberOfPoints(), len(fieldnames)))  # Preallocate velocity storage

# ---> Define boundary mesh object for inlet region
bmsh = BoundaryMesh(mesh, 'exterior')  # Extract boundary from the full mesh

# ---> Copy boundary markers to the new boundary mesh
bdim = bmsh.topology().dim()
boundary_boundaries = MeshFunction('size_t', bmsh, bdim)
boundary_boundaries.set_all(0)  # Initialize all boundaries as 0
for i, facet in enumerate(entities(bmsh, bdim)):
    parent_meshentity = bmsh.entity_map(bdim)[i]
    parent_boundarynumber = facet_ids.array()[parent_meshentity]  # Map to parent mesh
    boundary_boundaries.array()[i] = parent_boundarynumber  # Assign boundary IDs

inletMesh = SubMesh(bmsh, boundary_boundaries, 2)  # Extract only the inlet submesh

nVertices = mesh.num_vertices()  # Number of mesh vertices

# ---> Define numpy arrays to store velocity field over time
vel_np = np.zeros((nVertices, 3, obs_t_range.shape[0]))  # Full velocity field storage
u_i = np.zeros((nVertices, obs_t_range.shape[0]))  # u-component
v_i = np.zeros((nVertices, obs_t_range.shape[0]))  # v-component
w_i = np.zeros((nVertices, obs_t_range.shape[0]))  # w-component

# ---> Process each 4D Flow frame
for f in tqdm(range(len(allData)), desc='Writing point data for frame'):
    vel_grid = allData[f]  # Extract velocity data for frame f

    # ---> Probe velocity field onto the inlet mesh
    probe = vtk.vtkProbeFilter()
    probe.SetInputData(geo)  # Use inlet geometry as probe target
    probe.SetSourceData(vel_grid)  # Use velocity field as source data
    probe.Update()
    geoWithVars = probe.GetOutput()  # Get probed velocity data
    vtk_pts = geoWithVars.GetPoints()

    ptsArr = vtk_to_numpy(vtk_pts.GetData())  # Convert point coordinates to NumPy array
    velArr = vtk_to_numpy(geoWithVars.GetPointData().GetArray('velocity'))  # Extract velocity

    # ---> Match inlet coordinates with mesh coordinates
    geo_pv = pv.wrap(geo)
    msh_pv = pv.wrap(msh)

    for p, inletpoint in enumerate(geo_pv.points):
        i = msh_pv.find_closest_point(inletpoint)  # Find nearest mesh point
        df_arr[i, 0] = ptsArr[p, 0]
        df_arr[i, 1] = ptsArr[p, 1]
        df_arr[i, 2] = ptsArr[p, 2]
        # ---> Store velocity components for interpolation
        u_i[i, f + 1] = vel_np[i, 0, f + 1] = df_arr[i, 3] = velArr[p, 0]
        v_i[i, f + 1] = vel_np[i, 1, f + 1] = df_arr[i, 4] = velArr[p, 1]
        w_i[i, f + 1] = vel_np[i, 2, f + 1] = df_arr[i, 5] = velArr[p, 2]

    # ---> Save velocity and spatial data as CSV file
    df = pd.DataFrame(df_arr, columns=fieldnames)
    df.to_csv(join(outputDir, f'{patient}/mesh/probedData/point_data_{f:02d}.csv'), index=False)

    # ---> Save velocity data as VTP file for visualization
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(join(outputDir, f'{patient}/mesh/probedData/probedData_{f:02d}.vtp'))
    writer.SetInputData(geoWithVars)
    writer.Update()

# ---> Define function space for velocity field in FEniCS
V = VectorFunctionSpace(mesh, "CG", 1)  # Continuous Galerkin order 1 for velocity

# ---> Save all velocity results in a single HDF5 file
output_file_path = join(outputDir, f'{patient}/mesh/all_results_{psProfile}.h5')
Hdf_inlet = HDF5File(MPI.comm_world, output_file_path, 'w')

if not InterpolateDataOverTime:

    for i, t in enumerate(obs_t_range):
        # Define a function object for velocity at each time step
        u_obs = Function(V)  # Create a Function in the space V for velocity
        u_obs.vector()[vertex_to_dof_map(V)] = vel_np[:, :, i].flatten()  # Map the velocity data (flattened) to the function object

        # commented-out lines to save the observation in an XDMF file or in HDF5 format.
        # xf_u_obs = XDMFFile(join(root, 'u_obs_{:02d}.xdmf'.format(f)))
        # xf_u_obs.write(project(u_obs, V))  # Project u_obs onto V

        # INCORRECT way to write the observation to the HDF5 file
        # Hdf = HDF5File(MPI.comm_world, join(outputDir, '{}/mesh_{}/uin_{:02d}.h5'.format(patient, dx, t)), 'w')
        # Hdf.write(u_obs, "u")
        # Hdf.close()

        # Correct way to write the observation to the HDF5 file, using the 'u_{:02d}' format for the dataset name
        dataset_name = "u_{:02d}".format(t)  # Format dataset name with the time step 't'
        Hdf_inlet.write(u_obs, dataset_name)

else:
    # Interpolate velocity data over time is TRUE
    vel_npinterp = np.zeros((nVertices, 3, t_range.shape[0]))  # Initialize a numpy array for interpolated velocity data

    for idx in tqdm(range(nVertices)):  # Loop over each vertex in the mesh
        # Interpolate u-component of velocity using quadratic interpolation over time
        f = interpolate.interp1d(list(obs_t_range), list(u_i[idx, :]), kind='quadratic')
        vel_npinterp[idx, 0, :] = f(t_range)

        # Interpolate v-component of velocity using quadratic interpolation over time
        f = interpolate.interp1d(list(obs_t_range), list(v_i[idx, :]), kind='quadratic')
        vel_npinterp[idx, 1, :] = f(t_range)

        # Interpolate w-component of velocity using quadratic interpolation over time
        f = interpolate.interp1d(list(obs_t_range), list(w_i[idx, :]), kind='quadratic')
        vel_npinterp[idx, 2, :] = f(t_range)

        #  In case of linear interpolation, not quadratic
        # vel_npinterp[idx, 0, :] = np.interp(list(t_range), list(obs_t_range), list(u_i[idx, :]))
        # vel_npinterp[idx, 1, :] = np.interp(list(t_range), list(obs_t_range), list(v_i[idx, :]))
        # vel_npinterp[idx, 2, :] = np.interp(list(t_range), list(obs_t_range), list(w_i[idx, :]))

    # Saving of the interpolated velocity data to VTP files for verification
    # for i, t in enumerate(t_range[1:]):
    #     inlet_pv = pv.wrap(mesh.coordinates())  # Wrap mesh coordinates into a PyVista object
    #     inlet_pv.point_data['velocity'] = vel_npinterp[:, :, i]  # Set the interpolated velocity data
    #     inlet_pv.save(join(outputDir, '{}/check/uin_{:.8f}.vtp'.format(patient, np.round(t, 8))))  # Save the VTP file for each time step

    # Initialize observation vector at time t=0
    t = 0
    u_in = Function(V, name='inlet')  # Create the observation function at time 0
    # The 'inlet' name is given to this function to represent the inlet velocity at the start

    if psProfile:

        # ---> Output the initial observation at current time `t` in XDMF format for checking
        xf_u_in = XDMFFile(join(outputDir, '{}/mesh/check/uin_{:.8f}.xdmf'.format(patient, np.round(t, 8))))
        xf_u_in.write(u_in)

        # Open a HDF5 file for writing, to store the inlet velocities at time `t`
        Hdf = HDF5File(MPI.comm_world, join(outputDir, '{}/mesh/ps/uin_{:.8f}.h5'.format(patient, np.round(t, 8))), 'w'))

        # Write the velocity data into the HDF5 file with the key 'u'
        Hdf_inlet.write(u_in, 'u', 0)

        for i, t in enumerate(t_range[1:]):
            # Loop over the time range (excluding the first time step)

            u_in.vector()[vertex_to_dof_map(V)] = vel_npinterp[:, :, i].flatten()  # Update observation vector with interpolated velocity

            if i % 42 == 0:
                # Every 42 time steps, save a check file for the observation velocity at time `t`
                xf_u_obs = XDMFFile(join(outputDir, '{}/mesh/check/uin_{:.8f}.xdmf'.format(patient, np.round(t, 8))))
                xf_u_obs.write(u_in)

            # Write the updated velocity field to the HDF5 file at index `i + 1`
            Hdf_inlet.write(u_in, 'u', i + 1)

    else:

        # ---> Output the initial observation at current time `t` in XDMF format for checking
        xf_u_in = XDMFFile(join(outputDir, '{}/mesh/check/uin_{:.8f}.xdmf'.format(patient, np.round(t, 8))))
        xf_u_in.write(u_in)

        # Open a new HDF5 file for writing the inlet velocities at time `t`
        Hdf = HDF5File(MPI.comm_world, join(outputDir, '{}/mesh/plug/uin_{:.8f}.h5'.format(patient, np.round(t, 8))), 'w'))
        Hdf.write(u_in, "u")  # Write the velocity data to HDF5 file
        Hdf.close()

        # ---> Define Inlet Plug from averaged velocity data taken from 4D flow measurement
        # This is an implementation based on the following reference:
        # https://fenicsproject.discourse.group/t/define-a-time-varying-parabolic-profile-normal-to-a-boundary/6041

        df = pd.read_csv(join(outputDir, '{}/mesh/uin.csv'.format(patient)), sep=',')  # Read the velocity data from CSV
        averagedvel = np.array(df.iloc[:]['Velocity'])  # Extract velocity values

        # Define a function to create an interpolation from the velocity data
        def makeIC():
            v = averagedvel
            return splrep(obs_t_range[1:], v)  # Return the spline interpolation

        # Approximate facet normal using projection
        n = FacetNormal(mesh)

        # Formulate problem for solving the normal velocity at the facet
        u_ = TrialFunction(V)
        v_ = TestFunction(V)
        a = inner(u_, v_) * ds  # Weak form of the problem
        l = inner(n, v_) * ds   # Linear term for boundary conditions
        A = assemble(a, keep_diagonal=True)  # Assemble the system
        L = assemble(l)

        A.ident_zeros()  # Ensure the matrix is correctly formatted
        nh = Function(V)  # Create a function to store normal velocity
        solve(A, nh.vector(), L)  # Solve the system for normal velocity

        # Define a UserExpression to model the boundary condition at the inlet
        class InflowBoundaryValue(UserExpression):
            def __init__(self, t=None, period=None, **kwargs):
                super().__init__(**kwargs)
                self.t = t
                self.t_p = period
                self.bc_func = makeIC()  # Set the interpolation function

            def eval(self, values, x):
                n_eval = nh(x)  # Evaluate the normal velocity at the boundary
                t = self.t
                val = splev(t - int(t / self.t_p) * self.t_p, self.bc_func)  # Evaluate the spline at time `t`
                values[0] = -n_eval[0] * val
                values[1] = -n_eval[1] * val
                values[2] = -n_eval[2] * val

            def value_shape(self):
                return (3,)  # Shape of the vector field (3 components)

        expr = InflowBoundaryValue(t=0, period=T)  # Create the expression for boundary velocity

        # Apply Dirichlet BC with the defined inlet condition
        inlet = DirichletBC(V, expr, facet_ids, inflow_id)

        for c, t in enumerate(t_range[1:]):
            expr.t = t  # Update the boundary condition for the current time
            t = np.round(t, 8)  # Round the time to 8 decimal places
            u_in = Function(V, name='inlet')  # Create the inlet function
            t0 = time.time()
            inlet.apply(u_in.vector())  # Apply the boundary condition to the function
            t1 = time.time()
            print(t1 - t0)  # Print the time taken to apply the boundary condition
            if c % 42 == 0:
                # Save check files every 42 time steps
                xf_u_obs = XDMFFile(join(outputDir, '{}/mesh/check/uin_{:.8f}.xdmf'.format(patient, np.round(t, 8))))
                xf_u_obs.write(u_in)
            Hdf = HDF5File(MPI.comm_world, join(outputDir, '{}/mesh/plug/uin_{:.8f}.h5'.format(patient, np.round(t, 8))), 'w'))
            Hdf.write(u_in, "u")  # Write the updated inlet velocity to HDF5
            Hdf.close()


# Close the inlet HDF5 file after writing all data
Hdf_inlet.close()

# Test the outflow boundary
n = FacetNormal(mesh)  # Compute the facet normal
u_ = TrialFunction(V)
v_ = TestFunction(V)
a = inner(u_, v_) * ds
l = inner(n, v_) * ds
A = assemble(a, keep_diagonal=True)
L = assemble(l)

A.ident_zeros()
nh = Function(V)

solve(A, nh.vector(), L)

# Define the outflow boundary condition using the facet data
ds = Measure('ds', domain=mesh, subdomain_data=facet_ids)
area = assemble(1 * ds(3))  # Calculate the area of the outflow boundary

class OutflowBoundaryValue(UserExpression):
    def __init__(self, val=None, **kwargs):
        super().__init__(**kwargs)
        self.val = val

    def eval(self, values, x):
        n_eval = nh(x)
        val = self.val
        values[0] = -n_eval[0] * val
        values[1] = -n_eval[1] * val
        values[2] = -n_eval[2] * val

    def value_shape(self):
        return (3,)  # Shape of the outflow velocity

u_in = Function(V, name='inlet')

expr = OutflowBoundaryValue(val=0)  # Set outflow value to 0 for outflow boundary

outlet = DirichletBC(V, expr, facet_ids, 3)  # Apply the outflow condition at the boundary with ID 3

if not psProfile:
    for c, t in enumerate(t_range[1:]):
        # Loop through the time steps (excluding the first)

        # Read the inlet velocity from HDF5
        Hdf = HDF5File(MPI.comm_world, join(outputDir, '{}/mesh/plug/uin_{:.8f}.h5'.format(patient, np.round(t, 8))), 'r'))
        Hdf.read(u_in, "u")  # Read the inlet velocity
        Hdf.close()

        # Compute the flux at the outflow boundary
        flux = dot(u_in, - n) * ds(2)
        Qin = assemble(flux)  # Assemble the flux value
        Qout = Qin / 2  # Divide by 2 to get the outflow
        val = Qout / area  # Normalize by the area of the outflow boundary

        expr.val = val  # Update the outflow value for the expression
        u_out = Function(V, name='outlet')  # Create the outlet function
        t0 = time.time()
        outlet.apply(u_out.vector())  # Apply the outlet boundary condition
        t1 = time.time()
        print(t1 - t0)  # Print the time taken to apply the boundary condition
        if c % 42 == 0:
            # Save check files for the outlet velocity every 42 time steps
            xf_u_obs = XDMFFile(join(outputDir, '{}/mesh/check/uout_{:.8f}.xdmf'.format(patient, np.round(t, 8))))
            xf_u_obs.write(u_out)
