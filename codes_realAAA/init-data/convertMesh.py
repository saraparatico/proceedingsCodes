################################################################################
############################### - convertMesh.py - #############################
# This script converts the real AAA mesh from VTP and VTU files into a format
# suitable for further processing. It must be executed first in the workflow.
# !!! IMPORTANT: MODIFY ROW 47 TO SET YOUR OWN ROOT DIRECTORY !!!
################################################################################
################################################################################

from os.path import join
import pandas as pd
from dolfin import *
import vtk
import meshio
import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from tqdm import tqdm
import pyvista as pv

# Function to read a VTP file
def read_vtp(filename):
    reader = vtk.vtkXMLPolyDataReader()  # Initialize VTK reader for .vtp files
    reader.SetFileName(filename)  # Set the file to read
    reader.Update()
    return reader.GetOutput()

# Function to read a VTU file
def read_vtu(filename):
    reader = vtk.vtkXMLUnstructuredGridReader()  # Initialize VTK reader for .vtu files
    reader.SetFileName(filename)  # Set the file to read
    reader.Update()
    return reader.GetOutput()

# Function to apply a threshold filter to a VTK object
def threshold(pd, fieldName, lower, upper):
    thr = vtk.vtkThreshold()  # Create a threshold filter
    thr.SetInputData(pd)  # Set the input data
    thr.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS_THEN_CELLS, fieldName)  # Specify which field to threshold
    thr.ThresholdBetween(lower, upper)  # Set threshold range
    thr.Update()
    geo = vtk.vtkGeometryFilter()  # Convert thresholded data into geometry
    geo.SetInputData(thr.GetOutput())
    geo.Update()
    return geo.GetOutput()

##------------------------------------------
# Set the root directory for the project !!! SPECIFY YOUR OWN ROOT DIRECTORY!
root = '/home/biomech/sara/paratico-oasis/simple_forward_sara/tesi-Sara-Paratico'
##------------------------------------------

# Define the patient ID (Should be parameterized instead of hardcoded)
patient = 'AAA03'  # TODO: make it a user parameter

# Define file paths for the mesh and surface files
fn_mesh = join(root, f'init-data_prova/mesh/{patient}.vtu')  # Bulk mesh file
fn_surface = join(root, f'init-data_prova/mesh/{patient}.vtp')  # Surface mesh file
fn_surface_4dflow = join(root, f'init-data_prova/mesh/{patient}_4dflow.vtp')  # 4D Flow MRI surface mesh file

# Convert VTK mesh to XML format for FEniCS
mesh = meshio.read(fn_mesh)
meshio.write(join(root, f'init-data_prova/mesh/{patient}.xml'), mesh)

# Read surface data from VTP files
surface = read_vtp(fn_surface)
surface_4dflow = read_vtp(fn_surface_4dflow)

# Read face IDs from CSV file (which contains IDs for Wall, Inlet and Outlets)
df = pd.read_csv(join(root, f'init-data_prova/mesh/{patient}_faceids.csv'), sep=',')

# Extract Inlet and Wall IDs from the first row (original model)
inletID = df.iloc[0]['Inlet']
wallID = df.iloc[0]['Wall']

# Extract inlet and wall surfaces using thresholding
inlet = threshold(surface, 'ModelFaceID', inletID, inletID + 0.5)
wall = threshold(surface, 'ModelFaceID', wallID, wallID + 0.5)

# Extract outlet IDs and threshold each outlet
outletIDs = []
outlets = []
for i in range(2, df.iloc[0].shape[0]):  # Iterate through remaining columns (Outlets)
    outletID = df.iloc[0][f'Outlet_{i - 1}']  # Get outlet ID
    outletIDs.append(outletID)
    outlets.append(threshold(surface, 'ModelFaceID', outletID, outletID + 0.5))  # Apply threshold

# Read Inlet and Wall IDs for the 4D Flow model (from the second row of the CSV file)
inletID_4dflow = df.iloc[1]['Inlet']
wallID_4dflow = df.iloc[1]['Wall']

# Apply thresholding to extract surfaces from the 4D Flow model
inlet_4dflow = threshold(surface_4dflow, 'ModelFaceID', inletID_4dflow, inletID_4dflow + 0.5)
wall_4dflow = threshold(surface_4dflow, 'ModelFaceID', wallID_4dflow, wallID_4dflow + 0.5)

# Extract outlet IDs for the 4D Flow model and apply thresholding
outletIDs_4dflow = []
outlets_4dflow = []
for i in range(2, df.iloc[1].shape[0]):  # Iterate through outlets for the 4D Flow model
    outletID_4dflow = df.iloc[1][f'Outlet_{i - 1}']  # Get outlet ID
    outletIDs_4dflow.append(outletID_4dflow)
    outlets_4dflow.append(threshold(surface_4dflow, 'ModelFaceID', outletID_4dflow, outletID_4dflow + 0.5))  # Apply threshold

#==================================#
#==== Load mesh and boundaries ====#
#==================================#

#---> Read the entire mesh from an XML file
mesh = Mesh(join(root, f'init-data_prova/mesh/{patient}.xml'))
# Initialize mesh connectivity between cells and facets
mesh.init(mesh.topology().dim()-1, mesh.topology().dim())

def calc_mean_normal(pd):
    """
    Compute the mean normal of a given VTK polydata.
    """
    normalsFilter = vtk.vtkPolyDataNormals()  # Initialize the VTK normal computation filter
    normalsFilter.SetInputData(pd)  # Set input data
    normalsFilter.ComputePointNormalsOff()  # Disable point normals computation
    normalsFilter.ComputeCellNormalsOn()  # Enable cell normals computation
    normalsFilter.Update()  # Update the filter
    # Compute and return the mean normal vector
    mean_normal = np.mean(vtk_to_numpy(normalsFilter.GetOutput().GetCellData().GetArray('Normals')), 0)
    return mean_normal

#---> Read inlet STL file
inlet_pts = vtk_to_numpy(inlet.GetPoints().GetData())  # Convert VTK points to NumPy array
inlet_normal = calc_mean_normal(inlet)  # Compute inlet normal

#---> Read outlet STL files
outlets_pts = []  # List to store outlet points
outlets_normal = []  # List to store outlet normals
for outlet in outlets:
    outlets_pts.append(vtk_to_numpy(outlet.GetPoints().GetData()))  # Convert and store points
    outlets_normal.append(calc_mean_normal(outlet))  # Compute and store normal vectors

#====================================#
#==== Create boundary subdomains ====#
#====================================#

#---> Initialize all boundaries with id 0
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)  # Alternative to boundaries.set_all(0)

#---> Extract exterior facets
ext_facets = []
int_facets = []
for facet in tqdm(facets(mesh), total=mesh.num_facets()):  # Iterate over all mesh facets
    if facet.exterior():  # Check if the facet is on the exterior
        ext_facets.append(facet)
        boundaries.set_value(facet.index(), 1)  # Assign ID 1 to exterior facets

#======= INLET ========#
offset = 0.1  # Normal tolerance threshold
inlet_facets_candidates = []
for inlet_canditate in tqdm(ext_facets, total=len(ext_facets)):  # Iterate over exterior facets
    normal_misfit = np.linalg.norm(inlet_canditate.normal().array() - inlet_normal)  # Compute normal difference
    if normal_misfit < offset:  # Check if difference is within the threshold
        inlet_facets_candidates.append(inlet_canditate)

#---> Assign inlet ID = 2
c = 0  # Counter for assigned facets
for facet in tqdm(inlet_facets_candidates, total=len(inlet_facets_candidates)):  # Iterate over inlet candidates
    vertex_coords = []
    for vertex in vertices(facet):  # Iterate over vertices of the facet
        vertex_coords.append(list(vertex.point().array()))  # Append vertex coordinates
    vertex_coords = np.reshape(vertex_coords, (-1, mesh.topology().dim()))  # Reshape for comparison

    for i in range(inlet_pts.shape[0]):  # Iterate over inlet points
        if inlet_pts[i, :].tolist() in vertex_coords.tolist():  # Check if the point is in the facet
            boundaries.set_value(facet.index(), 2)  # Assign ID 2 to the inlet facet
            c += 1

#======= OUTLET ========#
offset = 1  # Normal tolerance threshold
outlets_facets_candidates = []
for outlet_normal in outlets_normal:  # Iterate over each outlet normal
    outlet_facets_candidates = []
    for outlet_canditate in tqdm(ext_facets, total=len(ext_facets)):  # Iterate over exterior facets
        normal_misfit = np.linalg.norm(outlet_canditate.normal().array() - outlet_normal)  # Compute normal difference
        if normal_misfit < offset:  # Check if difference is within the threshold
            outlet_facets_candidates.append(outlet_canditate)
    outlets_facets_candidates.append(outlet_facets_candidates)

#---> Assign outlet IDs (starting from 3)
for c, (outlet_facets_candidates, outlet_pts) in enumerate(zip(outlets_facets_candidates, outlets_pts)):
    for facet in tqdm(outlet_facets_candidates, total=len(outlet_facets_candidates)):  # Iterate over outlet candidates
        vertex_coords = []
        for vertex in vertices(facet):  # Iterate over vertices of the facet
            vertex_coords.append(list(vertex.point().array()))
        vertex_coords = np.reshape(vertex_coords, (-1, mesh.topology().dim()))  # Reshape for comparison

        for i in range(outlet_pts.shape[0]):  # Iterate over outlet points
            if outlet_pts[i, :].tolist() in vertex_coords.tolist():  # Check if the point is in the facet
                boundaries.set_value(facet.index(), 3 + c)  # Assign ID starting from 3 (3, 4, 5, ...)
                #print(facet.index())  # Debugging line (commented out)

XDMFFile(join(root, f'init-data_prova/mesh/boundaries.xdmf')).write(boundaries)

#====================================#
#==== Create cell subdomains ====#
#====================================#

# Initialize the bulk with ID 0
bulk = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)

#======= BULK ========#
# Define the cell index for the bulk region
offset = 1

bulk_cells_candidates = []
for cell in tqdm(cells(mesh), total=mesh.num_cells()):
    # Compute the center of each cell in the mesh
    coord_cell = cell.get_vertex_coordinates()
    center_cell_x = (coord_cell[0] + coord_cell[3] + coord_cell[6] + coord_cell[9]) / 4
    center_cell_y = (coord_cell[1] + coord_cell[4] + coord_cell[7] + coord_cell[10]) / 4
    center_cell_z = (coord_cell[2] + coord_cell[5] + coord_cell[8] + coord_cell[11]) / 4
    center_cell = [center_cell_x, center_cell_y, center_cell_z]

    # Check if the cell center is inside the 4D flow volume
    points = pv.PolyData(center_cell)
    surface_4dflow_pv = pv.wrap(surface_4dflow)
    select = points.select_enclosed_points(surface_4dflow_pv)

    if select['SelectedPoints']:
        # Compute distances from inlet and wall
        distance_inlet = vtk.vtkImplicitPolyDataDistance()
        distance_inlet.SetInput(inlet_4dflow)
        distance_wall = vtk.vtkImplicitPolyDataDistance()
        distance_wall.SetInput(wall_4dflow)

        # Check if the cell is sufficiently far from the boundaries
        if np.linalg.norm(distance_wall.EvaluateFunction(center_cell)) > offset and np.linalg.norm(
                distance_inlet.EvaluateFunction(center_cell)) > offset:
            bulk.set_value(cell.index(), 1)
    else:
        print('Found point not inside')

XDMFFile(join(root, f'init-data_prova/mesh/boundaries.xdmf')).write(boundaries)
XDMFFile(join(root, f'init-data_prova/mesh/bulk.xdmf')).write(bulk)

hdf = HDF5File(MPI.comm_world, join(root, f"init-data_prova/mesh/{patient}.h5"), "w")
hdf.write(mesh, "/mesh")
hdf.write(boundaries, "/boundaries")
hdf.write(bulk, "/bulk")
hdf.close()

'''
#====================================#
#==== Check the results ====#
#====================================#

# Load the saved mesh data
f = HDF5File(MPI.comm_world, join(root, f"mesh/mesh_2.2/{patient}.h5"), 'r')
mesh = Mesh()
f.read(mesh, "mesh", False)
cell_ids = MeshFunction("size_t", mesh, mesh.topology().dim())
f.read(cell_ids, "/bulk")
facet_ids = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
f.read(facet_ids, "/boundaries")
f.close()

# Initialize lists to store cell coordinates based on classification
cell_id0 = []  # Cells outside the bulk
cell_id1 = []  # Cells inside the bulk

i = 0
for cell in cells(mesh):
    coord_cell = cell.get_vertex_coordinates()
    center_cell_x = (coord_cell[0] + coord_cell[3] + coord_cell[6] + coord_cell[9]) / 4
    center_cell_y = (coord_cell[1] + coord_cell[4] + coord_cell[7] + coord_cell[10]) / 4
    center_cell_z = (coord_cell[2] + coord_cell[5] + coord_cell[8] + coord_cell[11]) / 4
    center_cell = [center_cell_x, center_cell_y, center_cell_z]

    if cell_ids.array()[i] == 1:
        cell_id1.append(center_cell)  # Add to bulk
    else:
        cell_id0.append(center_cell)  # Add to non-bulk
    i += 1

# Convert lists to NumPy arrays
cell_id0 = np.array(cell_id0)
cell_id1 = np.array(cell_id1)

# Plot the classified cells
p = pv.Plotter()
p.add_points(cell_id0, color='blue', opacity=0.8, point_size=5)
p.add_points(cell_id1, color='red', opacity=0.8, point_size=5)
p.show()
'''
