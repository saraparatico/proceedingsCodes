from os.path import join
import pandas as pd
from dolfin import *
import vtk
import meshio
import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from tqdm import tqdm
import pyvista as pv

def read_vtp(filename):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def read_vtu(filename):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def threshold(pd, fieldName, lower, upper):
    thr = vtk.vtkThreshold()
    thr.SetInputData(pd)
    thr.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS_THEN_CELLS, fieldName)
    thr.ThresholdBetween(lower, upper)
    thr.Update()
    geo = vtk.vtkGeometryFilter()
    geo.SetInputData(thr.GetOutput())
    geo.Update()
    return geo.GetOutput()

##------------------------------------------
#root = '../femda/data
root = '/home/biomech/sara/paratico-oasis/simple_forward_sara/tesi-Sara-Paratico'
##------------------------------------------p

#---> Get patient
patient = 'AAA03' #!todo make it user parameter
#dx = '220' ## DO NOT NEED dx

#---> Get surface and bulk mesh in vtk fromat ## DO NOT NEED dx
#fn_mesh = join(root, f'mesh/mesh_{dx}/{patient}.vtu')
#fn_surface = join(root, f'mesh/mesh_{dx}/{patient}.vtp')
#fn_surface_4dflow = join(root, f'mesh/mesh_{dx}/{patient}_4dflow.vtp')
fn_mesh = join(root, f'init-data_prova/mesh/{patient}.vtu')
fn_surface = join(root, f'init-data_prova/mesh/{patient}.vtp')
fn_surface_4dflow = join(root, f'init-data_prova/mesh/{patient}_4dflow.vtp')

mesh = meshio.read(fn_mesh)
meshio.write(join(root, f'init-data_prova/mesh/{patient}.xml'), mesh)

surface = read_vtp(fn_surface)
surface_4dflow = read_vtp(fn_surface_4dflow)

#---> read model ids from csv file [Wall, Inlet, Outlet, ...]
df = pd.read_csv(join(root, f'init-data_prova/mesh/{patient}_faceids.csv'), sep=',')

inletID = df.iloc[0]['Inlet']
wallID = df.iloc[0]['Wall']
inlet = threshold(surface, 'ModelFaceID', inletID, inletID + 0.5)
wall = threshold(surface, 'ModelFaceID', wallID, wallID + 0.5)
outletIDs = []
outlets = []

for i in range(2, df.iloc[0].shape[0]):
    outletID = df.iloc[0][f'Outlet_{i - 1}']
    outletIDs.append(outletID)
    outlets.append(threshold(surface, 'ModelFaceID', outletID, outletID + 0.5))

#---> read origina model ids from csv file [Wall, Inlet, Outlet, ...]

inletID_4dflow = df.iloc[1]['Inlet']
wallID_4dflow = df.iloc[1]['Wall']
inlet_4dflow = threshold(surface_4dflow, 'ModelFaceID', inletID, inletID_4dflow + 0.5)
wall_4dflow = threshold(surface_4dflow, 'ModelFaceID', wallID, wallID_4dflow + 0.5)
outletIDs_4dflow = []
outlets_4dflow = []

for i in range(2, df.iloc[1].shape[0]):
    outletID_4dflow = df.iloc[1][f'Outlet_{i - 1}']
    outletIDs_4dflow.append(outletID_4dflow)
    outlets_4dflow.append(threshold(surface_4dflow, 'ModelFaceID', outletID_4dflow, outletID_4dflow + 0.5))

#==================================#
#==== Load mesh and boundaries ====#
#==================================#

#--->Read entire mesh
mesh = Mesh(join(root, f'init-data_prova/mesh/{patient}.xml'))
mesh.init(mesh.topology().dim()-1, mesh.topology().dim()) #connectivity tra celle e facets

def calc_mean_normal(pd):
    normalsFilter = vtk.vtkPolyDataNormals()
    normalsFilter.SetInputData(pd)
    normalsFilter.ComputePointNormalsOff()
    normalsFilter.ComputeCellNormalsOn()
    normalsFilter.Update()
    mean_normal = np.mean(vtk_to_numpy(normalsFilter.GetOutput().GetCellData().GetArray('Normals')), 0)
    return mean_normal

#--->Read inlet stl
inlet_pts = vtk_to_numpy(inlet.GetPoints().GetData())
inlet_normal = calc_mean_normal(inlet)

#--->Read outlet stl
outlets_pts = []
outlets_normal = []
for outlet in outlets:
    outlets_pts.append(vtk_to_numpy(outlet.GetPoints().GetData()))
    outlets_normal.append(calc_mean_normal(outlet))

#====================================#
#==== Create boundary subdomains ====#
#====================================#

#--->Initialize all boundaries with id 0
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0) # this index 0 is an alternative to the command boundaries.set_all(0)

#--->Extract exterior facets
ext_facets = []
int_facets = []
for facet in tqdm(facets(mesh), total=mesh.num_facets()):
    if facet.exterior():
        ext_facets.append(facet)
        boundaries.set_value(facet.index(), 1)

#======= INLET ========#
#--->Extract facets with similar normal to inlet
offset = 0.1
inlet_facets_candidates = []
for inlet_canditate in tqdm(ext_facets, total=len(ext_facets)):
    normal_misfit = np.linalg.norm(inlet_canditate.normal().array() - inlet_normal)
    if normal_misfit < offset:
        inlet_facets_candidates.append(inlet_canditate)

#--->Assign inlet id = 1
c = 0
for facet in tqdm(inlet_facets_candidates, total=len(inlet_facets_candidates)):
    vertex_coords = []
    for vertex in vertices(facet):
        vertex_coords.append(list(vertex.point().array()))
    vertex_coords = np.reshape(vertex_coords, (-1, mesh.topology().dim()))

    for i in range(inlet_pts.shape[0]):
        if inlet_pts[i, :].tolist() in vertex_coords.tolist():
            boundaries.set_value(facet.index(), 2)
            c += 1

#======= OUTLET ========#
#--->Extract facets with similar normal to inlet
offset = 1
outlets_facets_candidates = []
for outlet_normal in outlets_normal:
    outlet_facets_candidates = []
    for outlet_canditate in tqdm(ext_facets, total=len(ext_facets)):
        normal_misfit = np.linalg.norm(outlet_canditate.normal().array() - outlet_normal)
        if normal_misfit < offset:
            outlet_facets_candidates.append(outlet_canditate)
    outlets_facets_candidates.append(outlet_facets_candidates)

#--->Assign outlet id = 3, ...
for c, (outlet_facets_candidates, outlet_pts) in enumerate(zip(outlets_facets_candidates, outlets_pts)):
    for facet in tqdm(outlet_facets_candidates, total=len(outlet_facets_candidates)):
        vertex_coords = []
        for vertex in vertices(facet):
            vertex_coords.append(list(vertex.point().array()))
        vertex_coords = np.reshape(vertex_coords, (-1, mesh.topology().dim()))

        for i in range(outlet_pts.shape[0]):
            if outlet_pts[i, :].tolist() in vertex_coords.tolist():
                boundaries.set_value(facet.index(), 3 + c)
                #print(facet.index())

XDMFFile(join(root, f'init-data_prova/mesh/boundaries.xdmf')).write(boundaries)

#====================================#
#==== Create cell subdomains ====#
#====================================#

# Initialize all bulk with id 0
bulk = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)

#======= BULK ========#
#---> set cell index for the bulk region
offset = 1 #distance from the boundaries (mm)

bulk_cells_candididates = []
for cell in tqdm(cells(mesh), total=mesh.num_cells()):
    #---> Compute the center cell for each cell of the mesh
    coord_cell = cell.get_vertex_coordinates()
    center_cell_x = (coord_cell[0] + coord_cell[3] + coord_cell[6] + coord_cell[9])/4
    center_cell_y = (coord_cell[1] + coord_cell[4] + coord_cell[7] + coord_cell[10])/4
    center_cell_z = (coord_cell[2] + coord_cell[5] + coord_cell[8] + coord_cell[11])/4
    center_cell = [center_cell_x, center_cell_y, center_cell_z]

    #---> Check if center cell is in the 4D flow volume
    points = pv.PolyData(center_cell)
    surface_4dflow_pv = pv.wrap(surface_4dflow)
    select = points.select_enclosed_points(surface_4dflow_pv)

    if select['SelectedPoints']:

        distance_inlet = vtk.vtkImplicitPolyDataDistance()
        distance_inlet.SetInput(inlet_4dflow)
        distance_wall = vtk.vtkImplicitPolyDataDistance()
        distance_wall.SetInput(wall_4dflow)

        #for outlet_4dflow in outlets_4dflow:

            #distance_outlet = vtk.vtkImplicitPolyDataDistance()
            #distance_outlet.SetInput(outlet_4dflow)

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
#XDMFFile(join(root, 'boundaries.xdmf')).write(boundaries)
'''
#Check

f = HDF5File(MPI.comm_world, join(root, f"mesh/mesh_2.2/{patient}.h5"), 'r')
mesh = Mesh()
f.read(mesh, "mesh", False)
cell_ids = MeshFunction("size_t", mesh, mesh.topology().dim())
f.read(cell_ids, "/bulk")
facet_ids = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
f.read(facet_ids, "/boundaries")
f.close()


cell_id0 = []
cell_id1 = []

i = 0
for cell in cells(mesh):
    if cell_ids.array()[i] == 1:
        coord_cell = cell.get_vertex_coordinates()
        center_cell_x = (coord_cell[0] + coord_cell[3] + coord_cell[6] + coord_cell[9]) / 4
        center_cell_y = (coord_cell[1] + coord_cell[4] + coord_cell[7] + coord_cell[10]) / 4
        center_cell_z = (coord_cell[2] + coord_cell[5] + coord_cell[8] + coord_cell[11]) / 4
        center_cell = [center_cell_x, center_cell_y, center_cell_z]
        cell_id1.append(center_cell)
    else:
        coord_cell = cell.get_vertex_coordinates()
        center_cell_x = (coord_cell[0] + coord_cell[3] + coord_cell[6] + coord_cell[9]) / 4
        center_cell_y = (coord_cell[1] + coord_cell[4] + coord_cell[7] + coord_cell[10]) / 4
        center_cell_z = (coord_cell[2] + coord_cell[5] + coord_cell[8] + coord_cell[11]) / 4
        center_cell = [center_cell_x, center_cell_y, center_cell_z]
        cell_id0.append(center_cell)
    i += 1

cell_id0 = np.array(cell_id0)
cell_id1 = np.array(cell_id1)

p = pv.Plotter()
p.add_points(cell_id0, color='blue', opacity=0.8, point_size=5)
p.add_points(cell_id1, color='red', opacity=0.8, point_size=5)
p.show()
'''