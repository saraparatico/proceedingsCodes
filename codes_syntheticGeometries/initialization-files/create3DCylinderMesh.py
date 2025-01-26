import gmsh
gmsh.initialize()
gmsh.clear()

# Create Cylinder
gmsh.model.add("circle_extrusion")
center_x, center_y, center_z = 0., 0., 0.
radius = 30.
circle = gmsh.model.occ.addCircle(center_x, center_y, center_z, radius)
gmsh.model.occ.addCurveLoop([circle], 1000)
surface = gmsh.model.occ.addPlaneSurface([1000])
gmsh.model.occ.synchronize()

inlet_surf_group = gmsh.model.addPhysicalGroup(2, [surface], tag = 1)

l = 200
extrusion = gmsh.model.occ.extrude([(2, surface)], 0, 0, l)

gmsh.model.occ.synchronize()

fluid_volume = gmsh.model.addPhysicalGroup(3, [extrusion[1][1]])
wall_surf_group = gmsh.model.addPhysicalGroup(2, [extrusion[2][1]], tag = 2,)
outlet_surf_group = gmsh.model.addPhysicalGroup(2, [extrusion[0][1]], tag = 3)

gmsh.model.occ.synchronize()

#lcar1 = 10 # 20000 elementi
#lcar1 = 8 # 40000 elementi
lcar1 = 7

# Assign a mesh size to all the points:
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lcar1)

gmsh.model.mesh.generate(3)
gmsh.model.mesh.refine()

gmsh.write("mesh/cylinder3D_7.msh")

import meshio
msh = meshio.read("mesh/cylinder3D_7.msh")

def create_mesh(mesh, cell_type, name, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(points=mesh.points, cells={
                           cell_type: cells}, cell_data={name: [cell_data]})
    if prune_z:
        out_mesh.prune_z_0()
    return out_mesh


tetra_mesh = create_mesh(msh, "tetra", "bulk")
triangle_mesh = create_mesh(msh, "triangle", "exterior")
meshio.write("mesh/cylindermesh3D_7.xdmf", tetra_mesh)

meshio.write("mesh/cylindermesh3D_7_exterior.xdmf", triangle_mesh)
