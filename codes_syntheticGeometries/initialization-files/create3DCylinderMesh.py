################################################################################
############################ - create3DCylinderMesh.py - #######################
# This file is responsible for generating the 3D cylindrical mesh.
# It must be executed first (in 3D cases). 
################################################################################
################################################################################

import gmsh
gmsh.initialize()
gmsh.clear()

# ! CREATION OF A 3D MESH !
# Create a Cylinder
gmsh.model.add("circle_extrusion") # Start a new GMSH model named "circle_extrusion"
center_x, center_y, center_z = 0., 0., 0. # Set the center coordinates for the base circle
radius = 30.  # Define the radius of the circle (it's a parameter, you choose the value)
circle = gmsh.model.occ.addCircle(center_x, center_y, center_z, radius) # Create a circle

# Create a plane surface bounded by the circle
gmsh.model.occ.addCurveLoop([circle], 1000)  # Add a curve loop to enclose the circle
surface = gmsh.model.occ.addPlaneSurface([1000]) # Define the plane surface within the curve loop
gmsh.model.occ.synchronize()  # Synchronize the OpenCASCADE geometry with the GMSH model

# Define the inlet surface as a physical group (if necessary to be used for boundary conditions later)
inlet_surf_group = gmsh.model.addPhysicalGroup(2, [surface], tag = 1) # Tag the inlet surface as "1"

# Extrude the circle surface to form a cylinder
l = 200 # Length of the cylinder (it's a parameter, you choose the value)
extrusion = gmsh.model.occ.extrude([(2, surface)], 0, 0, l) # Extrude the surface along the Z-axis by `l`

gmsh.model.occ.synchronize()  # Synchronize the geometry after extrusion

# Assign physical groups for the cylinder
fluid_volume = gmsh.model.addPhysicalGroup(3, [extrusion[1][1]]) # Tag the fluid volume (3D) for the cylinder
wall_surf_group = gmsh.model.addPhysicalGroup(2, [extrusion[2][1]], tag = 2,) # Tag the cylinder wall as "2"
outlet_surf_group = gmsh.model.addPhysicalGroup(2, [extrusion[0][1]], tag = 3) # Tag the outlet surface as "3"

gmsh.model.occ.synchronize() # Finalize the geometry

# Mesh size settings for refinement
# lcar1 defines the characteristic length of the mesh elements
# Uncomment the appropriate line for the desired resolution
# lcar1 = 10  # Coarser mesh (~20,000 elements)
# lcar1 = 8   # Finer mesh (~40,000 elements)
lcar1 = 7 # Very fine mesh

# Assign a mesh size to all the points
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lcar1)

# Generate and refine the 3D mesh
gmsh.model.mesh.generate(3) # Generate a 3D mesh
gmsh.model.mesh.refine() # Refine the mesh for better quality

gmsh.write("mesh/cylinder3D_7.msh")

import meshio
msh = meshio.read("mesh/cylinder3D_7.msh")

# Function to extract mesh components based on cell type
def create_mesh(mesh, cell_type, name, prune_z=False):
    cells = mesh.get_cells_type(cell_type) # Get the mesh cells of the specified type
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type) # Get physical group data
    out_mesh = meshio.Mesh(points=mesh.points, cells={
                           cell_type: cells}, cell_data={name: [cell_data]})
    if prune_z:  # Optionally remove the Z-component for 2D meshes
        out_mesh.prune_z_0()
    return out_mesh

# Extract and save the bulk (tetrahedral) mesh
tetra_mesh = create_mesh(msh, "tetra", "bulk") # Extract tetrahedral elements
meshio.write("mesh/cylindermesh3D_7.xdmf", tetra_mesh)

# Extract and save the surface (triangular) mesh
triangle_mesh = create_mesh(msh, "triangle", "exterior") # Extract triangular elements
meshio.write("mesh/cylindermesh3D_7_exterior.xdmf", triangle_mesh)
