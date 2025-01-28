from fenics import *
from mshr import *

# ! CREATION OF A 2D MESH !
# Define a rectangle with corners at (0, 0) and (200, 41)
# --> you can decide to choose other values for corners
channel = Rectangle(Point(0, 0), Point(200, 41))
# Generate a triangular mesh for the defined geometry with a resolution of 64
mesh = generate_mesh(channel, 64)
XDMFFile("./mesh/mesh2D.xdmf").write(mesh)
