from fenics import *
from mshr import *

# Create mesh
channel = Rectangle(Point(0, 0), Point(200, 41))
mesh = generate_mesh(channel, 64)
XDMFFile("./mesh/mesh2D.xdmf").write(mesh)