################################################################################
################################# - solvers.py - ###############################
# This file is not intended to be run directly, as it serves as a helper for
# the main script. It implements the solvers used to perform simulations.
################################################################################
################################################################################

#-----------------------------# IMPORT LIBRARIES #-----------------------------#
import time
from dolfin import *
from dolfin_adjoint import *
import numpy as np
from utils import StationaryBoundaryFunction3D

# -----------------------------# COUPLED SCHEME #-----------------------------#
class Coupled(object):
    def __init__(self, problem):
        self.problem = problem
        self.dx = Measure('dx', domain=problem.mesh, subdomain_data=problem.cell_ids)  # Defines the measure for volume integration over the mesh
        self.ds = Measure('ds', domain=problem.mesh, subdomain_data=problem.facet_marker)  # Defines the measure for surface integration over the facets

        # If the control type is 'inlet', set the inlet boundary condition
        if self.problem.control == 'inlet':
            self.dsu_ = self.ds(self.problem.inflow_id)  # Subset of the domain for the inlet

    # NONLINEAR SOLVER
    def nonlinear_run(self, T, dt, velocity_space, u_h, U_inlet, rho, nu, meshD, weakControl, nitsche_gamma, beta, theta, backflow,
                      tractionFree, simvascularstab, stationary):

        dx = self.dx
        ds = self.ds
        dsu_ = self.dsu_
        mesh = self.problem.mesh
        t_range = np.arange(0, T, dt)  # Time range from 0 to T with time step dt
        dim = mesh.topology().dim()  # Get the dimension of the mesh (2D or 3D)
        zero = (0,) * dim  # Zero vector for boundary conditions

        # Fluid parameters
        nu = Constant(nu, name="kinematic viscosity")
        rho = Constant(rho, name="density")
        weakControl = weakControl
        meshD = meshD

        # Stabilization parameters for the problem
        beta = Constant(beta, name="beta")
        theta = Constant(theta, name="theta")
        nitsche_gamma = Constant(nitsche_gamma, name="nitsche gamma")

        # Define function spaces for velocity, pressure, and control
        V = velocity_space
        Q = self.problem.Q
        W = self.problem.W
        control_space = V

        # Define trial and test functions for the velocity and pressure
        (u, p) = TrialFunctions(W)
        (v, q) = TestFunctions(W)
        U = Function(W, name='Solution', annotate=True)
        p_h = Function(Q, name='Pressure', annotate=True)
        n = FacetNormal(mesh)
        u, p = split(U)  # Split the solution into velocity and pressure components

        # Project the inlet expression onto the velocity function object
        g = project(U_inlet, V)

        # Define the boundary conditions based on mesh dimension (2D or 3D)
        if meshD == '2D':
            inflow = 'near(x[0], 0)'  # Inflow condition on the left boundary
            outflow = 'near(x[0], 200)'  # Outflow condition on the right boundary
            walls = 'near(x[1], 0) || near(x[1], 41)'  # Walls at the top and bottom

            # Boundary conditions for velocity and pressure
            if not weakControl:
                if self.problem.control == 'inlet':
                    noslip = DirichletBC(W.sub(0), zero, walls, method="pointwise")
                    inlet = DirichletBC(W.sub(0), g, inflow, method="pointwise")
                    bcu = [inlet, noslip]
                    bcp = []
                    if not tractionFree:
                        outlet = DirichletBC(W.sub(1), 0, outflow, method="pointwise")
                        bcp = [outlet]

        elif meshD == '3D':
            # For 3D meshes, boundary conditions are set using facet markers
            facet_marker = self.problem.facet_marker
            inflow = 1
            outflow = 3
            walls = 2

            if not weakControl:
                if self.problem.control == 'inlet':
                    bc_inlet = DirichletBC(W.sub(0), g, facet_marker, inflow)
                    bc_walls = DirichletBC(W.sub(0), zero, facet_marker, walls)
                    bcu = [bc_inlet, bc_walls]
                    bcp = []
                    if not tractionFree:
                        bcp = DirichletBC(W.sub(1), Constant(0), facet_marker, outflow)

        # Define theta scheme for velocity update
        u_theta = theta * u + (1 - theta) * u_h

        # Define strain rate tensor
        def epsilon(u):
            return sym(nabla_grad(u))

        # Define stress tensor
        def sigma(u, p):
            return 2 * nu * epsilon(u) - p * Identity(len(u))

        # Define the variational formulation for the nonlinear system
        F = (
            inner((u - u_h) / Constant(dt), v) * dx  # Transient term
            + inner(grad(u_theta) * u_theta, v) * dx()  # Convective term
            + nu * inner(grad(u_theta), grad(v)) * dx  # Diffusion term
            - inner(p, div(v)) * dx  # Pressure term
            - inner(q, div(u)) * dx  # Continuity term
        )

        # Apply Neumann boundary condition for traction-free surface
        if tractionFree:
            T = Constant((0,) * dim)
            F -= dot(T, v) * ds()  # Neumann BC for traction-free surface

        # Stabilization term for lower-order elements
        if V.ufl_element().degree() == 1:
            h = CellDiameter(mesh)
            F -= 0.001 * h ** 2 * inner(grad(p), grad(q)) * dx

        # Backflow stabilization term
        if backflow:
            bf = (inner(n, u_h) - abs(inner(n, u_h))) / 2
            b = Constant(0.2)
            if simvascularstab:
                F -= b * inner(bf * u_theta, v) * ds
            else:
                def t_grad(w, V):
                    # Tangential gradient for backflow stabilization
                    S = V
                    n = FacetNormal(S.mesh())
                    if len(w.ufl_shape) == 0:
                        return grad(w) - n * inner(n, w)
                    elif len(w.ufl_shape) == 1:
                        return grad(w) - outer(grad(w) * n, n)

                F -= b * inner(bf * t_grad(u_theta, V), t_grad(v, V)) * ds

        # Define the iterative solver for the nonlinear problem
        iterative_solver = True
        yield t, u_h, p_h, g  # Yield initial condition

        # Loop through time steps to solve the nonlinear problem
        for t in t_range[1:]:
            t = np.round(t, 8)

            # Update the inlet condition for each time step if not stationary
            if not stationary:
                U_inlet.t = t
                g = project(U_inlet, V)

            # Define and solve the nonlinear variational problem
            problem = NonlinearVariationalProblem(F, U, bcu + bcp, derivative(F, U))
            solver = NonlinearVariationalSolver(problem)

            # Set solver parameters for Newton's method
            prm = solver.parameters
            prm['newton_solver']['absolute_tolerance'] = 1E-8
            prm['newton_solver']['relative_tolerance'] = 1E-7
            prm['newton_solver']['maximum_iterations'] = 20
            prm['newton_solver']['error_on_nonconvergence'] = False
            prm['newton_solver']['relaxation_parameter'] = 1.0

            if iterative_solver:
                prm['newton_solver']['linear_solver'] = 'bicgstab'  # Choose linear solver
                prm['newton_solver']['preconditioner'] = 'hypre_amg'

            # Solve the system
            timer = Timer("Forward solve")
            solver.solve()
            timer.stop()

            # Update the previous solution
            u, p = U.split(deepcopy=True)
            u_h.assign(u, annotate=True)
            p_h.assign(p, annotate=True)

            # Print Courant number for stability check
            print("Worst possible Courant number=", (dt * (u_h.vector().max())) / mesh.hmin())

            yield t, u_h, p_h, g

      # LINEAR SOLVER
      def linear_run(self, T, dt, velocity_space, u_h, g, rho, nu, meshD, nitsche_gamma, beta, theta, weakControl, tractionFree, backflow, simvascularstab, stationary):
        mesh = self.problem.mesh

        t_range = np.arange(0, T, dt)
        dt = t_range[1] - t_range[0]

        # ---> Model parameters
        theta = Constant(theta, name="theta")  # SemiImplicit scheme
        nu = Constant(nu, name="kinematic viscosity")  # Kinematic viscosity
        rho = Constant(rho, name="density")  # Density
        weakControl = weakControl
        meshD = meshD

        # ---> Stabilization parameters
        beta = Constant(beta, name="beta")
        nitsche_gamma = Constant(nitsche_gamma, name="nitsche gamma")

        # Defining function space for velocity and pressure
        V = velocity_space
        Q = self.problem.Q
        W = self.problem.W
        control_space = V

        # ---> Define trial and test functions for the variational formulation
        (u, p) = TrialFunctions(W)  # Trial functions for velocity and pressure
        (v, q) = TestFunctions(W)   # Test functions for velocity and pressure
        U = Function(W, name='Solution')  # Solution function (velocity and pressure)
        p_h = Function(Q, name='pressure')  # Pressure field
        n = FacetNormal(mesh)  # Normal vector on facets

        # Integration over the domain
        dx = self.dx
        ds = self.ds

        # ---> Boundary conditions initialization
        dim = mesh.topology().dim()  # Dimension of the mesh
        zero = Constant((0,) * dim)  # Zero vector (used for no-slip boundary)
        f = Constant((0,) * dim)  # Source term for body force

        # Function for control variable
        g_theta = Function(control_space, name='g')
        if stationary:
            g_theta.assign(g)  # Assign g to the function if stationary

        # ---> Mesh dimension specific conditions
        if meshD == '2D':
            facet_marker = self.problem.facet_marker
            walls = "on_boundary && (x[1] >= 39.9 || x[1] < 0.1)"  # Walls in 2D mesh
            inflow = "on_boundary && x[0] <= 0.1"  # Inlet boundary
            outflow = "on_boundary && x[0] >= 199.9"  # Outlet boundary

            # Boundary conditions for 2D mesh
            if not weakControl:
                noslip = DirichletBC(W.sub(0), zero, walls)  # No-slip condition at walls
                inlet = DirichletBC(W.sub(0), g_theta, inflow)  # Inlet condition with prescribed velocity
                bcu = [inlet, noslip]
                bcp = []
                if not tractionFree:
                    bcp = DirichletBC(W.sub(1), 0, outflow)  # Pressure at outlet is zero if traction-free is False
                    bcp = [bcp]

        elif meshD == '3D':
            facet_marker = self.problem.facet_marker
            inflow = 1  # Inflow boundary marker in 3D
            outflow = 3  # Outflow boundary marker in 3D
            walls = 2  # Walls boundary marker in 3D

            # Boundary conditions for 3D mesh
            if not weakControl:
                bc_inlet = DirichletBC(V, g_theta, facet_marker, inflow)  # Inlet boundary for velocity
                bc_walls = DirichletBC(V, Constant((0, 0, 0)), facet_marker, walls)  # No-slip on walls
                bcs_u = [bc_inlet, bc_walls]
                bc_p = DirichletBC(Q, Constant(0), facet_marker, outflow)  # Zero pressure at outflow

        # ---> Define time-stepping scheme
        # Linear interpolation between previous and current velocities
        u_theta = theta * u + (1 - theta) * u_h

        # ---> Define strain-rate tensor
        def epsilon(u):
            return sym(nabla_grad(u))  # Symmetric gradient for strain-rate tensor

        # ---> Define stress tensor
        def sigma(u, p):
            return 2 * nu * epsilon(u) - p * Identity(len(u))  # Stress tensor

        # ---> Variational formulation (linear terms)
        F = (
            inner((u - u_h) / Constant(dt), v) * dx  # Transient term (time derivative)
            + inner(grad(u_theta) * u_h, v) * dx  # Convective term (linear term)
            + nu * inner(grad(u_theta), grad(v)) * dx  # Stress tensor term (viscous part)
            - inner(p, div(v)) * dx  # Pressure term in the formulation
            - inner(q, div(u_theta)) * dx  # Divergence-free condition for incompressibility
        )

        # ---> Neumann boundary condition (for traction-free boundary)
        if tractionFree:
            T = Constant((0,) * dim)  # Zero traction vector
            F -= dot(T, v) * ds()  # Apply Neumann condition (traction-free)

        # ---> Add stabilisation term for P1-P1 element (used for stabilization in finite element method)
        if V.ufl_element().degree() == 1:
            h = CellDiameter(mesh)  # Cell diameter for stabilization
            # Brezzi-Pitkaranta stabilization
            F -=  0.001 * h ** 2 * inner(grad(p), grad(q)) * dx  # Stabilization term

        # ---> Backflow stabilization (used for preventing backflow-related issues)
        if backflow:
            bf = (inner(n, u_h) - abs(inner(n, u_h))) / 2  # Backflow indicator
            b = Constant(0.2)  # Stabilization constant
            if simvascularstab:
                # Stabilization term specific for SimVascular
                F -= b * inner(bf * u_theta, v) * ds
            else:
                # Tangential gradient term for stabilization
                def t_grad(w, V):
                    # Returns the tangential gradient on cell facets
                    S = V
                    n = FacetNormal(S.mesh())
                    if len(w.ufl_shape) == 0:
                        return grad(w) - n * inner(n, w)
                    elif len(w.ufl_shape) == 1:
                        return grad(w) - outer(grad(w) * n, n)

                F -= b * inner(bf * t_grad(u_theta, V), t_grad(v, V)) * ds  # Apply backflow stabilization

        # ---> Define left-hand side and right-hand side for the linear system
        a = lhs(F)  # Left-hand side (bilinear form)
        b = rhs(F)  # Right-hand side (linear form)

        # Initial condition (yielding initial time, velocity, and pressure)
        yield t_range[0], u_h, p_h, g

        # ---> Time-stepping loop
        for t in t_range[1:]:
            t = np.round(t, 8)  # Round time to avoid floating-point issues

            # Create a linear variational problem for the solver
            problem = LinearVariationalProblem(a, b, U, bcu + bcp)
            solver = LinearVariationalSolver(problem)
            prm = solver.parameters
            prm["linear_solver"] = 'bicgstab'  # BiCGStab solver (iterative solver)
            prm["preconditioner"] = 'hypre_amg'  # AMG preconditioner (Algebraic Multigrid)
            prm["krylov_solver"]["error_on_nonconvergence"] = True  # Ensure error is raised on non-convergence
            prm["krylov_solver"]["nonzero_initial_guess"] = True  # Start with nonzero initial guess
            prm["krylov_solver"]["absolute_tolerance"] = 1E-7  # Absolute tolerance for convergence
            prm["krylov_solver"]["relative_tolerance"] = 1E-4  # Relative tolerance for convergence
            prm["krylov_solver"]["maximum_iterations"] = 1000  # Maximum number of iterations

            # Timer for performance monitoring
            timer = Timer("Forward solve")

            # Solve the linear problem
            solver.solve()
            timer.stop()

            # Update previous solution
            u, p = U.split(deepcopy=True)  # Split solution into velocity and pressure
            u_h.assign(u, annotate=True)  # Update velocity field
            p_h.assign(p)  # Update pressure field

            yield t, u_h, p_h, g_theta

# -----------------------------# INCREMENTAL PRESSURE CORRECTION SPLITTING SCHEME #-----------------------------#

class IPCS(object):
    def __init__(self, problem):
        """
        Initialize the IPCS object with the problem context.
        """
        self.problem = problem
        self.dx = Measure('dx', domain=problem.mesh, subdomain_data=problem.cell_ids)
        self.ds = Measure('ds', domain=problem.mesh, subdomain_data=problem.facet_marker)

        # If the control is 'inlet', assign a specific boundary for inflow.
        if self.problem.control == 'inlet':
            self.dsu_ = self.ds(self.problem.inflow_id)

    def runIPCS(self, t_range, g, nu, rho, BDF2, stationary, dt, weakControl, meshD):
        """
        Main IPCS function to run the simulation.
        """
        mesh = self.problem.mesh
        dim = mesh.topology().dim()  # Dimension of the mesh
        zero = Constant((0,) * dim)  # Zero vector
        f = Constant((0,) * dim)     # Source term

        # -----------------------------# MODEL PARAMS #-----------------------------#
        t = 0  # Initialize time step
        nu = Constant(nu, name="kinematic viscosity")  # Kinematic viscosity
        rho = Constant(rho, name="density")  # Density
        mu = Constant(rho * nu, name="dynamic viscosity")  # Dynamic viscosity

        # -----------------------------# DEFINE FUNCTION SPACES (TAYLOR-HOOD ELEMENT) #-----------------------------#
        V = self.problem.V  # Velocity space
        Q = self.problem.Q  # Pressure space

        # -----------------------------# SET-UP FUNCTIONS #-----------------------------#
        u_h = Function(V, name='velocity')  # Current velocity
        u_s = Function(V)  # Tentative velocity
        u_old = Function(V)  # Previous velocity

        p_h = Function(Q, name='pressure')  # Current pressure
        p_c = Function(Q)  # Pressure correction

        dt_c = Constant(dt)  # Time step size

        # -----------------------------# SET-UP FOR STEP 1: COMPUTING TENTATIVE VELOCITY #-----------------------------#
        u = TrialFunction(V)
        v = TestFunction(V)

        # Convection and diffusion terms for velocity computation
        n = FacetNormal(mesh)
        h = 2.0 * Circumradius(mesh)

        weight_time = Constant(3 / (2 * dt_c)) if BDF2 else Constant(1 / dt_c)
        weight_diffusion = nu if BDF2 else Constant(0.5) * nu

        a1 = (weight_time * inner(u, v) + weight_diffusion * inner(grad(u), grad(v))) * dx
        L1 = (inner(p_h, div(v)) + inner(f, v)) * dx

        # BDF2 implementation (Backward Differentiation Formula)
        if BDF2:
            L1 += (1 / (2 * dt_c) * inner(4 * u_h - u_old, v)
                   - 2 * inner(grad(u_h) * u_h, v)
                   + inner(grad(u_old) * u_old, v)) * dx
        else:
            u_AB = Constant(1.5) * u_h - Constant(0.5) * u_old
            a1 += Constant(0.5) * inner(grad(u) * u_AB, v) * dx
            L1 += (weight_time * inner(u_h, v)
                   - weight_diffusion * inner(grad(u_h), grad(v))
                   - 0.5 * inner(grad(u_h) * u_AB, v)) * dx

        # -----------------------------# BACKFLOW STABILIZATION #-----------------------------#
        # Backflow stabilization term to avoid negative velocities
        bf = (inner(n, u_h) - abs(inner(n, u_h))) / 2
        b = Constant(0.2)
        # Apply stabilization terms
        a1 -= b * Constant(0.5) * inner(bf * u, v) * ds
        L1 += b * Constant(0.5) * inner(bf * u_h, v) * ds

        def t_grad(w, V):
            """
            Compute the tangential part of the gradient on the facets of the mesh.
            """
            S = V
            n = FacetNormal(S.mesh())
            if len(w.ufl_shape) == 0:
                return grad(w) - n * inner(n, w)
            elif len(w.ufl_shape) == 1:
                return grad(w) - outer(grad(w) * n, n)

        # -----------------------------# DEFINE BOUNDARY CONDITIONS #-----------------------------#
        g_theta = Function(V, name='g')  # Boundary condition function for velocity
        if stationary:
            g_theta.assign(g)

        # Define boundary conditions based on mesh dimension (2D or 3D)
        if meshD == '2D':
            facet_marker = self.problem.facet_marker
            inflow = 1
            outflow = 3
            walls = 2
            walls = "on_boundary && (x[1] >= 39.9 || x[1] < 0.1)"
            inflow = "on_boundary && x[0] <= 0.1"
            outflow = "on_boundary && x[0] >= 199.9"

            if not weakControl:
                noslip = DirichletBC(V, zero, walls)
                inlet = DirichletBC(V, g_theta, inflow)
                bcs_u = [inlet, noslip]
                bc_p = DirichletBC(Q, 0, outflow)

        elif meshD == '3D':
            facet_marker = self.problem.facet_marker
            inflow = 1
            outflow = 3
            walls = 2

            if not weakControl:
                bc_inlet = DirichletBC(V, g_theta, facet_marker, inflow)
                bc_walls = DirichletBC(V, Constant((0, 0, 0)), facet_marker, walls)
                bcs_u = [bc_inlet, bc_walls]
                bc_p = DirichletBC(Q, Constant(0), facet_marker, outflow)

        # Assemble the system for the tentative velocity step
        A1 = assemble(a1)
        b1 = assemble(L1)

        # -----------------------------# SET-UP FOR STEP 2: COMPUTING PRESSURE UPDATE #-----------------------------#
        p = TrialFunction(Q)
        q = TestFunction(Q)

        a2 = inner(grad(p), grad(q)) * dX
        L2 = -weight_time * inner(div(u_s), q) * dX

        A2 = assemble(a2)
        b2 = assemble(L2)

        # -----------------------------# SET-UP FOR STEP 3: UPDATE VELOCITY #-----------------------------#
        a3 = inner(u, v) * dX
        L3 = (inner(u_s, v) - 1 / weight_time * inner(grad(p_c), v)) * dX
        A3 = assemble(a3)
        b3 = assemble(L3)

        # -----------------------------# CREATION OF SOLVER #-----------------------------#
        if BDF2:
            solver1 = KrylovSolver(A1, 'gmres', 'jacobi')
        solver2 = KrylovSolver(A2, 'minres', 'hypre_amg')
        solver3 = KrylovSolver(A3, 'cg', 'sor')

        t0 = time.time()
        c = 0

        # Initial condition saving if required
        yield t_range[0], u_h, p_h, g_theta

        # Loop through each time step
        for t in t_range[1:]:
            c += 1
            t = np.round(t, 8)

            # -----------------------------# INCREASE TIME AND UPDATE #-----------------------------#
            if not stationary:
                g_theta.assign(g[t])

            # -----------------------------# STEP 1: TENTATIVE VELOCITY STEP #-----------------------------#
            b1 = assemble(L1)
            if not BDF2:
                A1 = assemble(a1)
                solver1 = KrylovSolver(A1, 'gmres', 'jacobi')
            [bc.apply(A1, b1) for bc in bcs_u]
            solve(A1, u_s.vector(), b1, 'gmres', 'jacobi')

            # -----------------------------# STEP 2: PRESSURE CORRECTION STEP #-----------------------------#
            b2 = assemble(L2)
            bc_p.apply(A2, b2)
            solve(A2, p_c.vector(), b2, 'minres', 'hypre_amg')

            # Update pressure
            p_h.vector().axpy(1.0, p_c.vector())

            # Assign new velocity old value
            u_old.assign(u_h)

            # -----------------------------# STEP 3: VELOCITY CORRECTION STEP #-----------------------------#
            b3 = assemble(L3)
            solve(A3, u_h.vector(), b3, 'cg', 'sor')

            print("--> at time t =", t, ": \nu_h max is ", u_h.vector().max())
            print("g_theta:", g_theta.vector().max())
            print("Worst possible Courant number=", (dt * (u_h.vector().max())) / mesh.hmin())

            yield t, u_h, p_h, g_theta

#Hdf.close()
