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

# -----------------------------# COUPLED SCHEME #-----------------------------#class Coupled(object):

    def __init__(self, problem):
        self.problem = problem
        self.dx = Measure('dx', domain=problem.mesh, subdomain_data=problem.cell_ids)
        self.ds = Measure('ds', domain=problem.mesh, subdomain_data=problem.facet_marker)

        # If the problem is controlled at the inlet, define a specific surface measure for the inlet
        if self.problem.control == 'inlet':
            self.dsu_ = self.ds(self.problem.inflow_id)
            # self.dsu_ = self.ds(subdomain_data = problem.facet_marker)  # Alternative method (commented out)

    # NONLINEAR SOLVER
    def nonlinear_run(self, T, dt, velocity_space, u_h, U_inlet, rho, nu, weakControl, nitsche_gamma, beta, theta, backflow,
                      tractionFree, simvascularstab, stationary):

        dx = self.dx
        ds = self.ds
        dsu_ = self.dsu_
        mesh = self.problem.mesh
        t_range = np.arange(0, T, dt)  # Time discretization
        dim = mesh.topology().dim()  # Spatial dimension of the problem
        zero = (0,) * dim  # Zero vector for Dirichlet BCs

        # -----------------------------# MODEL PARAMS #-----------------------------#
        t = 0  # Initial time
        nu = Constant(nu, name="kinematic viscosity")  # Define kinematic viscosity as a constant
        rho = Constant(rho, name="density")  # Define density as a constant
        weakControl = weakControl  # Control flag

        # -----------------------------# STABILIZATION PARAMETERS #-----------------------------#
        beta = Constant(beta, name="beta")  # Stabilization parameter beta
        theta = Constant(theta, name="theta")  # Time stepping parameter (theta-method)
        nitsche_gamma = Constant(nitsche_gamma, name="nitsche gamma")  # Nitsche penalty parameter

        # -----------------------------# FUNCTION SPACES #-----------------------------#
        V = velocity_space  # Velocity function space
        Q = self.problem.Q  # Pressure function space
        W = self.problem.W  # Mixed function space (velocity + pressure)
        control_space = V  # Control space is the velocity space

        # -----------------------------# TRIAL AND TEST FUNCTIONS #-----------------------------#
        (u, p) = TrialFunctions(W)  # Trial functions for velocity and pressure
        (v, q) = TestFunctions(W)  # Test functions for velocity and pressure
        U = Function(W, name='Solution', annotate=True)  # Solution function
        p_h = Function(Q, name='Pressure', annotate=True)  # Pressure function
        n = FacetNormal(mesh)  # Normal vector to the mesh facets
        u, p = split(U)  # Split solution into velocity and pressure

        # -----------------------------# CONTROL FUNCTION #-----------------------------#
        g = project(U_inlet, V)  # Project inlet velocity onto function space

        # -----------------------------# BOUNDARY CONDITIONS #-----------------------------#
        facet_marker = self.problem.facet_marker  # Facet markers for boundary conditions
        inflow = 1  # Inflow boundary ID
        outflow = 3  # Outflow boundary ID
        walls = 2  # Wall boundary ID

        if not weakControl:  # If strong control is used
            if self.problem.control == 'inlet':
                bc_inlet = DirichletBC(W.sub(0), g, facet_marker, inflow)  # Dirichlet BC at inlet
                bc_walls = DirichletBC(W.sub(0), zero, facet_marker, walls)  # No-slip BC at walls
                bcu = [bc_inlet, bc_walls]  # List of velocity BCs
                bcp = []  # List of pressure BCs
                if not tractionFree:  # If traction-free condition is not applied
                    bcp = DirichletBC(W.sub(1), Constant(0), facet_marker, outflow)  # Dirichlet BC at outflow

        u_theta = theta * u + (1 - theta) * u_h  # Time-stepping intermediate velocity

        # -----------------------------# STRAIN RATE TENSOR #-----------------------------#
        def epsilon(u):
            return sym(nabla_grad(u))  # Symmetric gradient of velocity

        # -----------------------------# STRESS TENSOR #-----------------------------#
        def sigma(u, p):
            return 2 * nu * epsilon(u) - p * Identity(len(u))  # Stress tensor

        # -----------------------------# VARIATIONAL FORMULATION (NONLINEAR) #-----------------------------#
        F = (
            inner((u - u_h) / Constant(dt), v) * dx  # Time derivative term
            + inner(grad(u_theta) * u_theta, v) * dx  # Convective term
            + nu * inner(grad(u_theta), grad(v)) * dx  # Viscous term
            - inner(p, div(v)) * dx  # Pressure term
            - inner(q, div(u)) * dx  # Incompressibility constraint
        )

        # -----------------------------# NEUMANN BOUNDARY CONDITION #-----------------------------#
        if tractionFree:
            T = Constant((0,) * dim)  # Zero traction vector
            F -= dot(T, v) * ds  # Neumann boundary condition

        if V.ufl_element().degree() == 1:  # If linear elements are used
            h = CellDiameter(mesh)  # Compute cell diameter
            F -= 0.001 * h ** 2 * inner(grad(p), grad(q)) * dx  # Pressure stabilization term

        if backflow:  # If backflow stabilization is enabled
            bf = (inner(n, u_h) - abs(inner(n, u_h))) / 2  # Backflow term
            b = Constant(0.2)  # Stabilization coefficient
            if simvascularstab:  # If SimVascular stabilization is used
                F -= b * inner(bf * u_theta, v) * ds
            else:
                def t_grad(w, V):
                    S = V
                    n = FacetNormal(S.mesh())
                    if len(w.ufl_shape) == 0:
                        return grad(w) - n * inner(n, w)
                    elif len(w.ufl_shape) == 1:
                        return grad(w) - outer(grad(w) * n, n)
                F -= b * inner(bf * t_grad(u_theta, V), t_grad(v, V)) * ds

        iterative_solver = True  # Use iterative solver
        yield t, u_h, p_h, g  # Initial condition output

        for t in t_range[1:]:  # Time-stepping loop
            t = np.round(t, 8)  # Round time value for stability
            if not stationary:
                U_inlet.t = t  # Update inlet velocity time
                g = project(U_inlet, V)  # Recompute inlet velocity function

            problem = NonlinearVariationalProblem(F, U, bcu + bcp, derivative(F, U))  # Define nonlinear problem
            solver = NonlinearVariationalSolver(problem)  # Create solver
            prm = solver.parameters  # Solver parameters
            prm['newton_solver']['absolute_tolerance'] = 1E-8
            prm['newton_solver']['relative_tolerance'] = 1E-7
            prm['newton_solver']['maximum_iterations'] = 20
            prm['newton_solver']['error_on_nonconvergence'] = False

            solver.solve()  # Solve the nonlinear problem
            u, p = U.split(deepcopy=True)  # Extract solution components
            u_h.assign(u, annotate=True)  # Update velocity field
            p_h.assign(p, annotate=True)  # Update pressure field
            yield t, u_h, p_h, g  # Output solution at current timestep

      # LINEAR SOLVER
      def linear_run(self, T, dt, velocity_space, u_h, U_inlet, rho, nu, weakControl, nitsche_gamma, beta, theta, backflow,
                        tractionFree, simvascularstab, stationary):
            mesh = self.problem.mesh
            t_range = np.arange(0, T, dt)
            dt = t_range[1] - t_range[0]  # Ensuring time step consistency

            # ---> Model parameters
            theta = Constant(theta, name="theta")  # Semi-implicit time integration
            nu = Constant(nu, name="kinematic viscosity")  # Kinematic viscosity
            rho = Constant(rho, name="density")  # Density
            weakControl = weakControl

            # ---> Stabilization parameters
            beta = Constant(beta, name="beta")
            nitsche_gamma = Constant(nitsche_gamma, name="nitsche gamma")

            V = velocity_space  # Velocity function space
            Q = self.problem.Q  # Pressure function space
            W = self.problem.W  # Mixed function space (velocity + pressure)
            control_space = V  # Control space

            # ---> Define trial and test functions
            (u, p) = TrialFunctions(W)  # Unknowns (velocity and pressure)
            (v, q) = TestFunctions(W)  # Test functions
            U = Function(W, name='Solution')  # Solution function
            p_h = Function(Q, name='Pressure')  # Pressure function
            n = FacetNormal(mesh)  # Normal vector to mesh facets

            # ---> Define inlet boundary condition
            g = project(U_inlet, V)  # Projected inlet velocity

            dx = self.dx  # Measure for volume integration
            ds = self.ds  # Measure for surface integration

            # ---> Define boundary conditions
            dim = mesh.topology().dim()
            zero = Constant((0,) * dim)  # Zero vector for no-slip conditions

            mf = self.problem.mf  # Mesh function for boundary markers
            inflow = 1  # ID for inlet boundary
            outflow = 3  # ID for outlet boundary
            walls = 2  # ID for walls

            if not weakControl:
                  if self.problem.control == 'inlet':
                        bc_inlet = DirichletBC(W.sub(0), g, mf, inflow)  # Inlet velocity BC
                        bc_walls = DirichletBC(W.sub(0), zero, mf, walls)  # No-slip BC on walls
                        bcu = [bc_inlet, bc_walls]  # Velocity BC list
                        bcp = []  # Pressure BC list
                        if not tractionFree:
                              bcp = DirichletBC(W.sub(1), Constant(0), mf, outflow)  # Pressure BC at outlet

            # ---> Define time-stepping scheme
            u_theta = theta * u + (1 - theta) * u_h  # Theta scheme for time integration

            # ---> Define strain-rate tensor
            def epsilon(u):
                  return sym(nabla_grad(u))  # Symmetric gradient

            # ---> Define stress tensor
            def sigma(u, p):
                  return 2 * nu * epsilon(u) - p * Identity(len(u))  # Navier-Stokes stress tensor

            # ---> Variational formulation (linear problem)
            F = (
                  inner((u - u_h) / Constant(dt), v) * dx  # Transient term
                  + inner(grad(u_theta) * u_h, v) * dx  # Convective term (linearized)
                  + nu * inner(grad(u_theta), grad(v)) * dx  # Viscous term
                  - inner(p, div(v)) * dx  # Pressure gradient term
                  - inner(q, div(u_theta)) * dx  # Divergence-free constraint
            )

            # ---> Apply Neumann boundary condition
            if tractionFree:
                  T = Constant((0,) * dim)  # Traction-free stress vector
                  F -= dot(T, v) * ds()  # Neumann boundary condition

            # ---> Stabilization term for P1-P1 element
            if V.ufl_element().degree() == 1:
                  h = CellDiameter(mesh)  # Mesh cell diameter
                  F -= 0.001 * h ** 2 * inner(grad(p), grad(q)) * dx  # Brezzi-Pitkaranta stabilization

            # ---> Backflow stabilization
            if backflow:
                  bf = (inner(n, u_h) - abs(inner(n, u_h))) / 2  # Backflow computation
                  b = Constant(0.2)  # Stabilization coefficient
                  if simvascularstab:
                        F -= b * inner(bf * u_theta, v) * ds  # Simple backflow stabilization
                  else:
                        def t_grad(w, V):
                              # Computes the tangential part of the gradient on cell facets
                              S = V
                              n = FacetNormal(S.mesh())
                              if len(w.ufl_shape) == 0:
                                    return grad(w) - n * inner(n, w)
                              elif len(w.ufl_shape) == 1:
                                    return grad(w) - outer(grad(w) * n, n)

                        F -= b * inner(bf * t_grad(u_theta, V), t_grad(v, V)) * ds  # Tangential gradient stabilization

            # ---> Define left-hand side and right-hand side
            a = lhs(F)  # Left-hand side of equation
            b = rhs(F)  # Right-hand side of equation

            # ---> Initial condition
            yield t_range[0], u_h, p_h, g

            # ---> Time-stepping loop
            for t in t_range[1:]:
                  t = np.round(t, 8)  # Ensure numerical stability

                  problem = LinearVariationalProblem(a, b, U, bcu + bcp)  # Define linear problem
                  solver = LinearVariationalSolver(problem)  # Define solver
                  prm = solver.parameters  # Solver parameters
                  prm["linear_solver"] = 'bicgstab'  # BiCGStab solver
                  prm["preconditioner"] = 'hypre_amg'  # AMG preconditioner
                  prm["krylov_solver"]["error_on_nonconvergence"] = True
                  prm["krylov_solver"]["nonzero_initial_guess"] = True
                  prm["krylov_solver"]["absolute_tolerance"] = 1E-7
                  prm["krylov_solver"]["relative_tolerance"] = 1E-4
                  prm["krylov_solver"]["maximum_iterations"] = 1000

                  timer = Timer("Forward solve")  # Start timer
                  solver.solve()  # Solve the linear system
                  timer.stop()  # Stop timer

                  # ---> Update previous solution
                  u, p = U.split(deepcopy=True)  # Extract velocity and pressure
                  u_h.assign(u, annotate=True)  # Update velocity
                  p_h.assign(p)  # Update pressure

                  yield t, u_h, p_h, g  # Return current time step results

# -----------------------------# INCREMENTAL PRESSURE CORRECTION SPLITTING SCHEME #-----------------------------#
class IPCS(object):

    def __init__(self, problem):
        """
        Initialize the IPCS solver.

        Parameters:
        problem: Object containing the mesh and boundary conditions
        """
        self.problem = problem
        self.dx = Measure('dx', domain=problem.mesh, subdomain_data=problem.cell_ids)
        self.ds = Measure('ds', domain=problem.mesh, subdomain_data=problem.facet_marker)

        # Define the boundary integration measure for the inlet if control is applied
        if self.problem.control == 'inlet':
            self.dsu_ = self.ds(self.problem.inletID)

    def runIPCS(self, t_range, g, nu, rho, BDF2, stationary, dt, weakControl):
        """
        Run the IPCS scheme over the specified time range.

        Parameters:
        t_range : list of time steps
        g       : function specifying inlet boundary condition
        nu      : kinematic viscosity
        rho     : density
        BDF2    : boolean indicating whether to use the BDF2 scheme
        stationary : boolean indicating if the problem is stationary
        dt      : time step size
        weakControl : boolean to toggle weak boundary control
        """
        mesh = self.problem.mesh
        dim = mesh.topology().dim()
        zero = Constant((0,) * dim)
        f = Constant((0,) * dim)  # External forcing term (set to zero)

        # -----------------------------# MODEL PARAMETERS #-----------------------------#
        t = 0  # Initial time
        nu = Constant(nu, name="kinematic viscosity")
        rho = Constant(rho, name="density")
        mu = Constant(rho * nu, name="dynamic viscosity")

        # -----------------------------# DEFINE FUNCTION SPACES (TAYLOR-HOOD ELEMENT) #-----------------------------#
        V = self.problem.V  # Velocity space
        Q = self.problem.Q  # Pressure space

        # -----------------------------# SET-UP FUNCTIONS #-----------------------------#
        u_h = Function(V, name='velocity')  # Velocity field
        u_s = Function(V)  # Tentative velocity
        u_old = Function(V)  # Previous time step velocity
        p_h = Function(Q, name='pressure')  # Pressure field
        p_c = Function(Q)  # Pressure correction
        dt_c = Constant(dt)  # Time step constant

        # -----------------------------# SET-UP FOR STEP 1: COMPUTING TENTATIVE VELOCITY #-----------------------------#
        u = TrialFunction(V)
        v = TestFunction(V)

        n = FacetNormal(mesh)
        weight_time = Constant(3 / (2 * dt_c)) if BDF2 else Constant(1 / dt_c)
        weight_diffusion = nu if BDF2 else Constant(0.5) * nu

        # Variational form of tentative velocity step
        a1 = (weight_time * inner(u, v) + weight_diffusion * inner(grad(u), grad(v))) * dx
        L1 = (inner(p_h, div(v)) + inner(f, v)) * dx

        if BDF2:
            L1 += (1 / (2 * dt_c) * inner(4 * u_h - u_old, v) - 2 * inner(grad(u_h) * u_h, v)
                   + inner(grad(u_old) * u_old, v)) * dx
        else:
            u_AB = Constant(1.5) * u_h - Constant(0.5) * u_old
            a1 += Constant(0.5) * inner(grad(u) * u_AB, v) * dx
            L1 += (weight_time * inner(u_h, v) - weight_diffusion * inner(grad(u_h), grad(v))
                   - 0.5 * inner(grad(u_h) * u_AB, v)) * dx

        # -----------------------------# BACKFLOW STABILIZATION #-----------------------------#
        bf = (inner(n, u_h) - abs(inner(n, u_h))) / 2
        b = Constant(0.2)
        a1 -= b * Constant(0.5) * inner(bf * u, v) * ds
        L1 += b * Constant(0.5) * inner(bf * u_h, v) * ds

        # -----------------------------# DEFINE BOUNDARY CONDITIONS #-----------------------------#
        g_theta = Function(V, name='g')
        if stationary:
            g_theta.assign(g)

        facet_marker = self.problem.facet_marker
        inflow, outflow3, outflow4, walls = 2, 3, 4, 1

        if not weakControl:
            bc_inlet = DirichletBC(V, g_theta, facet_marker, inflow)
            bc_walls = DirichletBC(V, Constant((0, 0, 0)), facet_marker, walls)
            bcs_u = [bc_inlet, bc_walls]
            bc_outflow3 = DirichletBC(Q, Constant(0), facet_marker, outflow3)
            bc_outflow4 = DirichletBC(Q, Constant(0), facet_marker, outflow4)
            bc_p = [bc_outflow3, bc_outflow4]

        A1 = assemble(a1)
        b1 = assemble(L1)

        # -----------------------------# SOLVER SETUP #-----------------------------#
        solver1 = KrylovSolver(A1, 'gmres', 'jacobi') if BDF2 else None
        solver2 = KrylovSolver(A2, 'minres', 'hypre_amg')
        solver3 = KrylovSolver(A3, 'cg', 'sor')

        t0 = time.time()
        for t in t_range[1:]:
            t = np.round(t, 8)
            if not stationary:
                g_theta.assign(g[t])

            # STEP 1: Tentative velocity step
            b1 = assemble(L1)
            if not BDF2:
                A1 = assemble(a1)
                solver1 = KrylovSolver(A1, 'gmres', 'jacobi')
            [bc.apply(A1, b1) for bc in bcs_u]
            solve(A1, u_s.vector(), b1, 'gmres', 'jacobi')

            # STEP 2: Pressure correction step
            b2 = assemble(L2)
            [bc.apply(A2, b2) for bc in bc_p]
            solve(A2, p_c.vector(), b2, 'minres', 'hypre_amg')
            p_h.vector().axpy(1.0, p_c.vector())
            u_old.assign(u_h)

            # STEP 3: Velocity correction step
            b3 = assemble(L3)
            solve(A3, u_h.vector(), b3, 'cg', 'sor')

            print(f"--> at time t = {t}:\n\tu_h max = {u_h.vector().max()}\n\tg_theta max = {g_theta.vector().max()}\n\tWorst Courant number = {(dt * u_h.vector().max()) / mesh.hmin()}")

            yield t, u_h, p_h, g_theta
