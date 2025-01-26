#-----------------------------# IMPORT LIBRARIES #-----------------------------#
import time
from dolfin import *
from dolfin_adjoint import *
import numpy as np
from utils import StationaryBoundaryFunction3D

class Coupled(object):

      def __init__(self, problem):
            self.problem = problem
            self.dx = Measure('dx', domain=problem.mesh, subdomain_data=problem.cell_ids)
            self.ds = Measure('ds', domain=problem.mesh, subdomain_data=problem.facet_marker)

            if self.problem.control == 'inlet':
                  self.dsu_ = self.ds(self.problem.inflow_id)
                  #self.dsu_ = self.ds(subdomain_data = problem.facet_marker)

      def nonlinear_run(self, T, dt, velocity_space, u_h, U_inlet, rho, nu, meshD, weakControl, nitsche_gamma, beta, theta, backflow,
                        tractionFree, simvascularstab, stationary):
            dx = self.dx
            ds = self.ds
            dsu_ = self.dsu_
            mesh = self.problem.mesh
            t_range = np.arange(0, T, dt)
            dim = mesh.topology().dim()
            zero = (0,) * dim

            #-----------------------------# MODEL PARAMS #-----------------------------#
            t = 0
            nu = Constant(nu, name="kinematic viscosity")  # Kinematic viscosity
            rho = Constant(rho, name="density")  # Density
            weakControl = weakControl
            meshD = meshD

            #-----------------------------# STAB PARAMS #-----------------------------#
            beta = Constant(beta, name="beta")
            theta = Constant(theta, name="theta")
            nitsche_gamma = Constant(nitsche_gamma, name="nitsche gamma")

            # -----------------------------# DEFINITION OF FUNCTION SPACES #-----------------------------#
            V = velocity_space
            Q = self.problem.Q
            W = self.problem.W
            control_space = V

            # -----------------------------# DEFINITION OF TRIAL AND TEST FUNCTION #-----------------------------#
            (u, p) = TrialFunctions(W)
            (v, q) = TestFunctions(W)
            U = Function(W, name='Solution', annotate=True)
            p_h = Function(Q, name='Pressure', annotate=True)
            n = FacetNormal(mesh)
            u, p = split(U)  # trial functions

            # -----------------------------# SET UP CONTROLS: FROM EXPRESSION TO FUNCTION #-----------------------------#
            g = project(U_inlet, V)

            # -----------------------------# CREATION OF BOUNDARY CONDITIONS #-----------------------------#
            if meshD == '2D':
                  inflow = 'near(x[0], 0)'
                  outflow = 'near(x[0], 200)'
                  walls = 'near(x[1], 0) || near(x[1], 41)'

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

            u_theta = theta * u + (1 - theta) * u_h

            #-----------------------------# DEFINITION OF STRAIN RATE TENSOR #-----------------------------#
            def epsilon(u):
                  return sym(nabla_grad(u))

            #-----------------------------# DEFINITION OF STRESS TENSOR #-----------------------------#
            def sigma(u, p):
                  return 2 * nu * epsilon(u) - p * Identity(len(u))

            #-----------------------------# VARIATIONAL FORMULATION (NON LINEAR) #-----------------------------#
            F = (
                        inner((u - u_h) / Constant(dt), v) * dx  # Transient term
                        + inner(grad(u_theta) * u_theta, v) * dx()  # Convective Term
                        + nu * inner(grad(u_theta), grad(v)) * dx  # Stress Tensor (1)
                        - inner(p, div(v)) * dx  # Stress Tensor (2)
                        - inner(q, div(u)) * dx  # Divergence Free Term
            )

            #-----------------------------# NEUMANN BOUNDARY CONDITION #-----------------------------#
            if tractionFree:
                  T = Constant((0,) * dim)
                  F -= dot(T, v) * ds()  # Neumann BC
                
            if V.ufl_element().degree() == 1:
                  h = CellDiameter(mesh)
                  F -= 0.001 * h ** 2 * inner(grad(p), grad(q)) * dx

            if backflow:
                  bf = (inner(n, u_h) - abs(inner(n, u_h))) / 2
                  b = Constant(0.2)
                  if simvascularstab:
                        F -= b * inner(bf * u_theta, v) * ds
                  else:
                        def t_grad(w, V):
                              # Returns the Tangential part of the gradient on cell facets
                              # S = w.function_space()
                              S = V
                              n = FacetNormal(S.mesh())
                              if len(w.ufl_shape) == 0:
                                    return grad(w) - n * inner(n, w)
                              elif len(w.ufl_shape) == 1:
                                    return grad(w) - outer(grad(w) * n, n)

                        F -= b * inner(bf * t_grad(u_theta, V), t_grad(v, V)) * ds

            iterative_solver = True
            # ---> initial condition
            yield t, u_h, p_h, g


            for t in t_range[1:]:
                  t = np.round(t, 8)
                  #t = t + dt

                  if not stationary:
                        U_inlet.t = t
                        g = project(U_inlet, V)


                  problem = NonlinearVariationalProblem(F, U, bcu + bcp, derivative(F, U))
                  solver = NonlinearVariationalSolver(problem)

                  prm = solver.parameters  # https://fenicsproject.discourse.group/t/problem-with-solver/96

                  prm['newton_solver']['absolute_tolerance'] = 1E-8
                  prm['newton_solver']['relative_tolerance'] = 1E-7
                  prm['newton_solver']['maximum_iterations'] = 20
                  prm['newton_solver']['error_on_nonconvergence'] = False
                  prm['newton_solver']['relaxation_parameter'] = 1.0

                  if iterative_solver:
                        prm['newton_solver']['linear_solver'] = 'bicgstab'  # bicgstab, minres, gmres, mumps
                        prm['newton_solver']['preconditioner'] = 'hypre_amg'
                        # prm['newton_solver']['krylov_solver']['preconditioner'] = 'ilu'
                        # Use nonzero guesses - essential for CG with non-symmetric BC
                        prm['newton_solver']['krylov_solver']['nonzero_initial_guess'] = True
                        prm['newton_solver']['krylov_solver']['absolute_tolerance'] = 1E-9
                        prm['newton_solver']['krylov_solver']['relative_tolerance'] = 1E-7
                        prm['newton_solver']['krylov_solver']['maximum_iterations'] = 1000

                  timer = Timer("Forward solve")

                  solver.solve()
                  timer.stop()

                  # Update previous solution
                  u, p = U.split(deepcopy=True)
                  u_h.assign(u, annotate=True)
                  p_h.assign(p, annotate=True)
                  print("Worst possible Courant number=", (dt * (u_h.vector().max())) / mesh.hmin())


                  yield t, u_h, p_h, g


      def linear_run(self, T, dt, velocity_space, u_h, g, rho, nu, meshD, nitsche_gamma, beta, theta, weakControl, tractionFree, backflow, simvascularstab, stationary):
            mesh = self.problem.mesh
            t_range = np.arange(0, T, dt)
            dt = t_range[1] - t_range[0]

            # ---> model parameters
            theta = Constant(theta, name="theta")  # SemiImplicit scheme
            nu = Constant(nu, name="kinematic viscosity")  # Kinematic viscosity
            rho = Constant(rho, name="density")  # Density
            weakControl = weakControl
            meshD = meshD

            # ---> stab parameters
            beta = Constant(beta, name="beta")
            nitsche_gamma = Constant(nitsche_gamma, name="nitsche gamma")

            V = velocity_space

            Q = self.problem.Q
            W = self.problem.W
            control_space = V

            # ---> define trial and test functions
            (u, p) = TrialFunctions(W)
            (v, q) = TestFunctions(W)
            U = Function(W, name='Solution')
            p_h = Function(Q, name='pressure')
            n = FacetNormal(mesh)

            dx = self.dx
            ds = self.ds

            # ---> create boundary conditions
            dim = mesh.topology().dim()
            zero = Constant((0,) * dim)
            f = Constant((0,) * dim)

            g_theta = Function(control_space, name='g')
            if stationary:
                  g_theta.assign(g)

            if meshD == '2D':
                  facet_marker = self.problem.facet_marker
                  walls = "on_boundary && (x[1] >= 39.9 || x[1] < 0.1)"
                  inflow = "on_boundary && x[0] <= 0.1"
                  outflow = "on_boundary && x[0] >= 199.9"
                  #inflow = 'near(x[0], 0)'
                  #outflow = 'near(x[0], 200)'
                  #walls = 'near(x[1], 0) || near(x[1], 41)'

                  if not weakControl:
                        noslip = DirichletBC(W.sub(0), zero, walls)
                        inlet = DirichletBC(W.sub(0), g_theta, inflow)
                        bcu = [inlet, noslip]
                        bcp = []
                        if not tractionFree:
                              bcp = DirichletBC(W.sub(1), 0, outflow)
                              bcp = [bcp]

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

            # ---> define time-stepping scheme
            u_theta = theta * u + (1 - theta) * u_h

            # ---> define strain-rate tensor
            def epsilon(u):
                  return sym(nabla_grad(u))

            # ---> define stress tensor
            def sigma(u, p):
                  return 2 * nu * epsilon(u) - p * Identity(len(u))

            # ---> Variational formulation (linear)
            F = (
                    inner((u - u_h) / Constant(dt), v) * dx  # Transient term
                    + inner(grad(u_theta) * u_h, v) * dx  # Convective Term (linear term)
                    + nu * inner(grad(u_theta), grad(v)) * dx  # Stress Tensor (1)
                    - inner(p, div(v)) * dx  # Stress Tensor (2)
                    - inner(q, div(u_theta)) * dx  # Divergence Free Term
            )

            # ---> Neumann boundary condition
            if tractionFree:
                  T = Constant((0,) * dim)
                  F -= dot(T, v) * ds()  # Neumann BC

            # ---> Add stabilisation term for P1-P1 element
            if V.ufl_element().degree() == 1:
                  h = CellDiameter(mesh)

                  # Brezzi-Pitkaranta
                  F -=  0.001 * h ** 2 * inner(grad(p), grad(q)) * dx

            if backflow:
                  bf = (inner(n, u_h) - abs(inner(n, u_h))) / 2
                  b = Constant(0.2)
                  if simvascularstab:
                        F -= b * inner(bf * u_theta, v) * ds
                  else:
                        def t_grad(w, V):
                              # Returns the Tangential part of the gradient on cell facets
                              # S = w.function_space()
                              S = V
                              n = FacetNormal(S.mesh())
                              if len(w.ufl_shape) == 0:
                                    return grad(w) - n * inner(n, w)
                              elif len(w.ufl_shape) == 1:
                                    return grad(w) - outer(grad(w) * n, n)

                        F -= b * inner(bf * t_grad(u_theta, V), t_grad(v, V)) * ds

            # ---> define lefthandside and righthandside
            a = lhs(F)
            b = rhs(F)

            # initial condition
            yield t_range[0], u_h, p_h, g


            for t in t_range[1:]:
                  t = np.round(t, 8)

                  problem = LinearVariationalProblem(a, b, U, bcu + bcp)
                  solver = LinearVariationalSolver(problem)
                  prm = solver.parameters
                  prm["linear_solver"] = 'bicgstab'  # gmres
                  prm["preconditioner"] = 'hypre_amg'  # hypre_amg
                  prm["krylov_solver"]["error_on_nonconvergence"] = True
                  prm["krylov_solver"]["nonzero_initial_guess"] = True
                  prm["krylov_solver"]["absolute_tolerance"] = 1E-7
                  prm["krylov_solver"]["relative_tolerance"] = 1E-4
                  prm["krylov_solver"]["maximum_iterations"] = 1000
                  # info(prm, True)

                  timer = Timer("Forward solve")

                  solver.solve()
                  timer.stop()

                  # Update previous solution
                  u, p = U.split(deepcopy=True)
                  u_h.assign(u, annotate=True)
                  p_h.assign(p)


                  yield t, u_h, p_h, g_theta


# -----------------------------# INCREMENTAL PRESSURE SPLITTING SCHEME #-----------------------------#
class IPCS(object):

      def __init__(self, problem):
            self.problem = problem
            self.dx = Measure('dx', domain=problem.mesh, subdomain_data=problem.cell_ids)
            self.ds = Measure('ds', domain=problem.mesh, subdomain_data=problem.facet_marker)

            if self.problem.control == 'inlet':
                  self.dsu_ = self.ds(self.problem.inflow_id)

      def runIPCS(self, t_range, g, nu, rho, BDF2, stationary, dt, weakControl, meshD):
            mesh = self.problem.mesh
            dim = mesh.topology().dim()
            zero = Constant((0,) * dim)
            f = Constant((0,) * dim)

            # -----------------------------# MODEL PARAMS #-----------------------------#
            t =  0
            nu = Constant(nu, name="kinematic viscosity")  # Kinematic viscosity
            rho = Constant(rho, name="density")  # Density
            mu = Constant(rho * nu, name="dynamic viscosity")

            # -----------------------------# DEFINE FUNCTION SPACES (TAYLOR-HOOD ELEMENT) #-----------------------------#
            V = self.problem.V
            Q = self.problem.Q

            # -----------------------------# SET-UP FUNCTIONS #-----------------------------#
            u_h = Function(V, name = 'velocity')
            u_s = Function(V)
            u_old = Function(V)
            # in case I want to read a pre-given u_old value
            #self.problem.read_u0()

            p_h = Function(Q, name  = 'pressure')
            p_c = Function(Q)

            dt_c = Constant(dt)

            # -----------------------------# SET-UP FOR STEP 1: COMPUTING TENTATIVE VELOCITY #-----------------------------#
            u = TrialFunction(V)
            v = TestFunction(V)

            # N.B. Full explicit, 1st order convection add_terms
            n = FacetNormal(mesh)
            h = 2.0 * Circumradius(mesh)

            weight_time = Constant(3 / (2 * dt_c)) if BDF2 else Constant(1 / dt_c)
            weight_diffusion = nu if BDF2 else Constant(0.5) * nu
            a1 = (weight_time * inner(u, v) + weight_diffusion * inner(grad(u), grad(v))) * dx
            L1 = (inner(p_h, div(v)) + inner(f, v)) * dx

            if BDF2:
                  L1 += (1 / (2 * dt_c) * inner(4 * u_h - u_old, v)
                         - 2 * inner(grad(u_h) * u_h, v)
                         + inner(grad(u_old) * u_old, v)) * dx

            else:
                  u_AB = Constant(1.5) * u_h - Constant(0.5) * u_old
                  a1 += Constant(0.5) * inner(grad(u) * u_AB, v) * dx
                  # RHS
                  L1 += (weight_time * inner(u_h, v)
                         - weight_diffusion * inner(grad(u_h), grad(v))
                         - 0.5 * inner(grad(u_h) * u_AB, v)) * dx

            # -----------------------------# BACKFLOW STABILIZATION #-----------------------------#
            bf = (inner(n, u_h) - abs(inner(n, u_h))) / 2
            b = Constant(0.2)
            ########### VERSION 1)
            a1 -= b * Constant(0.5) * inner(bf * u, v) * ds
            L1 += b * Constant(0.5) * inner(bf * u_h, v) * ds

            def t_grad(w, V):
                  # Returns the Tangential part of the gradient on cell facets
                  # S = w.function_space()
                  S = V
                  n = FacetNormal(S.mesh())
                  if len(w.ufl_shape) == 0:
                        return grad(w) - n * inner(n, w)
                  elif len(w.ufl_shape) == 1:
                        return grad(w) - outer(grad(w) * n, n)

            ########### VERSION 2)
            #a1 -= b * Constant(0.5) * inner(bf * t_grad(u, V), t_grad(v, V)) * ds
            #L1 += b * Constant(0.5) * inner(bf * t_grad(u_h, V), t_grad(v, V)) * ds

            # -----------------------------# DEFINE BOUNDARY CONDITIONS #-----------------------------#
            g_theta = Function(V, name='g')
            if stationary:
                  g_theta.assign(g)

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
                              inlet = DirichletBC(V, g_theta,inflow)
                              bcs_u = [inlet, noslip]
                              bc_p = DirichletBC(Q, 0, outflow)

            elif meshD == '3D':
                  facet_marker = self.problem.facet_marker
                  inflow = 1
                  outflow = 3
                  walls = 2

                  if not weakControl:
                        bc_inlet = DirichletBC(V, g_theta, facet_marker, inflow)
                        bc_walls = DirichletBC(V, Constant((0,0,0)), facet_marker, walls)
                        bcs_u = [bc_inlet, bc_walls]
                        bc_p = DirichletBC(Q, Constant(0), facet_marker, outflow)


            #if BDF2:
            A1 = assemble(a1)
            # Assemble L just to apply lock_inactive_dofs for A
            # lock_inactive_dofs should have additional variants
            # for only A or b
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

            #parameters["krylov_solver"]["nonzero_initial_guess"] = True
            #parameters['krylov_solver']['absolute_tolerance'] = 1E-7
            #parameters['krylov_solver']['relative_tolerance'] = 1E-7
            #parameters['krylov_solver']['maximum_iterations'] = 1000

            t0 = time.time()
            c = 0

            ########### EVENTUAL SAVING OF INITIAL CONDITION
            yield t_range[0], u_h, p_h, g_theta

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
                  #solver1.solve(u_s.vector(), b1, annotate=True)
                  solve(A1, u_s.vector(), b1, 'gmres', 'jacobi')

                  # -----------------------------# STEP 2: PRESSURE CORRECTION STEP #-----------------------------#
                  b2 = assemble(L2)
                  bc_p.apply(A2, b2)
                  #solver2.solve(p_c.vector(), b2, annotate=True)
                  solve(A2, p_c.vector(), b2, 'minres', 'hypre_amg')

                  ########### UPDATE PRESSURE
                  p_h.vector().axpy(1.0, p_c.vector())

                  ########### ASSIGN NRE u_old
                  u_old.assign(u_h)

                  # -----------------------------# STEP 3: VELOCITY CORRECTION STEP #-----------------------------#
                  b3 = assemble(L3)
                  solve(A3, u_h.vector(), b3, 'cg', 'sor')

                  print("--> at time t =", t, ": \nu_h max is ", u_h.vector().max())
                  print("g_theta:", g_theta.vector().max())
                  print("Worst possible Courant number=", (dt * (u_h.vector().max())) / mesh.hmin())


                  yield t, u_h, p_h, g_theta

#Hdf.close()