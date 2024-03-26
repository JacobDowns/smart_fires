#!/home/john/opt/firedrake/bin/python3
# -*- coding: utf-8 -*-
"""
Solves the unsteady state advection-diffusion-reaction problem, using SUPG

Strong form (SF):

          dc/dt + div(c*u) = div(D*grad(c)) - k*c^n + f

The problem is either advection- or diffusion-dominated, depending on the ratio
u*h/D, where h is the characteristic length scale.

Weak form:

Find c in W, such that,

        (w, dc/dt) + (w, u.grad(c)) + (grad(w),D*grad(c)) 
                   + (w, k*c^n) + SUPG = (w, f)            on Omega

where,
        SUPG  = (grad(w),tau*res*u)
          res = dc/dt + (w, u.grad(c)) + (grad(w),D*grad(c)) 
                   + (w, k*c^n) - (w, f)
          tau = h_mesh/(2*||u||)

for all w in W'.

"""

# %%
# 0) importing modules
import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt

class Model:
    def __init__(
            self,
            dx = 100., # Domain width (km)
            dy = 100., # Domain height (km)
            nx = 100, # X res
            ny = 100, # Y res
            Dm = 5e-5,  # diffusion (m²/s)
            d_l = 0.0, # longitidinal dispersion (km)
            d_t = 0.0, # transversal dispersion (km)
            K = 0.00, # reaction rate (mol/m³/km)
            order = 1, # FEM element order
            quad_mesh = True # Use quadrilateral mesh
        ):

        ### Domain variables
        #===========================================

        self.LEFT = 1
        self.RIGHT = 2
        self.BOTTOM = 3
        self.TOP = 4

        self.dx = dx
        self.dy = dy
        self.nx = nx
        self.ny = ny

        ### Mesh and function spaces
        #===========================================

        self.mesh = mesh = fd.RectangleMesh(nx, ny, dx, dy, quadrilateral=quad_mesh)
        self.X = X = fd.FunctionSpace(mesh, "CG", order)
        self.V = V =  fd.VectorFunctionSpace(mesh, "CG", order)
        self.DG0 = DG0 = fd.FunctionSpace(mesh, "DG", 0)
        w = self.w = fd.TestFunction(X)
        # Concentration
        self.c0 = c0 = fd.Function(X, name="c")
        self.c = c = fd.Function(X)
        # Source term
        self.fc = fc = fd.Function(X, name='source')
        # Advective velocity
        self.vel = vel = fd.Function(V, name='velocity')
        self.vnorm = vnorm = fd.sqrt(fd.dot(vel, vel))
        # Time step (hours)
        dt = 1.

        ### Define advection / diffusion / reaction equations
        #===========================================

        # Diffusion tensor
        Diff = fd.Identity(mesh.geometric_dimension())*(Dm + d_t*vnorm) + \
            fd.Constant(d_l-d_t)*fd.outer(vel, vel)/vnorm
        Diff = fd.Identity(mesh.geometric_dimension())*Dm

        self.Dt = Dt = fd.Constant(dt)
        self.K = K = fd.Constant(K)
        # Crank-Nicolson timestepping
        c_mid = 0.5 * (c + c0)  

        # weak form (transport)
        F = w*(c - c0)*fd.dx + Dt*(w*fd.dot(vel, fd.grad(c_mid))
                                + fd.dot(fd.grad(w),
                                            Diff*fd.grad(c_mid)) + w*K*c_mid
                                - fc*w)*fd.dx  # - Dt*h_n*w*fd.ds(outlet)

        # strong form
        R = self.R =  (c - c0) + Dt*(fd.dot(vel, fd.grad(c_mid)) -
                        fd.div(Diff*fd.grad(c_mid)) + K*c_mid - fc)


        # SUPG stabilisation parameters
        self.h = h = fd.sqrt(2)*fd.CellVolume(mesh) / fd.CellDiameter(mesh)
        # h = fd.CellSize(mesh)
        tau = h / (2.0 * vnorm)
        # Residual and stabilizing terms
        F += tau*fd.dot(vel, fd.grad(w)) * R * fd.dx
        self.F = F

        c0.assign(0.)
        Dt.assign(dt)

        # Boundary conditions
        bc0 = fd.DirichletBC(X, 0., self.LEFT)
        bc1 = fd.DirichletBC(X, 0., self.BOTTOM)
        bc2 = fd.DirichletBC(X, 0., self.RIGHT)
        bc3 = fd.DirichletBC(X, 0., self.TOP)


        ### Variational problem and solver
        #===========================================

        problem = self.problem = fd.NonlinearVariationalProblem(F, c, bcs=[bc0, bc1, bc2, bc3])
        solver = self.solver = fd.NonlinearVariationalSolver(problem)

        ### Jacobian 
        #===========================================
        self.J = fd.derivative(self.F, self.c)
        # Stores residual transpose Jacobian product 
        self.rJ = fd.Function(self.X)



    def get_dt(self):
        dt = 0.07*fd.assemble(fd.interpolate(self.h/self.vnorm, self.DG0)).dat.data.min()
        return dt

    def step(self, dt=60.*60.):

        self.Dt.assign(dt)

        # Solve
        self.solver.solve()
        #res = fd.norm(fd.interpolate(self.R, self.X))
        #print('Residual = {}'.format(res))

        # Update solution
        self.c0.assign(self.c)