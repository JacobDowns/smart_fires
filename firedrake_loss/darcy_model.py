"""
Steady Darcy flow model. 
"""

import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt

class DarcyModel:
    def __init__(
            self,
            dx = 1., # Domain width (m)
            dy = 1., # Domain height (m)
            nx = 100, # X res
            ny = 100, # Y res
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
        self.V = V = fd.FunctionSpace(mesh, "CG", 1)
        # Test function
        w = self.w = fd.TestFunction(V)
        # Concentration
        self.u = u = fd.Function(V)
        # Source term
        self.f = f = fd.Function(V, name='source')
        f.interpolate(fd.Constant(1.))
        # Diffusivity 
        self.k = k = fd.Function(V, name='permeability')
        # Boundary conditions
        bc0 = fd.DirichletBC(V, 0., self.LEFT)
        bc1 = fd.DirichletBC(V, 0., self.BOTTOM)
        bc2 = fd.DirichletBC(V, 0., self.RIGHT)
        bc3 = fd.DirichletBC(V, 0., self.TOP)


        ### Variational problem and solver
        #===========================================

        #a = self.a = k * fd.dot(fd.grad(u), fd.grad(w)) * dx
        #L = self.L =  f * w * dx
        #self.F = F = a - L
        self.F = F = k*fd.dot(fd.grad(u),fd.grad(w))*fd.dx - f*w*fd.dx

        problem = self.problem = fd.NonlinearVariationalProblem(F, u, bcs=[bc0, bc1, bc2, bc3])
        solver = self.solver = fd.NonlinearVariationalSolver(problem)

        ### Jacobian 
        #===========================================
        #self.J = fd.derivative(self.F, self.c)
        # Stores residual transpose Jacobian product 
        #self.rJ = fd.Function(self.X)


    def solve(self):
        # Solve
        self.solver.solve()

