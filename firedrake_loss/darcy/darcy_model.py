"""
Steady state Darcy flow model. 
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as fd
import numpy as np
import torch
from petsc4py import PETSc
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


class DarcyModel:
    def __init__(
            self,
            dx = 1., # Domain width (m)
            dy = 1., # Domain height (m)
            nx = 50, # X res
            ny = 50, # Y res
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

        self.scale = 1. / ((self.dx / float(self.nx)) * (self.dy / float(self.ny)))

        ### Mesh and function spaces
        #===========================================

        self.mesh = mesh = fd.RectangleMesh(nx, ny, dx, dy)
        self.V = V = fd.FunctionSpace(mesh, "CG", 2)
        # Test function
        w = self.w = fd.TestFunction(V)
        # Pressure
        self.u = u = fd.Function(V)
        # Source term
        self.f = f = fd.Function(V, name='source')
        f.interpolate(fd.Constant(1.))
        # Permeability 
        self.k = k = fd.Function(V, name='permeability')
        # Boundary conditions
        bc0 = fd.DirichletBC(V, 0., self.LEFT)
        bc1 = fd.DirichletBC(V, 0., self.BOTTOM)
        bc2 = fd.DirichletBC(V, 0., self.RIGHT)
        bc3 = fd.DirichletBC(V, 0., self.TOP)
        self.bcs = [bc0, bc1, bc2, bc3]

        ### Variational problem and solver
        #===========================================

        self.F = F = k*fd.dot(fd.grad(u),fd.grad(w))*fd.dx - f*w*fd.dx + u**2*w*fd.ds
        problem = self.problem = fd.NonlinearVariationalProblem(F, u, bcs=[bc0, bc1, bc2, bc3])
        solver = self.solver = fd.NonlinearVariationalSolver(problem)
        self.R = fd.sqrt((fd.div(k * fd.grad(u)) - f)**2 + fd.Constant(1e-10))*fd.dx + u**2*fd.ds

        

        ### Index map from dofs to pixels
        #===========================================


        x = fd.SpatialCoordinate(self.mesh)
        x0 = fd.interpolate(x[0], self.V).dat.data
        y0 = fd.interpolate(x[1], self.V).dat.data
        self.indexes = np.lexsort((x0, y0))

        ### Jacobian 
        #===========================================

        self.J = fd.derivative(self.F, self.u)
        self.J_strong = fd.derivative(self.R, self.u)
        # Stores r^T J product
        self.rJ = fd.Function(self.V)
        self.J_strong_func = fd.Function(self.V)



    """
    Takes in a CG field and a 2d array of values, setting the CG field dofs appropriately.
    """
    def set_field(self, field, vals):
        field.dat.data[self.indexes] = vals.flatten() 


    """
    Converts CG dofs to a 2D image array. 
    """
    def get_field(self, field):
        return field.dat.data[self.indexes].reshape(2*self.nx+1, 2*self.ny+1)


    """
    Solve flow equations.
    """
    def solve(self):
        # Solve
        self.solver.solve()


    def batch_solve(self, N = 10):
        K = self.get_permeability(N)


    """
    Generate random permeability fields.
    """
    def get_permeability(self, N=50, sigma=10., kmin=1e-2, kmax=1.):

        # Smooth some noise
        noise = np.random.randn(N, 2*self.nx+1, 2*self.ny+1)
        K = gaussian_filter(noise, sigma, axes=(1,2))

        # Threshold
        K[K < 0.] = 0.
        K[K > 0.] = 1.
        # Remap values
        K = K*(kmax - kmin) + kmin
        K = torch.tensor(K, dtype=torch.float32)

        return K


"""
Pinn loss function. Accepts 2D permeability and pressure.
"""
class WeakLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, k, model):

        ctx.model = model 
        ctx.k = k
        ctx.u = u

        model.set_field(model.k, k)
        model.set_field(model.u, u)

        #r = torch.tensor(fd.assemble(model.F, bcs=model.bcs).dat.data[:], dtype=torch.float32) 
        r = torch.tensor(fd.assemble(model.F, bcs=model.bcs).dat.data[:], dtype=torch.float32) 
        return 0.5*(r**2).sum()

    @staticmethod
    def backward(ctx, grad_output):

        model = ctx.model  
        k = ctx.k
        u = ctx.u

        model.set_field(model.k, k)
        model.set_field(model.u, u)

        # Compute r^T J
        #r = fd.assemble(model.F, bcs=model.bcs)
        #J = fd.assemble(model.J, bcs=model.bcs).M.handle

        r = fd.assemble(model.F)
        J = fd.assemble(model.J).M.handle

        with r.dat.vec_ro as r_p:
            with model.rJ.dat.vec as rJ:
                J.multTranspose(r_p, rJ)

        # Get r^T J as image
        rJ = model.get_field(model.rJ)

        out = torch.tensor(rJ, dtype=torch.float32) 
        return out*grad_output, None, None, None
    
class StrongLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, k, model):

        ctx.model = model 
        ctx.k = k
        ctx.u = u

        model.set_field(model.k, k)
        model.set_field(model.u, u)

        r = torch.tensor(fd.assemble(model.R), dtype=torch.float32) 
        return r

    @staticmethod
    def backward(ctx, grad_output):

        model = ctx.model  
        k = ctx.k
        u = ctx.u

        model.set_field(model.k, k)
        model.set_field(model.u, u)

        J = torch.tensor(fd.assemble(model.J_strong).dat.data[:], dtype=torch.float32)
        #out = 
        return  J*grad_output, None, None, None