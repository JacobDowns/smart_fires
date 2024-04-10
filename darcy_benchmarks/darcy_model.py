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
            nx = 128, # X res
            ny = 128, # Y res
            bc_val = 0., # Value on boundary
            pde_loss = 'strong', # What form of the PDE to use (weak, strong, variational)
            strong_bcs = True # Enforce strong bcs?
            w_data = 0., # Weight for data loss
            w_pde = 1., # Weight for PDE loss
            w_bc = 0. # Weight for boundary condition loss
        ):

        self.pde_loss = pde_loss
        self.strong_bcs = strong_bcs

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
        bc0 = fd.DirichletBC(V, bc_val, self.LEFT)
        bc1 = fd.DirichletBC(V, bc_val, self.BOTTOM)
        bc2 = fd.DirichletBC(V, bc_val, self.RIGHT)
        bc3 = fd.DirichletBC(V, bc_val, self.TOP)
        self.bcs = [bc0, bc1, bc2, bc3]
        # Observed (FEM) velocity used in data loss
        self.u_obs = u_obs = fd.Function(V, name='u_obs')

        ### Different loss functions for the Darcy flow equations
        #===========================================

        # Weak form
        self.F_weak = k*fd.dot(fd.grad(u),fd.grad(w))*fd.dx - f*w*fd.dx

        # Strong form
        self.F_strong = (-fd.div(k * fd.grad(u)) + f)**2 * fd.dx

        # Variational principle
        self.F_var = k*fd.Constant(0.5)*fd.dot(fd.grad(u), fd.grad(u))*fd.dx - u*f*fd.dx

        # Data loss function
        self.F_data = (u_obs - u)**2 * fd.dx

        # Strong boundary condition condition loss
        self.F_bc = (u - fd.Constant(bc_val))**2 * fd.dx

        # Weak boundary condition loss
        self.F_weak_bc = (u - fd.Constant(bc_val))**2 * w * fd.dx

        self.forms = {}
        self.forms['data'] = self.F_data
        self.forms['weak'] = self.F_weak
        self.forms['strong'] = self.F_strong
        self.forms['var'] = self.F_var
        self.forms['bc'] = self.F_bc
        self.forms['weak_bc'] = self.F_weak_bc

        # Construct the form to use for the loss function
        self.loss_form = fd.Constant(w_pde)*self.forms[pde_loss]

        if self.pde_loss == 'weak':
           

            if self.w_bc 

  
        ### Solver 
        #===========================================

        # Traditional solver
        problem = self.problem = fd.NonlinearVariationalProblem(self.F_weak, u, bcs=[bc0, bc1, bc2, bc3])
        solver = self.solver = fd.NonlinearVariationalSolver(problem)


        ### Index map from dofs to pixels
        #===========================================

        x = fd.SpatialCoordinate(self.mesh)
        x0 = fd.interpolate(x[0], self.V).dat.data
        y0 = fd.interpolate(x[1], self.V).dat.data
        self.indexes = np.lexsort((x0, y0))


        ### Jacobian
        #===========================================


        self.derivatives = {}
        self.derivatives['weak'] = self.J_weak
        self.derivatives['weak_bc'] = self.J_weak_bc
        self.derivatives['strong'] = self.J_strong
        self.derivatives['strong_bc'] = self.J_strong_bc
        self.derivatives['var'] = self.J_var
        self.derivatives['var_bc'] = self.J_var_bc

        # Stores r^T J product
        self.rJ = fd.Function(self.V)


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
        K = self.get_permeability(N, kmin=0.2, kmax=2., sigma=20.)
        U = np.zeros_like(K)
        print(U.shape)

        for i in range(N):
            print(i)
            self.set_field(self.k, K[i])
            self.solve()
            u = self.get_field(self.u)
            U[i,:,:] = u

        return K, U

    

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
class Loss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, u_obs, k, model):

        ctx.model = model 
        ctx.k = k
        ctx.u = u

        model.set_field(model.k, k)
        model.set_field(model.u, u)

        form_name = model.form
        bcs = []
        if not model.strong_bcs:   
            bcs = model.bcs
            form_name += '_bc'

        R = model.forms[form_name]
    
        if model.form == 'weak':
            r = torch.tensor(fd.assemble(R, bcs=bcs).dat.data[:], dtype=torch.float32) 
            r = 0.5*(r**2).sum()
        else: 
            r = torch.tensor(fd.assemble(R, bcs=bcs), dtype=torch.float32) 
     
        return r

    @staticmethod
    def backward(ctx, grad_output):

        model = ctx.model  
        k = ctx.k
        u = ctx.u

        model.set_field(model.k, k)
        model.set_field(model.u, u)

        form_name = model.form
        bcs = []
        if not model.strong_bcs:   
            bcs = model.bcs
            form_name += '_bc'

        R = model.forms[form_name]
        J = model.derivatives[form_name]

        if model.form == 'weak':
            r = fd.assemble(R, bcs=bcs)
            J = fd.assemble(J, bcs=bcs).M.handle

            with r.dat.vec_ro as r_p:
                with model.rJ.dat.vec as rJ:
                    J.multTranspose(r_p, rJ)
        else:
            fd.assemble(J, bcs=bcs, tensor=model.rJ)
        
        rJ = model.get_field(model.rJ)
        out = torch.tensor(rJ, dtype=torch.float32) 
        return out*grad_output, None, None, None