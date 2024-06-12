"""
Steady state Darcy flow model. 
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as fd
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from firedrake.petsc import PETSc

class AdvectionModel:
    def __init__(
            self,
            dx = 1., # Domain width (m)
            dy = 1., # Domain height (m)
            nx = 128, # X res
            ny = 128, # Y res
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

        self.mesh = mesh = fd.RectangleMesh(nx, ny, dx, dy)
        self.V = V = fd.FunctionSpace(mesh, "CG", 2)
        self.Q = Q = fd.VectorFunctionSpace(mesh, 'CG', 2, 2)
        # Test function
        w = self.w = fd.TestFunction(V)
        # Pressure
        self.u = u = fd.Function(V)
        # Source term
        self.f = f = fd.Function(V, name='source')
        f.interpolate(fd.Constant(1.))
        # Permeability 
        self.k = k = fd.Function(V, name='permeability')
        k.interpolate(fd.Constant(1e-2))
        # Boundary conditions
        bc = fd.DirichletBC(V, 0., 'on_boundary')
        self.bcs = bcs = [bc]
        # Observed (FEM) velocity used in data loss
        self.u_obs = u_obs = fd.Function(V, name='u_obs')
        # Advective velocity
        self.v = v = fd.Function(Q)
        self.vx = vx = fd.Function(V)
        self.vy = vy = fd.Function(V)
        self.advection = False


        ### Different loss functions for the Darcy flow equations
        #===========================================

        # Weak form
        self.F_weak = k*fd.dot(fd.grad(u),fd.grad(w))*fd.dx + fd.div(v*w)*u*fd.dx - f*w*fd.dx

        # Strong form
        self.F_strong = fd.sqrt((-k*fd.div(fd.grad(u)) - fd.dot(v,fd.grad(u)) - f)**2 + fd.Constant(1e-16))*fd.dx

        # Data loss function
        self.F_data = fd.Constant(100.)*fd.sqrt((u_obs - u)**2 + fd.Constant(1e-16))* fd.dx


        ### Variational principle
        #===========================================

        # Define form for transformed function G(u)
        #self.G = G = fd.TrialFunction(V)
        #self.G_sol = G_sol = fd.Function(V)
        #self.a = -k*fd.dot(fd.grad(G), fd.grad(w))*fd.dx
        #self.L = k*fd.dot(fd.grad(u), fd.grad(w))*fd.dx - fd.dot(v, fd.grad(u))*w*fd.dx
        self.G = G = fd.Function(V)
        self.R_G = -k*fd.dot(fd.grad(G), fd.grad(w))*fd.dx + k*fd.dot(fd.grad(u), fd.grad(w))*fd.dx - fd.dot(v, fd.grad(u))*w*fd.dx


        # Variational principle
        self.I_u =  k*fd.Constant(0.5)*fd.dot(fd.grad(G), fd.grad(G))*fd.dx - G*f*fd.dx
        #self.I_u = G*fd.dx

        self.forms = {}
        self.J_weak = fd.derivative(self.F_weak, self.u)
        self.J_strong = fd.derivative(self.F_strong, self.u)
        self.J_data = fd.derivative(self.F_data, self.u)
        #self.J_var = fd.derivative(self.I_u, self.G)


  
        ### Solvers
        #===========================================

        # Solver for u 
        problem = self.problem = fd.NonlinearVariationalProblem(self.F_weak, u, bcs=bcs)
        solver = self.solver = fd.NonlinearVariationalSolver(problem)
        

        ### Index map from dofs to pixels
        #===========================================

        x = fd.SpatialCoordinate(self.mesh)
        x0 = fd.interpolate(x[0], self.V).dat.data
        y0 = fd.interpolate(x[1], self.V).dat.data
        self.coords = np.c_[x0, y0]
        self.indexes = np.lexsort((x0, y0))

        # Stores derivative of loss function w.r.t. u
        self.du = fd.Function(self.V)
        self.R = fd.Function(self.V)
        self.Lambda = fd.Function(self.V)
        self.Delta = fd.Function(self.V)


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
    Sets the advective velocity fields from 2D image arrays.
    """
    def set_velocity(self, vx, vy):
        self.advection = True
        self.vx.dat.data[self.indexes] = vx.flatten()
        self.vy.dat.data[self.indexes] = vy.flatten()
        self.v.interpolate(fd.as_vector([self.vx, self.vy]))

    """
    Solve flow equations.
    """
    def solve(self):
        # Solve
        self.solver.solve()

    """
    Evaluate PDE loss and derivative.
    """
    def eval_loss(self, pde_form):

        if pde_form == 'weak':
            R = fd.assemble(self.F_weak, bcs=self.bcs)
            J = fd.assemble(self.J_weak, bcs=self.bcs).M.handle

            # Derivative of loss function is computed as r^T J 
            with R.dat.vec_ro as r_p:
                with self.du.dat.vec as du:
                    J.multTranspose(r_p, du)

            R = R.dat.data[:]
            R = 0.5*(R**2).sum()

        elif pde_form == 'strong':
            
            # For strong form, loss function is straightforward to compute
            R = fd.assemble(self.F_strong, bcs=self.bcs)
            J = fd.assemble(self.J_strong, bcs=self.bcs, tensor=self.du)

        elif pde_form == 'var':

            # Solve for G given u
            fd.solve(self.R_G==0, self.G, bcs=self.bcs)

            # Solve adjoint equation
            A = fd.derivative(self.R_G, self.G)
            A = fd.assemble(A, bcs=self.bcs)
            fd.solve(A, self.Lambda, fd.assemble(fd.derivative(-self.I_u, self.G), bcs=self.bcs))

            # Compute derivative of loss w.r.t. u
            R_u = fd.assemble(fd.derivative(self.R_G, self.u))
            self.product1(R_u, self.Lambda, self.du)

            # Eval loss 
            R = fd.assemble(self.I_u)    

        return R, self.du.dat.data


    """
    Compute c = b^T A. 
    """
    def product(self, A, b, c):
        A = A.M.handle
        with b.dat.vec_ro as r_p:
            with c.dat.vec as c_p:
                A.multTranspose(r_p, c_p)
        
    """
    y = A^T x
    """
    def product1(self, A, b, c):
        A = A.M.handle.transpose()
        with b.dat.vec_ro as r_p:
            with c.dat.vec as c_p:
                A.mult(r_p, c_p)
        

    """
    Get potential fields for advective velocity
    """
    def get_velocity(self, N=50, sigma=25.):
        noise_x = np.random.randn(N, 2*self.nx+1, 2*self.ny+1)
        noise_y = np.random.randn(N, 2*self.nx+1, 2*self.ny+1)
         
        noise_x = gaussian_filter(noise_x, sigma, axes=(1,2))
        noise_y = gaussian_filter(noise_y, sigma, axes=(1,2))

        v_x = 1. + 10.*noise_x 
        v_y = 35.*noise_y
        return v_x, v_y


"""
Pinn loss function. Accepts 2D diffusivity, velocity, and concentration.
"""
class PDELoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, k, v_x, v_y, model, pde_form):

        ctx.model = model 
        ctx.k = k
        ctx.u = u
        ctx.v_x = v_x
        ctx.v_y = v_y

        model.set_field(model.u, u)
        model.set_field(model.k, k)
        model.set_velocity(v_x, v_y)
        r, du = model.eval_loss(pde_form)
        ctx.du = du
        
        r = torch.tensor(r, dtype=torch.float32)
        return r

    @staticmethod
    def backward(ctx, grad_output):

        model = ctx.model  
        du = ctx.du
        model.du.dat.data[:] = du 
        du = model.get_field(model.du)
        du = torch.tensor(du, dtype=torch.float32)

        return du*grad_output, None, None, None, None, None
    

"""
Data  loss function.
"""
class DataLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, u_obs, model):

        ctx.model = model 
        ctx.u_obs = u_obs
        ctx.u = u

        model.set_field(model.u_obs, u_obs)
        model.set_field(model.u, u)
    
        r = torch.tensor(fd.assemble(model.forms['data'], bcs=model.bcs), dtype=torch.float32) 
     
        return r

    @staticmethod
    def backward(ctx, grad_output):

        model = ctx.model  
        u_obs = ctx.u_obs
        u = ctx.u

        model.set_field(model.u_obs, u_obs)
        model.set_field(model.u, u)

        J = model.derivatives['data']
        fd.assemble(J, bcs=model.bcs, tensor=model.rJ)
        
        rJ = model.get_field(model.rJ)
        out = torch.tensor(rJ, dtype=torch.float32) 
        
        return out*grad_output, None, None, None