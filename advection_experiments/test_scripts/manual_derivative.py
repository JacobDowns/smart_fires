"""
Steady state Darcy flow model. 
"""

import os
import sys
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.append('../model/')
import firedrake as fd
import numpy as np
import torch
from petsc4py import PETSc
import matplotlib.pyplot as plt
from advection_model import AdvectionModel
#from firedrake_adjoint import *

np.random.seed(35532)


N = 10
m = AdvectionModel(nx=100, ny=100)

v_x, v_y = m.get_velocity(N)
f = np.ones_like(v_x[0])
ys = np.linspace(0.,1.,f.shape[1])
f *= np.sin(2*ys*2.*np.pi)[:,np.newaxis]
m.set_field(m.f, f)
m.set_velocity(v_x[0], v_y[0])



R, du = m.eval_loss('var')

du = m.get_field(m.du)
plt.imshow(du)
plt.colorbar()
plt.show()
quit()
#A = fd.assemble(m.a, bcs=m.bcs)
A = fd.derivative(m.R_G, m.G)
A = fd.assemble(A, bcs=m.bcs)

x = fd.Function(m.V)
b = fd.assemble(fd.derivative(-m.I_u, m.G_sol), bcs=m.bcs)
fd.solve(A, x, b)



g_u = fd.assemble(fd.derivative(m.R_G, m.u))
du = fd.Function(m.V)

m.product1(g_u, x, du)

du = m.get_field(du)
plt.imshow(du)
plt.colorbar()
plt.show()

quit()

fd.solve(m.R_G==0, m.G, bcs=m.bcs)

g = m.get_field(m.G_sol)
plt.imshow(g)
plt.colorbar()
plt.show()
