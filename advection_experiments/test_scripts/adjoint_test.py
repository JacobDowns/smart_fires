"""
Steady state Darcy flow model. 
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as fd
from firedrake_adjoint import *
import numpy as np
import torch
from petsc4py import PETSc
import matplotlib.pyplot as plt
from advection_model import AdvectionModel, PDELoss, DataLoss

m = AdvectionModel()
N = 10



v_x, v_y = m.get_velocity(N)

"""
for i in range(N):
    plt.subplot(2,1,1)
    plt.imshow(v_x[i])
    plt.colorbar()

    plt.subplot(2,1,2)
    plt.imshow(v_y[i])
    plt.colorbar()

    plt.show()
"""

f = np.ones_like(v_x[0])
ys = np.linspace(0.,1.,f.shape[1])
f *= np.sin(2*ys*2.*np.pi)[:,np.newaxis]
m.set_field(m.f, f)
m.set_velocity(v_x[0], v_y[0])

fd.assemble(fd.derivative(m.F_var, m.u), tensor=m.rJ)
dU = m.get_field(m.rJ)
plt.imshow(dU)
plt.colorbar()
plt.show()
quit()

m.G_solver.solve()
G = m.get_field(m.G)
J = fd.assemble(m.I_u)
dJ_df = compute_gradient(J, Control(m.u))
m.rJ.assign(dJ_df)

plt.imshow(G)
plt.colorbar()
plt.show()

dU = m.get_field(m.rJ)
plt.imshow(dU)

out = fd.File('out.pvd')
out.write(m.rJ)

plt.colorbar()
plt.show()



# Define a functional for which we want to compute the derivative
# Here we choose J = 0.5 * ||u||^2
#J = assemble(0.5 * u**2 * dx)

# Compute the derivative dJ/df
#dJ_df = compute_gradient(J, Control(f))