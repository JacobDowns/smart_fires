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
from advection_model import AdvectionModel, PDELoss, DataLoss

# Solve PDE, save inputs and outputs

np.random.seed(35532)
m = AdvectionModel()

# Define source term
f = np.ones((2*m.nx+1, 2*m.ny+1))
ys = np.linspace(0.,1.,f.shape[1])
f *= np.sin(2*ys*2.*np.pi)[:,np.newaxis]
m.set_field(m.f, f)

# Potential field for advective velocity
phi = m.get_potential(1)[0]
phi -= phi.min()
m.set_field(m.phi, phi)
m.v.interpolate(fd.grad(m.phi))

# Solve
m.solver.solve()
u = m.get_field(m.u)

plt.subplot(3,1,1)
plt.imshow(phi)
plt.colorbar()

plt.subplot(3,1,2)
plt.imshow(f)
plt.colorbar()

plt.subplot(3,1,3)
plt.imshow(u)
plt.colorbar()
plt.show()

data = np.stack([phi, f, u])
np.save('solution_data/data.npy', data)