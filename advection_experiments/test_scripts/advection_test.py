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

for i in range(N):
    m.set_velocity(v_x[i], v_y[i])
    m.solver.solve()
    u = m.get_field(m.u)
    plt.imshow(u)
    plt.colorbar()
    plt.show()
