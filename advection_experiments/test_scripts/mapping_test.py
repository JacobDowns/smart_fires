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

N = 10
m = AdvectionModel(nx=100, ny=100)
v_x, v_y = m.get_velocity(N)

f0 = fd.Function(m.V)
f1 = fd.Function(m.V)
m.set_field(f0, v_x[0])

#f0 = m.get_field(f)
#plt.imshow(f0)
#plt.colorbar()
#plt.show()

out = np.array(f0(m.coords))
f1.dat.data[:] = out
out1 = m.get_field(f1)

plt.subplot(2,1,1)
plt.imshow(out1)
plt.colorbar()


plt.subplot(2,1,2)
plt.imshow(v_x[0] - out1)
plt.colorbar()
plt.show()
#m.set_field(f1, out)

