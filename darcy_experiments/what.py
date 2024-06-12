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
from darcy_model import DarcyModel, PDELoss, DataLoss

m = DarcyModel()
k = m.get_permeability(N=10, kmin=0.2, kmax=2.)[0]
#plt.imshow(k)
#plt.show()

m.set_field(m.u, k)

m.set_field(m.k, k)

J = fd.derivative(m.F_var, m.u)
j0 = fd.assemble(J, bcs=m.bcs).dat.data

m.set_field(m.rJ, j0)
vals = m.get_field(m.rJ)
plt.subplot(2,1,1)
plt.imshow(vals)
plt.colorbar()


j1 = fd.assemble(m.F_weak, bcs=m.bcs).dat.data
m.set_field(m.rJ, j1)
vals = m.get_field(m.rJ)
plt.subplot(2,1,2)
plt.imshow(vals)
plt.colorbar()
plt.show()