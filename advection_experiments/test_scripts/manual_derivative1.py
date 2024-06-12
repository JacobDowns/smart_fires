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
from advection_model_backup import AdvectionModel
from firedrake_adjoint import *

np.random.seed(35532)


N = 10
m = AdvectionModel(nx=100, ny=100)

v_x, v_y = m.get_velocity(N)
f = np.ones_like(v_x[0])
ys = np.linspace(0.,1.,f.shape[1])
f *= np.sin(2*ys*2.*np.pi)[:,np.newaxis]
m.set_field(m.f, f)
m.set_velocity(v_x[0], v_y[0])

#u_opt = fd.File('u_opt.pvd')
#m.solver.solve()
#u_opt.write(m.u)
#m.u.assign(0.*m.u)

m.eval_loss('var')
du = m.get_field(m.du)
plt.imshow(du)
plt.colorbar()
plt.show()

quit()

A = fd.assemble(m.a, bcs=m.bcs)
b = fd.assemble(m.L, bcs=m.bcs)
fd.solve(A, m.G_sol, b)

g = m.get_field(m.G_sol)


plt.imshow(g)
plt.colorbar()
plt.show()


#print(A)
quit()

f = fd.File('out.pvd')
for i in range(50000):

    m.G_solver.solve()

    """
    plt.subplot(2,1,1)
    plt.imshow(m.get_field(m.G))
    plt.colorbar()

    plt.subplot(2,1,2)
    plt.imshow(m.get_field(m.u))
    plt.colorbar()
    plt.show()
    """

    J = fd.assemble(m.I_u)
    print(i,J)
    du = compute_gradient(J, Control(m.u))

    #du = m.get_field(du)
    #plt.imshow(du)
    #plt.colorbar()
    #plt.show()
    #quit()

    m.u.assign(m.u - du)
    if i % 50 == 0:
        f.write(m.u, idx=i)
   

quit()
"""
m.set_field(m.u, u0)
I0 = fd.assemble(m.F_strong)
noise = 10.*np.random.randn(*u0.shape)
noise[1::2,1::2] = 0.
u0 += noise
m.set_field(m.u, u0)
I1 = fd.assemble(m.F_strong)
"""


du = fd.assemble(fd.derivative(m.F_strong, m.u), bcs=m.bcs).dat.data[:]
#du = fd.assemble(m.F_weak, bcs=m.bcs).dat.data[:]


m.rJ.dat.data[:] = du
du = m.get_field(m.rJ)

plt.subplot(3,1,1)
plt.imshow(du[::2,::2])
plt.colorbar()

plt.subplot(3,1,2)
plt.imshow(du[1::2,1::2])
plt.colorbar()

plt.subplot(3,1,3)
plt.imshow(du)
plt.colorbar()
plt.show()