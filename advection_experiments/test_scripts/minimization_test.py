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

np.random.seed(35532)
m = AdvectionModel()

# Define source term
f = np.ones((2*m.nx+1, 2*m.ny+1))
ys = np.linspace(0.,1.,f.shape[1])
f *= np.sin(2*ys*2.*np.pi)[:,np.newaxis]
m.set_field(m.f, f)


f0 = fd.File('velocity.pvd')
f1 = fd.File('u.pvd')
alpha = fd.File('alpha.pvd')


# Potential field for advective velocity
phi = m.get_potential(1)[0]
phi -= phi.min()
phi = phi*0.
m.set_field(m.phi, phi)
m.v.interpolate(fd.grad(m.phi))

# Solve
#m.solver.solve()
f0.write(m.v)
f1.write(m.u)

u = m.get_field(m.u)
#plt.imshow(u)
#plt.colorbar()
#plt.show()
#quit() 

for i in range(50000):

    if i % 1000 == 0:
        r = fd.assemble(m.forms['var'], bcs=m.bcs)
        print(i, r)
        
    J = fd.assemble(m.derivatives['var'], bcs=m.bcs)
    m.u.dat.data[:] -= J.dat.data[:]

""""
rs = []
for i in range(50000):

    r = fd.assemble(m.forms['weak'], bcs=m.bcs)
    J = fd.assemble(m.derivatives['weak'], bcs=m.bcs).M.handle

    with r.dat.vec_ro as r_p:
        with m.rJ.dat.vec as rJ:
            J.multTranspose(r_p, rJ)

    m.u.dat.data[:] -= m.rJ.dat.data[:]
    
    r = (r.dat.data**2).sum()
    print(i, r)
    if i%1000 == 0:
        rs.append(r)
"""

plt.plot(rs)
plt.show()
f = fd.File('u_opt.pvd')
f.write(m.u)
