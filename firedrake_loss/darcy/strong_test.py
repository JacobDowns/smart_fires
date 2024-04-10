import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import MSELoss
from darcy_model import DarcyModel
import firedrake as fd


model = DarcyModel()
k = model.get_permeability(5, kmin=0.1, kmax=1.)[0]
model.set_field(model.k, k)

model.solve()
f0 = fd.File('k.pvd')
f0.write(model.k)

f1 = fd.File('u.pvd')
f1.write(model.u)
quit()

for i in range(5000):
    r = fd.assemble(model.R)
    print(r)
    J = fd.assemble(model.J_strong).dat.data
    model.u.dat.data[:] -= 2e-7*J


f0 = fd.File('k.pvd')
f0.write(model.k)

f1 = fd.File('u.pvd')
f1.write(model.u)

quit()

f0 = fd.File('k.pvd')
f0.write(model.k)

f1 = fd.File('u.pvd')
f1.write(model.u)


plt.subplot(3,1,1)
plt.imshow(k)
plt.colorbar()

plt.subplot(3,1,2)
k = model.get_field(model.k)
plt.imshow(k[:,:])
plt.colorbar()

u = model.get_field(model.u)
plt.subplot(3,1,3)
plt.imshow(u[:,:])
plt.colorbar()
plt.show()

quit()

x = fd.SpatialCoordinate(model.mesh)
x0 = fd.interpolate(x[0], model.V_sol).dat.data
y0 = fd.interpolate(x[1], model.V_sol).dat.data

coords = np.c_[x0, y0]
print(coords.shape)
print(np.unique(coords,axis=0).shape)

plt.scatter(x0, y0, color='k', s=2)
plt.show()
quit()

indexes = np.lexsort((x0, y0))
print(indexes)

coords = np.c_[x0, y0][indexes]
print(coords)
print(coords.shape)


f = fd.File('out.pvd')
f.write(model.u)