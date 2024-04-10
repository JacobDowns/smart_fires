import numpy as np
import matplotlib.pyplot as plt
import firedrake as fd
from darcy_model import DarcyModel

### Domain and mesh
#===========================================

dx = 1.
dy = 1.
nx = 100
ny = 100

config = {
    'dx' : dx,
    'dy' : dy,
    'nx' : nx,
    'ny' : ny,
}

model = DarcyModel(**config)
coords = model.mesh.coordinates.dat.data[:]
xs0 = coords[:,0]
ys0 = coords[:,1]
indexes = np.lexsort((xs0, ys0))

N = 50
K = model.get_permeability(50)

f0 = fd.File('k.pvd')
f1 = fd.File('u.pvd')
for i in range(N):
    print(i)

    model.set_field(model.k, K[i,:,:])
    model.solve()
    u = model.get_field(model.u)
    print(u)

    plt.subplot(2,1,1)
    plt.imshow(K[i,:,:])
    plt.colorbar()

    plt.subplot(2,1,2)
    plt.imshow(u)
    plt.colorbar()

    plt.show()
    f0.write(model.k, idx=i)
    f1.write(model.u, idx=i)

#coords1 = np.c_[xs1, ys1]
#print((coords[indexes] - coords1).max())
#print(coords1)

