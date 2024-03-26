import numpy as np
import matplotlib.pyplot as plt
import firedrake as fd
from darcy_model import DarcyModel
from scipy.interpolate import LinearNDInterpolator
from gen_permeability import get_permeability

### Domain and mesh
#===========================================

dx = 1.
dy = 1.
nx = 50
ny = 50

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

xx, yy, K = get_permeability(dx, dy, nx+1, ny+1)
xs1 = xx.flatten()
ys1 = yy.flatten()

model.k.dat.data[indexes] = K[:,:,0].flatten()


f = fd.File('k.pvd')
f.write(model.k)

#coords1 = np.c_[xs1, ys1]
#print((coords[indexes] - coords1).max())
#print(coords1)



quit()


print()

model.k.interpolate(fd.Constant(1.))
model.solve()

f = fd.File('u.pvd')
f.write(model.u)
