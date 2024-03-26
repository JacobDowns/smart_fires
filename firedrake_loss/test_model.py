import numpy as np
import matplotlib.pyplot as plt
from gen_wind import gen_wind_field
import firedrake as fd
from model import Model
from scipy.interpolate import LinearNDInterpolator

### Domain and mesh
#===========================================

dx = 100.
dy = 100.
nx = 50
ny = 50

config = {
    'dx' : dx,
    'dy' : dy,
    'nx' : nx,
    'ny' : ny,
    'Dm' : 10.,
    'K' : 0.1
}

model = Model(**config)
coords = model.mesh.coordinates.dat.data[:]
xs = coords[:,0]
ys = coords[:,1]

def set_wind():
    xx, yy, wind_field = gen_wind_field(100., 100., 100, 100, 10., 1., 1., 10., 25)

    points = list(zip(xx.flatten(), yy.flatten()))
    v_interp = LinearNDInterpolator(points, wind_field.reshape(-1, wind_field.shape[-1]))
    model.vel.dat.data[:] = v_interp(xs, ys)

def set_source():
    indexes0 = np.logical_and(xs > 10, xs < 20)
    indexes1 = np.logical_and(ys > 10, ys < 20)
    indexes = np.logical_and(indexes0, indexes1)
    model.fc.dat.data[indexes] = 1.


set_wind()
set_source()
t = 0.
dt = model.get_dt()
T = 12.
i = 0

print(dt)


u = fd.Function(model.X)
x = fd.Function(model.X)


f = fd.File('out.pvd')


for k in range(200):
    r = fd.assemble(model.F)
    J = fd.assemble(model.J).M.handle

    print((r.dat.data**2).sum())

    with u.dat.vec as u0_p:
        with r.dat.vec_ro as r_p:
            with x.dat.vec as x_p:
                J.multTranspose(r_p, x_p)
                u0_p.axpy(-1e-3, x_p)
               

    model.c.assign(u)
    f.write(model.c, idx=k)