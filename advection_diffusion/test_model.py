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
nx = 100
ny = 100

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

out = fd.File('plots/out.pvd')

while t < T:
    print(i, t)
    model.step(dt)
    t += dt
    i += 1

    if i % 25 == 0:
        out.write(model.c, time=t)

