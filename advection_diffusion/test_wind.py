import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from gen_wind import gen_wind_field
import firedrake as fd
from scipy.interpolate import LinearNDInterpolator


### Domain and mesh
#===========================================
dx = 100.
dy = 100.
nx = 100
ny = 100
quad_mesh = True
mesh = fd.RectangleMesh(nx, ny, dx, dy, quadrilateral=quad_mesh)
coords = mesh.coordinates.dat.data[:]
xs = coords[:,0]
ys = coords[:,1]


### Function spaces
#============================================

order = 1
V = fd.FunctionSpace(mesh, "CG", order)
Q = fd.VectorFunctionSpace(mesh, "CG", order)
DG0 = fd.FunctionSpace(mesh, "DG", 0)
w = fd.TestFunction(V)


### Variable definitions
#=============================================

# Wind speed (km/hour)
vel = fd.Function(Q)
# Concentration
c0 = fd.Function(V, name="c")
c = fd.Function(V)
# Source term
fc = fd.Function(V, name='source')
Dm = 2e-7                   # diffusion (km²/ hour)
d_l = 0.0                   # longitidinal dispersion (km)
d_t = 0.0                   # transversal dispersion (km)
K = 0.00                    # reaction rate (mol/km³/hour)
s = 0.0                     # source
# Seconds per hour
sph = 60.*60.
dt = sph

def set_wind():
    xx, yy, wind_field = gen_wind_field(100., 100., 100, 100, 10., 1., 1., 10., 50)
    wind_mag = np.linalg.norm(wind_field, axis=-1)

    points = list(zip(xx.flatten(), yy.flatten()))
    v_interp = LinearNDInterpolator(points, wind_field.reshape(-1, wind_field.shape[-1]))
    vel.dat.data[:] = v_interp(xs, ys)

    out = fd.File('out.pvd')
    out.write(vel)

def set_source():

    indexes0 = np.logical_and(xs > 10, xs < 20)
    indexes1 = np.logical_and(ys > 10, ys < 20)
    indexes = np.logical_and(indexes0, indexes1)
    fc.dat.data[indexes] = 1.

### Advection diffusion reaction equation definition
#=============================================