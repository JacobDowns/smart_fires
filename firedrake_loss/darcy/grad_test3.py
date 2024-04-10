import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import MSELoss
from darcy_model import DarcyModel, PinnLoss
import firedrake as fd


model = DarcyModel()
pde_loss = PinnLoss().apply
k = model.get_permeability(5, kmin=0.1, kmax=1.)[0]
model.set_field(model.k, k)

V0 = fd.FunctionSpace(model.mesh, 'CG', 1)
V1 = fd.FunctionSpace(model.mesh, 'CG', 2)

x = fd.SpatialCoordinate(model.mesh)
x0 = fd.interpolate(x[0], V1).dat.data
y0 = fd.interpolate(x[1], V1).dat.data
plt.scatter(x0, y0)

x1 = fd.interpolate(x[0], V0).dat.data
y1 = fd.interpolate(x[1], V0).dat.data
plt.scatter(x0, y0, color='r')
plt.show()

print(x0)
