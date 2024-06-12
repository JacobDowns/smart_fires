import os
import sys
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.append('../model/')
from advection_model import AdvectionModel
import hydra
from omegaconf import DictConfig
import numpy as np
import mlflow
import firedrake as fd
import matplotlib.pyplot as plt
import torch
from modulus.models.fno import FNO
from modulus.launch.utils import load_checkpoint, save_checkpoint
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

N = 1
np.random.seed(12321)
m = AdvectionModel(nx=100, ny=100)
v_x, v_y = m.get_velocity(N)

f = np.ones_like(v_x[0])
ys = np.linspace(0.,1.,f.shape[1])
f *= np.sin(2*ys*2.*np.pi)[:,np.newaxis]
m.set_field(m.f, f)
m.set_velocity(v_x[0], v_y[0])
k = np.ones_like(f)*1e-2

X = np.stack([
    k, v_x[0], v_y[0], f
])

X = torch.tensor(X, dtype=torch.float32)

print(X)
quit()


N = 1
np.random.seed(12321)
m = AdvectionModel(nx=100, ny=100)
v_x, v_y = m.get_velocity(N)

f = np.ones_like(v_x[0])
ys = np.linspace(0.,1.,f.shape[1])
f *= np.sin(2*ys*2.*np.pi)[:,np.newaxis]
m.set_field(m.f, f)
m.set_velocity(v_x[0], v_y[0])

m.solver.solve()
u = m.get_field(m.u)

m.solver.solve()
u = m.get_field(m.u)
plt.imshow(u)
plt.colorbar()
plt.show()
quit()


for i in range(50000):
    r, du = m.eval_loss('strong')
    print(i,r)
    m.u.assign(m.u - 1e-5*m.du)

