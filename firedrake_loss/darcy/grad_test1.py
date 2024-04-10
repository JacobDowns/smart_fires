import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import MSELoss
from darcy_model import DarcyModel, PinnLoss
import firedrake as fd
from modulus.models.fno import FNO
from modulus.distributed import DistributedManager

darcy_model = DarcyModel()
pde_loss = PinnLoss().apply
epochs = 5000
dist = DistributedManager()

model = FNO(
    in_channels=1,
    out_channels=1,
    decoder_layers=1,
    decoder_layer_size=32,
    dimension=2,
    latent_channels=32,
    num_fno_layers=4,
    num_fno_modes=12,
    padding=9,
).to(dist.device)

darcy_model = DarcyModel()
pde_loss = PinnLoss().apply
indexes = torch.tensor(darcy_model.indexes, dtype=torch.int64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

for i in range(epochs):
    print(i)
    optimizer.zero_grad()

    K = darcy_model.get_permeability(1, kmin=0.1, kmax=1.)[:,np.newaxis,:,:]
    U = model(K)

    l = torch.tensor(0.)
    for j in range(len(U)):
        
        k_j = K[j][0]
        u_j = U[j][0]

        #boundary_loss = (u_j[:,0]**2).sum() + (u_j[:,-1]**2).sum() + (u_j[0,:]**2).sum() + (u_j[-1,:]**2).sum()
        #l += 1e1*boundary_loss
        l += pde_loss(u_j, k_j, darcy_model)

    l.backward()
    optimizer.step()
    print(l)
