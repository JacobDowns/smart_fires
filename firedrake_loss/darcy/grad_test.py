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


u = torch.zeros(k.shape[0], k.shape[1], requires_grad=True)
optimizer = torch.optim.Adam([u], lr=2e-8)

for i in range(50000):
    print(i)
    optimizer.zero_grad()
    loss = pde_loss(u, k, model)
    loss.backward()
    optimizer.step()

    print(loss)
    #return xi


plt.subplot(2,1,1)
plt.imshow(k)
plt.colorbar()

plt.subplot(2,1,2)
plt.imshow(u.detach().numpy())
plt.colorbar()
plt.show()

