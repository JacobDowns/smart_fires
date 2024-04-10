import numpy as np
import matplotlib.pyplot as plt
from darcy_model import DarcyModel
import firedrake as fd


darcy_model = DarcyModel()
#K = darcy_model.get_permeability(1, kmin=0.2, kmax=2.)[0]
#darcy_model.set_field(darcy_model.k, K)

K, U = darcy_model.batch_solve(2000)

np.save('darcy_data/K_train.npy', K)
np.save('darcy_data/U_train.npy', U)

for i in range(10):
    plt.subplot(2,1,1)
    plt.imshow(K[i])
    plt.colorbar()

    plt.subplot(2,1,2)
    plt.imshow(U[i])
    plt.colorbar()

    plt.show()
quit()
#darcy_model.solve()


r = fd.assemble(darcy_model.F_strong, bcs=[])
print(r)
J = fd.assemble(darcy_model.J_strong, bcs=[])
quit()

#f = fd.File('a.pvd')
#f.write(darcy_model.u)

for i in range(50000):
    r = fd.assemble(darcy_model.F_strong, bcs=darcy_model.bcs)
    print(r)
    J = fd.assemble(darcy_model.J_strong, bcs = darcy_model.bcs)


    darcy_model.u.dat.data[:] -= 5e-8*J.dat.data[:]

f = fd.File('b.pvd')
f.write(darcy_model.u)
