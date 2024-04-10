import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import MSELoss
from darcy_model import DarcyModel, Loss
import firedrake as fd
from modulus.models.fno import FNO
from modulus.distributed import DistributedManager
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from modulus.launch.utils import load_checkpoint, save_checkpoint
from modulus.launch.logging import PythonLogger, LaunchLogger, initialize_mlflow


DistributedManager.initialize()  
dist = DistributedManager()  

# initialize monitoring
log = PythonLogger(name="darcy_fno")
log.file_logging()

initialize_mlflow(
    experiment_name=f"weak pinn",
    experiment_desc=f"Training an FNO model for the Darcy problem",
    run_name=f"PINN ",
    run_desc=f"PINN",
    user_name="Jacob Downs",
    mode="offline",
)
LaunchLogger.initialize(use_mlflow=True)

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
).cpu()

darcy_model = DarcyModel()
pde_loss = WeakLoss().apply
indexes = torch.tensor(darcy_model.indexes, dtype=torch.int64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 25000


ckpt_args = {
    "path": f"./checkpoints",
    "optimizer": optimizer,
    "models": model,
}
loaded_pseudo_epoch = load_checkpoint(device=dist.device, **ckpt_args)

log_args = {
    "name_space": "train",
    "num_mini_batch": 1,
    "epoch_alert_freq": 1,
}

for i in range(epochs):
    print(i)
    model.train()

    K = darcy_model.get_permeability(20, kmin=0.1, kmax=1.)[:,np.newaxis,:,:]
    
    repeat = 50
    l_avg = torch.tensor(0.0)
    for z in range(repeat):
        optimizer.zero_grad()
        U = model(K)
        l = torch.tensor(0.)

        for j in range(len(U)):
            k_j = K[j][0]
            u_j = U[j][0]
            l += pde_loss(u_j, k_j, darcy_model)


        l.backward()
        optimizer.step()
        l_avg += l

    with LaunchLogger(**log_args, epoch=i) as logger:
        logger.log_minibatch({"loss": l_avg.detach()})


    if i % 1 == 0:
        save_checkpoint(**ckpt_args, epoch=i)
            
    if i % 1 ==  0:
        model.eval()
        with torch.no_grad():
            K = darcy_model.get_permeability(1, kmin=0.1, kmax=1.)[:,np.newaxis,:,:]
            U = model(K)

            k_j = K[0][0]
            u_j = U[0][0]

            darcy_model.set_field(darcy_model.k, k_j)
            darcy_model.solve()
            u_pde = darcy_model.get_field(darcy_model.u)


            with LaunchLogger(**log_args, epoch=i) as logger:
            
                plt.close("all")
                fig = plt.figure()

                plt.subplot(3,1,1)
                plt.imshow(k_j)
                plt.colorbar()

                plt.subplot(3,1,2)
                plt.imshow(u_pde)
                plt.colorbar()

                plt.subplot(3,1,3)
                plt.imshow(u_j)
                plt.colorbar()

                logger.log_figure(figure=fig, artifact_file=f"validation_step_{i:03d}.png")
