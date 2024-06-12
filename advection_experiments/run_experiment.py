import hydra
from omegaconf import DictConfig
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import MSELoss
from advection_model import DarcyModel, PDELoss, DataLoss
import firedrake as fd
from modulus.models.fno import FNO
from modulus.distributed import DistributedManager
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from modulus.launch.utils import load_checkpoint, save_checkpoint
from modulus.launch.logging import PythonLogger, LaunchLogger, initialize_mlflow
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn


DistributedManager.initialize()  
dist = DistributedManager()  

# initialize monitoring
log = PythonLogger(name="darcy_fno")
log.file_logging()


@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def run_experiment(cfg: DictConfig) -> None:
    
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    
    # Create checkpoint directory
    checkpoint_dir = f'checkpoints/{cfg.experiment.name}/{hydra_cfg.job.num}'
    print(checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    run_name = ''

    initialize_mlflow(
        experiment_name=cfg.experiment.name,
        experiment_desc=cfg.experiment.desc,
        run_name=hydra_cfg.job.num,
        run_desc=hydra_cfg.job.num,
        user_name="Jacob Downs",
        mode="offline",
    )
    LaunchLogger.initialize(use_mlflow=True)

    a = np.zeros(5)
    np.savetxt(f'{checkpoint_dir}/a.txt', a)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fno = FNO(
            in_channels=1,
            out_channels=1,
            decoder_layers=1,
            decoder_layer_size=32,
            dimension=2,
            latent_channels=32,
            num_fno_layers=4,
            num_fno_modes=12,
            padding=9,
        ).cuda()

    def forward(self, x):
        x = self.fno(x)
        # Set boundaries to zero
        x[:,0,0,:] = 0.
        x[:,0,-1,:] = 0.
        x[:,0,:,0] = 0.
        x[:,0,:,-1] = 0.
        return x

model = Model()

darcy_model = DarcyModel()
pde_loss = PDELoss().apply
data_loss = DataLoss().apply
mse_loss = MSELoss()


optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
epochs = 25000

ckpt_args = {
    "path": f"./checkpoints",
    "optimizer": optimizer,
    "models": model,
}

#loaded_epoch = 0
loaded_epoch = load_checkpoint(device=dist.device, **ckpt_args)

if loaded_epoch == 0:
    log.success("Training started...")
else:
    log.warning(f"Resuming training from pseudo epoch {loaded_epoch+1}.")

log_args = {
    "name_space": "train",
    "num_mini_batch": 1,
    "epoch_alert_freq": 1,
}


# Load data
K = np.load('darcy_data/K.npy')[:,np.newaxis,:,:]
U = np.load('darcy_data/U.npy')[:,np.newaxis,:,:]

#K /= K.std()
#U /= U.std()

U = torch.tensor(U, dtype=torch.float32).cuda()
K = torch.tensor(K, dtype=torch.float32).cuda()

N_train = 1750
K_train = K[0:N_train]
U_train = U[0:N_train]
K_validate = K[N_train:(N_train+100)]
U_validate = U[N_train:(N_train+100)]

dataset = TensorDataset(K_train, U_train)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

for i in range(
        max(1, loaded_epoch + 1), epochs + 1
    ):




    print('Epoch', i)
    model.train()
   
    l_avg = torch.tensor(0.0)
    num = 0
    for batch, (k, u_mod) in enumerate(data_loader):
        optimizer.zero_grad()
        u = model(k).cpu()
        u_mod = u_mod.cpu()
        k = k.cpu()
        l = torch.tensor(0.)

        for j in range(len(u)):
            k_j = k[j][0]
            u_j = u[j][0]
            l += pde_loss(u_j, k_j, darcy_model)
            l += data_loss(u_j, u_mod, darcy_model)
            #l += (1./100.)*pde_loss(u_j, k_j, darcy_model)
            l_avg += l

        l.backward()
        optimizer.step()
        num += 1
       
    l_avg /= num 


    with LaunchLogger(**log_args, epoch=i) as logger:
        logger.log_epoch({"loss": l_avg.detach()})

            
    if i % cfg.training.checkpoint_frequency ==  0:
        save_checkpoint(**ckpt_args, epoch=i)

        model.eval()
        with torch.no_grad():
        
            u = model(K_validate)
            loss = mse_loss(u, U_validate)


            with LaunchLogger(**log_args, epoch=i) as logger:

                logger.log_epoch({"validation_loss": loss.detach()})
            
                plt.close("all")
                fig = plt.figure()

                plt.subplot(3,1,1)
                plt.title('Permeability')
                plt.imshow(K_validate[10,0].cpu())
                plt.colorbar()

                plt.subplot(3,1,2)
                plt.title('U (FEM)')
                plt.imshow(U_validate[10,0].cpu())
                plt.colorbar()
                
                plt.title('U (FNO)')
                plt.subplot(3,1,3)
                plt.imshow(u[10,0].cpu())
                plt.colorbar()

                plt.tight_layout()

                logger.log_figure(figure=fig, artifact_file=f"validation_step_{i:03d}.png")

