import os
import sys
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.append('../model/')
from advection_model import AdvectionModel, PDELoss
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



@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run_experiment(cfg: DictConfig) -> None:


    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()

    # Get existing or create new experiment id
    if mlflow.get_experiment_by_name(cfg.experiments.name) is None:
        experiment_tags = {
            "mlflow.note.content": cfg.experiments.description,
        }   

        experiment_id  = mlflow.create_experiment(
            name=cfg.experiments.name,
            tags=experiment_tags
        )
    else:
        experiment_id = mlflow.get_experiment_by_name(cfg.experiments.name).experiment_id


    
    run_id = None

    with mlflow.start_run(run_name=str(0), experiment_id=experiment_id, run_id=run_id) as run:
                
    
        ### Setup the physics model
        #################################################################

        np.random.seed(1232431)
        physics_model = AdvectionModel()
        
        # Randomly generate velocity
        v_x, v_y = physics_model.get_velocity(1)
        v_x = v_x[0]
        v_y = v_y[0]
        physics_model.set_velocity(v_x, v_y)

        # Source term
        f = np.ones_like(v_x)
        ys = np.linspace(0.,1.,f.shape[1])
        f *= np.sin(2*ys*2.*np.pi)[:,np.newaxis]
        physics_model.set_field(physics_model.f, f)

        # Permeability
        k = np.ones_like(f)*1e-2
        physics_model.set_field(physics_model.k, k)
        physics_model.solver.solve()
        u_opt = physics_model.get_field(physics_model.u)

        # All inputs
        X = np.stack([
            k, v_x, v_y, f
        ])
        X = X[np.newaxis,:,:,:]
        X = torch.tensor(X, dtype=torch.float32).cuda()


        ### FNO Model
        #################################################################

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fno = FNO(
                    in_channels=4,
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

        model = Model().cuda()

        # Training loop
        #################################################################

        pde_loss = PDELoss().apply
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.experiments.lr)
        epochs = cfg.training.epochs

        # Create checkpoint directory
        checkpoint_dir = f'checkpoints/{cfg.experiments.name}'
        print(checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        checkpoint_interval = cfg.training.checkpoint_interval
        validation_interval = cfg.training.validation_interval

        ckpt_args = {
            "path": f"./{checkpoint_dir}",
            "optimizer": optimizer,
            "models": model,
        }

        loaded_epoch = 0 
        #loaded_epoch = load_checkpoint(device='cuda', **ckpt_args)
        
        print(cfg)

        model.train()
        for i in range(max(1, loaded_epoch + 1), epochs + 1):

            print('Epoch', i)
            optimizer.zero_grad()
            u = model(X).cpu()

            loss = pde_loss(u[0,0], k, v_x, v_y, physics_model, cfg.experiments.pde_form)
            loss.backward()
            optimizer.step()
        
            # Log the loss
            metrics = {
                'loss' : loss.item()
            }
            mlflow.log_metrics(metrics, step=i) 
            print(metrics)

            if i % validation_interval == 0:
                with torch.no_grad():
                    model.eval()
                    u = model(X).cpu()[0,0]
                    error = torch.sqrt((u - u_opt)**2).mean()

                    mlflow.log_metrics({
                        "validation_loss": error.item()
                    }, step=i)

                    plt.close("all")
                    fig = plt.figure()

                    plt.subplot(3,1,1)
                    plt.title('U (FEM)')
                    plt.imshow(u_opt)
                    plt.colorbar()
                        
                    plt.subplot(3,1,2)
                    plt.title('U (FNO)')
                    plt.imshow(u)
                    plt.colorbar()

                    plt.subplot(3,1,3)
                    plt.title('Error')
                    plt.imshow(u - u_opt)
                    plt.colorbar()

                    plt.tight_layout()
                    mlflow.log_figure(fig, f'val_{i}.png')
                 

            # Write checkpoint
            if i % checkpoint_interval == 0:
                save_checkpoint(**ckpt_args, epoch=i)
    
       

if __name__ == "__main__":
    run_experiment()
