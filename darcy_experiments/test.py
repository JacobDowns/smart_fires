import hydra
from omegaconf import DictConfig
import numpy as np
import mlflow
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import MSELoss
from darcy_model import DarcyModel, PDELoss, DataLoss
import firedrake as fd
from modulus.models.fno import FNO
from modulus.distributed import DistributedManager
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from modulus.launch.utils import load_checkpoint, save_checkpoint
from modulus.launch.logging import PythonLogger, LaunchLogger, initialize_mlflow
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

#client = MlflowClient()


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


    # Get existing or create new run
    runs =  mlflow.search_runs(
        [experiment_id], 
        filter_string=f"run_name='{str(hydra_cfg.job.num)}'"
    )

    if runs.empty:
        run_id = None
    else:
        run_id = runs['run_id'].item()

    with mlflow.start_run(run_name=str(hydra_cfg.job.num), experiment_id=experiment_id, run_id=run_id) as run:
                
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

        model = Model().cuda()
        darcy_model = DarcyModel(pde_form = str(cfg.experiments.pde_form))
        pde_loss = PDELoss().apply
        data_loss = DataLoss().apply
        mse_loss = MSELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        epochs = cfg.training.epochs

        # Create checkpoint directory
        checkpoint_dir = f'checkpoints/{cfg.experiments.name}/{hydra_cfg.job.num}'
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
        loaded_epoch = load_checkpoint(device='cuda', **ckpt_args)


        # Load data
        K = np.load('darcy_data/K.npy')[:,np.newaxis,:,:]
        U = np.load('darcy_data/U.npy')[:,np.newaxis,:,:]

        U = torch.tensor(U, dtype=torch.float32).cuda()
        K = torch.tensor(K, dtype=torch.float32).cuda()

        N_train = cfg.training.training_examples
        N_validate = cfg.training.validation_examples
        K_train = K[0:N_train]
        U_train = U[0:N_train]
        K_validate = K[N_train:(N_train+N_validate)]
        U_validate = U[N_train:(N_train+N_validate)]

        dataset = TensorDataset(K_train, U_train)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

        pde_weight = cfg.pde_weight
        data_weight = 1. - pde_weight

        mlflow.log_params({
            'training_examples' : N_train,
            'validation_examples' : N_validate,
            'pde_weight' : pde_weight,
            'data_weight' : data_weight,
            'lr' : cfg.training.lr
        })

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
                    l += data_weight*data_loss(u_j, u_mod, darcy_model)
                    l += pde_weight*pde_loss(u_j, k_j, darcy_model)
                    l_avg += l

                l.backward()
                optimizer.step()
                num += 1
            
            l_avg /= num 

            # Log the loss
            mlflow.log_metrics({
               'loss' : l_avg.item()
            }, step=i) 

            print(l_avg.item())

            # Write checkpoint
            if i % checkpoint_interval == 0:
                save_checkpoint(**ckpt_args, epoch=i)

            # Validate
            if i % validation_interval == 0:
                model.eval()
                with torch.no_grad():
                
                    u = model(K_validate)
                    loss = mse_loss(u, U_validate)

                    mlflow.log_metrics({
                        "validation_loss": loss.detach()
                    }, step=i)
                
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
                    mlflow.log_figure(fig, 'val_{i}.png')


if __name__ == "__main__":
    run_experiment()