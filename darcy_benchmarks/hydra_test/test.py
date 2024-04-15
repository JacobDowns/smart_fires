import hydra
from omegaconf import DictConfig
import numpy as np
import os
@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def stuff(cfg: DictConfig) -> None:
    
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()

    
    checkpoint_dir = f'checkpoints/{cfg.experiment.name}/{hydra_cfg.job.num}'
    print(checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)


    a = np.zeros(5)
    np.savetxt(f'{checkpoint_dir}/a.txt', a)


if __name__ == "__main__":
    stuff()