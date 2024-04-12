import hydra
from omegaconf import DictConfig

@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def stuff(cfg: DictConfig) -> None:
    print(cfg)


if __name__ == "__main__":
    stuff()