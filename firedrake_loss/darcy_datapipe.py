
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, Union
import numpy as np
import torch
from modulus import Datapipe, DatapipeMetaData
from darcy.darcy_model import DarcyModel
from scipy.ndimage import gaussian_filter

Tensor = torch.Tensor

# Get thresholded permeability field
def get_permeability(dx, dy, nx, ny, N=50, correlation_scale=25, kmin=1e-2, kmax=1.):

    # Grid
    xs = np.linspace(0., dx, nx)
    ys = np.linspace(0., dy, ny)
    xx, yy = np.meshgrid(xs, ys)

    # Smooth some noise
    noise = np.random.randn(nx, ny, N)
    K = gaussian_filter(noise, 10., axes=(0,1))

    # Trheshodld
    K[K < 0.] = 0.
    K[K > 0.] = 1.

    K = K*(kmax - kmin) + kmin

    return xx, yy, K

@dataclass
class MetaData(DatapipeMetaData):
    name: str = "Darcy2D"
    # Optimization
    auto_device: bool = False
    cuda_graphs: bool = False
    # Parallel
    ddp_sharding: bool = False

class Darcy2D(Datapipe):
 
    def __init__(
        self,
        dx: float = 1.,
        dy: float = 1.,
        nx: int = 100,
        ny: int = 100,
        batch_size: int = 50,
        kmin: float = 2.0,
        kmax: float = 0.5,
    ):
        super().__init__(meta=MetaData())

        # simulation params
        self.dx = dx
        self.dy = dy
        self.nx = nx
        self.ny = ny
        self.batch_size = batch_size
        self.kmin = kmin
        self.kmax = kmax

        self.model = DarcyModel(dx, dy, nx, ny, quad_mesh=True)


    def __iter__(self) -> Tuple[Tensor]:
        """
        Yields
        ------
        Iterator[Tuple[Tensor, Tensor]]
            Infinite iterator that returns a batch of (permeability, darcy pressure)
            fields of size [batch, resolution, resolution]
        """
        # infinite generator
        while True:
            # Generate permeability fields.
            K = get_permeability(
                self.dx,
                self.dy,
                self.nx,
                self.ny,
                self.batch_size,
                correlation_scale=25,
                kmin=self.kmin,
                kmax = self.kmax
            )

            self.K = torch.tensor(K, dtype=torch.float32)
            yield {"permeability": self.K}

    def __len__(self):
        return sys.maxsize
