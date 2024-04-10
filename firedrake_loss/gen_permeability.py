import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

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