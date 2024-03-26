import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt


def gen_field(nx, ny, correlation_scale):

    # Create the smoothing kernel
    x = np.arange(-correlation_scale, correlation_scale)
    y = np.arange(-correlation_scale, correlation_scale)
    X, Y = np.meshgrid(x, y)
    dist = np.sqrt(X*X + Y*Y)
    filter_kernel = np.exp(-dist**2/(2*correlation_scale))

    # Generate random noise and smooth it
    noise = np.random.randn(nx, ny) 
    z = fftconvolve(noise, filter_kernel, mode='same')
        
    # Normalize so its in -0.5 to 0.5 range
    z -= z.min()
    z /= z.max()
    z -= 0.5

    return z


def gen_wind_field(dx, dy, nx, ny, wind_mag=10., px = 1., py=1., noise_level=10., correlation_scale=50):

    # Grid
    xs = np.linspace(0., dx, nx)
    ys = np.linspace(0., dy, ny)
    xx, yy = np.meshgrid(xs, ys)

    # Set prevailing wind direction and magnitude
    wind_field = np.zeros((nx, ny, 2))
    wind_field[:,:,0] = px
    wind_field[:,:,1] = py
    wind_field /= np.linalg.norm(wind_field, axis=-1)[:,:,np.newaxis]
    wind_field *= wind_mag

    # Add noise
    wx = gen_field(nx, ny, correlation_scale)
    wy = gen_field(nx, ny, correlation_scale)
    wind_field[:,:,0] += noise_level*wx
    wind_field[:,:,1] += noise_level*wy

    return xx, yy, wind_field