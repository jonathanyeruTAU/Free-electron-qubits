from typing import Callable, Tuple
from numpy.fft import fft2
import numpy as np
import matplotlib.pyplot as plt


def get_fourier_transform(func: Callable, x_lim: Tuple[float, float], y_lim: Tuple[float, float], num_points: int):
    x_start, x_end = x_lim
    x = np.linspace(x_start, x_end, num_points)

    y_start, y_end = y_lim
    y = np.linspace(y_start, y_end, num_points)

    X, Y = np.meshgrid(x, y)

    vectorized_func = np.vectorize(func)

    Z = vectorized_func(X, Y)
    print(Z)
    Z_fft = fft2(Z)

    abs_z_fft = np.abs(Z_fft)
    print(abs_z_fft)
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, abs_z_fft, cmap='viridis')
    plt.show()