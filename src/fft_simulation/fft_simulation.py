from typing import Callable, Tuple
import scipy
import numpy as np
import matplotlib.pyplot as plt


def double_well_potential(x, y):
    sigma = 10
    amplitude1 = 0.7
    amplitude2 = 0.3

    # Define the centers of the two Gaussians along the x-axis
    center1x = -30
    center1y = -30
    center2x = 30
    center2y = 30


    # Calculate the potentials due to the two Gaussians
    potential1 = amplitude1 * np.exp(-((x - center1x) ** 2 + (y - center1y) ** 2) / (2 * sigma ** 2))
    potential2 = amplitude2 * np.exp(-((x - center2x) ** 2 + (y - center2y) ** 2) / (2 * sigma ** 2))

    # Combine the potentials to form the double-well potential
    double_well = potential1 + potential2

    return -double_well

def get_meshgrid(x_lim: Tuple[float, float], y_lim: Tuple[float, float], num_points: int) -> Tuple[np.array, np.array]:
    x_start, x_end = x_lim
    x = np.linspace(x_start, x_end, num_points)

    y_start, y_end = y_lim
    y = np.linspace(y_start, y_end, num_points)

    X, Y = np.meshgrid(x, y)
    return X, Y


def get_meshgrid_k(x_spacing: float, y_spacing: float, num_points: int) -> Tuple[np.array, np.array]:
    freq_x = scipy.fft.fftfreq(num_points, d=x_spacing)
    freq_y = scipy.fft.fftfreq(num_points, d=y_spacing)
    K_x, K_y = np.meshgrid(freq_x, freq_y)
    return K_x, K_y


def get_fourier_transform(func: Callable, x_lim: Tuple[float, float], y_lim: Tuple[float, float], num_points: int) -> \
        Tuple[np.array, np.array, np.array]:
    X, Y = get_meshgrid(x_lim=x_lim,
                        y_lim=y_lim,
                        num_points=num_points)

    vectorized_func = np.vectorize(func)

    Z = vectorized_func(X, Y)
    Z_fft = scipy.fft.fft2(Z)
    K_x, K_y = get_meshgrid_k(x_spacing=get_spacing(x_lim, num_points),
                              y_spacing=get_spacing(y_lim, num_points),
                              num_points=num_points)
    return K_x, K_y, Z_fft

def get_spacing(lim: Tuple[float, float], num_of_points) -> float:
    return lim[1] - lim[0] / (num_of_points - 1)





def simulate(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')

    # ax.set_xlim([-0.002, 0.002])
    # ax.set_ylim([-0.002, 0.002])
    plt.show()
