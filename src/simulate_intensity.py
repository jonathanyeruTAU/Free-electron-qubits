import numpy as np
import matplotlib.pyplot as plt


def simulate_intensity_electron_wave_function(intensity_function):
    """
    :param intensity_function: expects to get x and y and returns the intensity to that point
    need to enter single values, not numpy array
    :return:
    """
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for x_row, y_row, i in zip(X, Y, range(len(X))):
        print("i", i)
        for x_value, y_value, j in zip(x_row, y_row, range(len(x_row))):
            print("j", j)
            Z[i][j] = intensity_function(x_value, y_value)

    print(Z)

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    plt.show()
